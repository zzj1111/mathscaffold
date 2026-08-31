#!/usr/bin/env bash
# SAGE-bench: our auto-scaffold (GPT-5.5 teacher, per-problem solution-prefix dose R0=50
# + text scaffold/skills) run on SAGE's setup (arXiv 2602.03143) for a head-to-head:
# Qwen3-4B-Instruct on the OpenR1-Math-220k 15k subset, GRPO, lr 1e-6, batch 128 x n 4,
# 2 updates/step (mini 64), clip 0.2/0.28, KL off, 8K response cap, 500 steps.
# SAGE's own numbers for this setting (avg over 6 benchmarks): base - / GRPO +2.9 / SAGE +4.2.
#
# One-time prep on the running machine (8 GPUs assumed):
#   cd <repo> && pip install -e . (or use the packaged env; needs verl 0.7 + vllm + datasets)
#   huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
#   $MS_PYTHON scripts/prepare_sage_data.py --model models/Qwen3-4B-Instruct-2507 \
#       --out data/sage15k/openr1_sage15k.jsonl          # ~30 min; prints the MS_BARE_SETS path
# Launch:
#   export MS_PYTHON=/path/to/env/bin/python WANDB_API_KEY=... OPENAI_API_KEY=...
#   bash scripts/launch_sagebench.sh teacher            # 50 cycles x 10 = 500 steps
set -eu
ARM=${1:-teacher}; CYCLES=${2:-50}; K=${3:-${MS_K:-10}}

# ---- site --------------------------------------------------------------------------
export MS_ROOT=${MS_ROOT:-$PWD}
export MS_PYTHON=${MS_PYTHON:?export MS_PYTHON=/path/to/env/bin/python first}
export MS_MODEL=${MS_MODEL:-$MS_ROOT/models/Qwen3-4B-Instruct-2507}
export MS_CKPTS=${MS_CKPTS:-$MS_ROOT/ckpts}
export MS_DATA=${MS_DATA:-$MS_ROOT/data/sage15k/openr1_sage15k.jsonl}
export MS_BARE_SETS=${MS_BARE_SETS:-${MS_DATA%.jsonl}.bare_probe_sets.json}
export MS_N_GPUS=${MS_N_GPUS:-8}
export MS_WANDB=1 MS_WANDB_ENTITY=${MS_WANDB_ENTITY:-mhong-university-of-minnesota} MS_WANDB_PROJECT=${MS_WANDB_PROJECT:-mathscaffold}
export WANDB_ENTITY=$MS_WANDB_ENTITY WANDB_PROJECT=$MS_WANDB_PROJECT
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"
[ "$ARM" = teacher ] && : "${OPENAI_API_KEY:?teacher arm needs OPENAI_API_KEY}"

# ---- SAGE-aligned experiment -------------------------------------------------------
export MS_EXP=sagebench_${ARM}_qwen3_4b MS_WORK=$MS_ROOT/runs/sagebench_${ARM} MS_WANDB_RUN_ID=sagebench_${ARM}_qwen3_4b
export MS_LR=1e-6              # SAGE run script (their verl fork)
export MS_BS=128 MS_N=4        # n=4 for Qwen3 per the paper (length)
export MS_MINI_BS=64           # 2 updates/step, as in their script
export MS_CLIP_LOW=0.2 MS_CLIP_HIGH=0.28
export MS_MAXRESP=8192 MS_MAXPROMPT=8192   # their prompt cap is 2048 bare; hints need more
export MS_R0=50 MS_R_MAX=50 MS_PROMPT_STYLE=paper
export MS_STALL_MIN=${MS_STALL_MIN:-45}
export MS_PROBE_EVERY=5 MS_BARE_PROBE=1 MS_BARE_EVERY=1
# official probe in SAGE's protocol: temp 0.6 / top-p 0.95 / 8K, their benchmark sets
export MS_PROBE_SETS=${MS_PROBE_SETS:-aime24,aime25,math500,amc23}
export MS_PROBE_TEMP=0.6 MS_PROBE_MAXTOK=8192 MS_PROBE_MAXLEN=16384
# length penalty OFF by default (SAGE has none); MS_OVERLONG_LEN=1024 to enable if needed

# ---- preflight ---------------------------------------------------------------------
cd $MS_ROOT
for p in $MS_PYTHON $MS_MODEL/config.json $MS_DATA $MS_BARE_SETS scripts/run_arm.sh; do
  [ -e "$p" ] || { echo "[preflight] missing: $p  (data/sets: run scripts/prepare_sage_data.py first)"; exit 3; }
done
if [ -e $MS_CKPTS/$MS_EXP ] || [ -e $MS_WORK ]; then
  echo "[preflight] $MS_CKPTS/$MS_EXP or $MS_WORK already exists: from-scratch launcher; move them aside or resume per LAUNCH_B200.md 3d."; exit 3
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
[ "$busy" = 0 ] || { echo "[preflight] $busy GPU(s) still hold memory:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3; }
$MS_PYTHON -c "import verl, vllm, datasets; print('[preflight] verl', verl.__version__, 'vllm', vllm.__version__)"
$MS_PYTHON -c "
import json, sys; sys.path.insert(0, '.')
from mathscaffold import data as D
ps = D.load_problems('$MS_DATA'); assert len(ps) > 1000, f'only {len(ps)} problems loaded'
s = json.load(open('$MS_BARE_SETS')); ids = {p['qid'] for p in ps}
missing = [q for q in s['heldout'] + s['train'] if q not in ids]
print(f'[preflight] {len(ps)} problems; bare sets ok' if not missing else f'[preflight] WARN {len(missing)} probe qids not in pool')"
echo "[preflight] arm=$ARM exp=$MS_EXP model=$MS_MODEL lr=$MS_LR bs=${MS_BS}xn${MS_N} mini=$MS_MINI_BS clip=$MS_CLIP_LOW/$MS_CLIP_HIGH maxresp=$MS_MAXRESP R0=$MS_R0 cycles=$CYCLES steps/cycle=$K total=$((CYCLES*K)) git=$(git log --oneline -1 | cut -c1-40)"

# ---- launch ------------------------------------------------------------------------
bash scripts/launch.sh $ARM $CYCLES $K
echo "[sagebench] verify in ~3 min:  grep -E 'Training from scratch|actor/lr' $MS_WORK/logs/latest/train_c0.log | head -3"
echo "[sagebench] wandb: $MS_EXP (trainer), ${MS_EXP}_arm (cycle metrics + probes), ${MS_EXP}_watch (liveness)"
