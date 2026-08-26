#!/usr/bin/env bash
# v8 = QuestA's OFFICIAL two-stage curriculum as the backbone + our teacher steering on
# top, at the winning stabilizer stack (lr 1e-5 + overlong penalty; v6 proved 2e-5 toxic
# even with the penalty). Faithful to the paper's data design:
#   stage 1 (cycles 0-9 = 100 steps): serve ONLY the OpenR1-50-0-4 rows (hard WITH a 50%
#     hint), starting dose 50; stage 2 (cycle 10+): serve ONLY the OpenR1-25-0-4 rows,
#     starting dose 25 — same switch point (step 100) as the paper.
# The teacher adjusts per-row doses from those starting points (wean successes, rescue
# all-fail, cap 50, step <=20), keeps its text notes, and the high-dose budget is
# widened to 16% (the stage-1 set alone is 14.8% of rows at r=50 by design).
# Head-to-head: v7 (teacher, hint=0 minimal-dose) vs v8 (teacher on QuestA curriculum)
# at identical lr/penalty isolates the value of the official curriculum backbone,
# 24K, 128 x n16, one update/step + QuestA's original data (no dedupe) + MINIMAL-DOSE design:
#   hint=0 start, cap 50%, steps <=20, and at most 10% of the pool above 25% dose.
# Data handling shared with v6:
#   * the ORIGINAL 12,506 rows, no dedupe (MS_NO_DEDUP=1): duplicate problems keep their
#     own reference generation, dose and state (qids base, base-1, ...);
#   * starting dose by file provenance (MS_R0_BY_SRC): rows from OpenR1-50-0-4 start at
#     r=50, rows from OpenR1-25-0-4 at r=25 — each row at the prefix ratio its difficulty
#     filter (0-4/8 correct WITH that hint) selected it for. The teacher steers from there.
#   * held-out bare-probe sets exclude every duplicate of a held-out problem (base-hash match).
#
#   cd /scratch/hongpaul-sandbox/mathscaffold && git pull
#   export MS_PYTHON=/scratch/<you>/msenv/bin/python OPENAI_API_KEY=... WANDB_API_KEY=...
#   bash scripts/launch_b200_v8.sh teacher
#
# Everything else is derived below; edit the "site" block if your paths differ.
set -eu
ARM=${1:?usage: launch_b200_v8.sh teacher|static [n_cycles]}; CYCLES=${2:-50}
case $ARM in teacher|static) ;; *) echo "arm must be teacher or static"; exit 2;; esac

# ---- site (B200 pod) ---------------------------------------------------------------
export MS_ROOT=${MS_ROOT:-/scratch/hongpaul-sandbox/mathscaffold}
export MS_PYTHON=${MS_PYTHON:?export MS_PYTHON=/scratch/<you>/msenv/bin/python first}
export MS_MODEL=${MS_MODEL:-$MS_ROOT/models/OpenMath-Nemotron-1.5B}
export MS_CKPTS=${MS_CKPTS:-$MS_ROOT/ckpts}
export MS_DATA=${MS_DATA:-$MS_ROOT/data/questa_12k/OpenR1-25-0-4.jsonl,$MS_ROOT/data/questa_12k/OpenR1-50-0-4.jsonl}
export MS_N_GPUS=${MS_N_GPUS:-8}
export MS_WANDB=1 MS_WANDB_ENTITY=${MS_WANDB_ENTITY:-mhong-university-of-minnesota} MS_WANDB_PROJECT=${MS_WANDB_PROJECT:-mathscaffold}
export WANDB_ENTITY=$MS_WANDB_ENTITY WANDB_PROJECT=$MS_WANDB_PROJECT
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"
[ "$ARM" = teacher ] && : "${OPENAI_API_KEY:?teacher arm needs OPENAI_API_KEY}"

# ---- experiment (v4) ---------------------------------------------------------------
export MS_EXP=questa_${ARM}_v8 MS_WORK=$MS_ROOT/runs/${ARM}_v8 MS_WANDB_RUN_ID=questa_${ARM}_v8
export MS_LR=1e-5            # winning stack from the scan; overlong penalty on
export MS_BETA2=0.95         # QuestA's AReaL optimizer: beta2 0.95 tracks grad-norm shifts ~50x
                             # faster than verl's default 0.999 (MS_WD=0.05 also available)
export MS_MAXRESP=24000      # QuestA training cap (bare probe follows it; official probe stays 32K)
export MS_MINI_BS=128        # one optimizer update per step (= AReaL ppo_n_minibatches 1)
export MS_BS=128 MS_N=16
export MS_NO_DEDUP=1
# v8 dosing (curriculum + teacher): EVERY row starts BARE (r=0); no mechanical dose moves at all (the old
# bare-all-fail auto-jump is deleted) — the teacher raises doses only where the model
# fails, in steps of at most MS_MAX_DELTA=20, cap 50, under a hard budget: at most
# 10% of the pool may sit above r=25 (promotions past 25 beyond that are clamped).
export MS_R0=25 MS_R_MAX=50 MS_MAX_DELTA=20 MS_PROMPT_STYLE=paper
export MS_R0_BY_SRC="OpenR1-50-0-4=50,OpenR1-25-0-4=25"
export MS_STAGE1_SRC="OpenR1-50-0-4" MS_STAGE2_SRC="OpenR1-25-0-4" MS_SWITCH_CYCLE=10
export MS_HIGH_DOSE_R=25 MS_HIGH_DOSE_FRAC=0.16
export MS_BARE_SETS=$MS_ROOT/mathscaffold/bare_probe_sets_nodedup.json
export MS_SWITCH_CYCLE=${MS_SWITCH_CYCLE:-25}   # static arm: 50% -> 25% at cycle 25 (250 steps)
export MS_STALL_MIN=60       # stage watchdog: a 24K step incl. ckpt save is 15-30 min
# soft length penalty (DAPO overlong buffer): reward falls linearly over the last MS_OVERLONG_LEN
# tokens before the cap, down to -MS_OVERLONG_PENALTY at the cap (a truncated correct answer
# then scores 1-0.5=0.5, a truncated wrong one -0.5); responses under 24000-4096=19904 tokens
# are untouched. Unset MS_OVERLONG_LEN (export MS_OVERLONG_LEN=) to run without it.
export MS_OVERLONG_LEN=${MS_OVERLONG_LEN-4096} MS_OVERLONG_PENALTY=${MS_OVERLONG_PENALTY:-0.5}
export MS_PROBE_EVERY=5 MS_BARE_PROBE=1 MS_BARE_EVERY=1

# ---- preflight ---------------------------------------------------------------------
cd $MS_ROOT
for p in $MS_PYTHON $MS_MODEL/config.json ${MS_DATA%%,*} ${MS_DATA##*,} scripts/run_arm.sh; do
  [ -e "$p" ] || { echo "[preflight] missing: $p"; exit 3; }
done
if [ -e $MS_CKPTS/$MS_EXP ] || [ -e $MS_WORK ]; then
  echo "[preflight] $MS_CKPTS/$MS_EXP or $MS_WORK already exists: this launcher is FROM SCRATCH only."
  echo "            move them aside (or resume with MS_START_CYCLE per LAUNCH_B200.md 3d) and rerun."; exit 3
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
[ "$busy" = 0 ] || { echo "[preflight] $busy GPU(s) still hold memory:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3; }
$MS_PYTHON -c "import verl, vllm; print('[preflight] verl', verl.__version__, 'vllm', vllm.__version__)"
echo "[preflight] arm=$ARM exp=$MS_EXP work=$MS_WORK lr=$MS_LR beta2=${MS_BETA2:-0.999} wd=${MS_WD:-0.01} maxresp=$MS_MAXRESP overlong=${MS_OVERLONG_LEN:-off}/${MS_OVERLONG_PENALTY} mini=$MS_MINI_BS R0=by-src(${MS_R0_BY_SRC}) stages=${MS_STAGE1_SRC}->${MS_STAGE2_SRC}@c${MS_SWITCH_CYCLE} cap=$MS_R_MAX dmax=$MS_MAX_DELTA high=${MS_HIGH_DOSE_R}@${MS_HIGH_DOSE_FRAC} nodedup=$MS_NO_DEDUP cycles=$CYCLES git=$(git log --oneline -1 | cut -c1-40)"

# ---- launch ------------------------------------------------------------------------
bash scripts/launch.sh $ARM $CYCLES
echo "[v4] verify in ~3 min:  grep -E 'Training from scratch|actor/lr' $MS_WORK/logs/latest/train_c0.log | head -3"
echo "[v4] wandb runs: $MS_EXP (trainer), ${MS_EXP}_arm (cycle metrics), ${MS_EXP}_watch (liveness)"
