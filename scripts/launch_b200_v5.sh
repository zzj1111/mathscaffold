#!/usr/bin/env bash
# v5: FROM SCRATCH (OpenMath-Nemotron-1.5B base) on one 8xB200 node, QuestA's lr 2e-5
# with a KL loss 1e-3 against the ref policy as the stabilizer (no length penalty; bare
# lr 2e-5 collapsed by length runaway at steps 65-90 in four runs — v5 tests whether the
# KL anchor alone holds it),
# 24K response cap (QuestA training cap), batch 128 x n 16, one update per step (mini 128),
# R0 = cap = 50, paper prompt, official probe every 5 cycles, bare probe every cycle.
#
#   cd /scratch/hongpaul-sandbox/mathscaffold && git pull
#   export MS_PYTHON=/scratch/<you>/msenv/bin/python OPENAI_API_KEY=... WANDB_API_KEY=...
#   bash scripts/launch_b200_v5.sh teacher      # node 1
#   bash scripts/launch_b200_v5.sh static       # node 2 (QuestA static curriculum control)
# Teacher mechanism: the new SQL-style where-ops + slow mixed-group wean (69b5a64) — make
# sure this checkout includes them (git log --oneline | head).
#
# Everything else is derived below; edit the "site" block if your paths differ.
set -eu
ARM=${1:?usage: launch_b200_v5.sh teacher|static [n_cycles]}; CYCLES=${2:-50}
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
export MS_EXP=questa_${ARM}_v5 MS_WORK=$MS_ROOT/runs/${ARM}_v5 MS_WANDB_RUN_ID=questa_${ARM}_v5
export MS_LR=2e-5            # QuestA's lr; the KL anchor below is the departure
export MS_MAXRESP=24000      # QuestA training cap (bare probe follows it; official probe stays 32K)
export MS_MINI_BS=128        # one optimizer update per step (= AReaL ppo_n_minibatches 1)
export MS_BS=128 MS_N=16
export MS_R0=50 MS_R_MAX=50 MS_PROMPT_STYLE=paper
export MS_SWITCH_CYCLE=${MS_SWITCH_CYCLE:-25}   # static arm: 50% -> 25% at cycle 25 (250 steps)
export MS_STALL_MIN=60       # stage watchdog: a 24K step incl. ckpt save is 15-30 min
# KL loss vs the frozen ref policy: gentle anchor against distribution drift (the length
# runaway is a drift phenomenon). MS_KL= (empty) disables; no overlong penalty in v5.
export MS_KL=${MS_KL-1e-3}
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
echo "[preflight] arm=$ARM exp=$MS_EXP work=$MS_WORK lr=$MS_LR maxresp=$MS_MAXRESP kl=${MS_KL:-off} overlong=${MS_OVERLONG_LEN:-off} mini=$MS_MINI_BS R0=$MS_R0 cycles=$CYCLES git=$(git log --oneline -1 | cut -c1-40)"

# ---- launch ------------------------------------------------------------------------
bash scripts/launch.sh $ARM $CYCLES
echo "[v4] verify in ~3 min:  grep -E 'Training from scratch|actor/lr' $MS_WORK/logs/latest/train_c0.log | head -3"
echo "[v4] wandb runs: $MS_EXP (trainer), ${MS_EXP}_arm (cycle metrics), ${MS_EXP}_watch (liveness)"
