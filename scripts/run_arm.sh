#!/usr/bin/env bash
# The arm loop: prepare data (controller update) -> train K steps -> repeat.
# Usage: MS_ROOT=... run_arm.sh adaptive|static <n_cycles> [steps_per_cycle]
set -eu
ARM=$1; CYCLES=$2; K=${3:-10}
ROOT=${MS_ROOT:?set MS_ROOT to the mathscaffold checkout}
WORK=${MS_WORK:-$ROOT/runs/$ARM}
mkdir -p $WORK
for ((c=0; c<CYCLES; c++)); do
  STEP=$(( (c + 1) * K ))
  RL=$WORK/rollouts_c$c.jsonl
  python $ROOT/scripts/prepare_cycle.py --arm $ARM --cycle $c \
      --state $WORK/ratio_state.json \
      --rollout-log $WORK/rollouts_c$((c-1)).jsonl \
      --out $WORK/train_c$c.parquet 2>&1 | tee -a $WORK/arm.log
  MATHSCAFFOLD_ROLLOUT_LOG=$RL MS_EXP=questa_$ARM \
      bash $ROOT/scripts/train_stage.sh $STEP $WORK/train_c$c.parquet \
      2>&1 | tee -a $WORK/train_c$c.log
done
