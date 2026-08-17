#!/usr/bin/env bash
# The arm loop: prepare data (controller update) -> train K steps -> repeat.
# Usage: MS_ROOT=... run_arm.sh adaptive|static <n_cycles> [steps_per_cycle]
set -eu
ARM=$1; CYCLES=$2; K=${3:-10}
ROOT=${MS_ROOT:?set MS_ROOT to the mathscaffold checkout}
WORK=${MS_WORK:-$ROOT/runs/$ARM}
mkdir -p $WORK
# Resume: MS_START_CYCLE=c restarts at cycle c; MS_SKIP_PREPARE=1 additionally reuses
# the already-generated train_c$c.parquet (a training stage that died mid-way — the
# ratio state and parquet for that cycle are still valid, only the trainer needs to
# resume from its last checkpoint).
for ((c=${MS_START_CYCLE:-0}; c<CYCLES; c++)); do
  STEP=$(( (c + 1) * K ))
  RL=$WORK/rollouts_c$c.jsonl
  if [ "$c" = "${MS_START_CYCLE:-0}" ] && [ "${MS_SKIP_PREPARE:-0}" = "1" ] && [ -f $WORK/train_c$c.parquet ]; then
    echo "[resume] cycle $c: reusing existing train_c$c.parquet, skipping prepare" | tee -a $WORK/arm.log
  else
    python $ROOT/scripts/prepare_cycle.py --arm $ARM --cycle $c \
        --state $WORK/ratio_state.json \
        --rollout-log $WORK/rollouts_c$((c-1)).jsonl \
        --out $WORK/train_c$c.parquet 2>&1 | tee -a $WORK/arm.log
  fi
  # a training stage may hang (seen: vLLM/Ray stall at 0% GPU for an hour); retry the
  # stage up to 2 more times from its checkpoint before giving up on the cycle
  for attempt in 1 2 3; do
    bash $ROOT/scripts/stage_watchdog.sh $WORK/train_c$c.log ${MS_EXP:-questa_$ARM} &
    WD=$!
    MATHSCAFFOLD_ROLLOUT_LOG=$RL MS_EXP=${MS_EXP:-questa_$ARM} \
        bash $ROOT/scripts/train_stage.sh $STEP $WORK/train_c$c.parquet \
        2>&1 | tee -a $WORK/train_c$c.log
    kill $WD 2>/dev/null
    LAST=$(cat ${MS_CKPTS:?}/${MS_EXP:-questa_$ARM}/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
    [ "$LAST" -ge "$STEP" ] && break
    echo "[retry] cycle $c attempt $attempt ended with ckpt $LAST < $STEP; resuming" | tee -a $WORK/arm.log
  done
  # hint-free probe every MS_PROBE_EVERY cycles (default 5 = 50 steps): AIME24/25,
  # HMMT25 -> probe.json (teacher preamble + wandb). Failure never stops training.
  if [ $(( (c + 1) % ${MS_PROBE_EVERY:-5} )) -eq 0 ]; then
    MS_EXP=${MS_EXP:-questa_$ARM} bash $ROOT/scripts/probe_ckpt.sh $((c + 1)) \
        2>&1 | tee -a $WORK/probe.log || echo "[probe] failed at cycle $((c+1)), continuing"
  fi
done
