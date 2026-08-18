#!/usr/bin/env bash
# Launch an arm detached with a DATED stdout log and print where everything goes.
#   MS_EXP=questa_teacher_b200 bash scripts/launch.sh teacher 30
# Env: same as run_arm.sh (MS_ROOT, MS_MODEL, MS_CKPTS, MS_DATA, MS_WANDB=1 ..., MS_START_CYCLE / MS_SKIP_PREPARE for resume).
set -eu
ARM=${1:?usage: launch.sh adaptive|static|teacher <n_cycles>}; CYCLES=${2:-30}
ROOT=${MS_ROOT:?set MS_ROOT}; WORK=${MS_WORK:-$ROOT/runs/$ARM}
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p $WORK/logs
STDOUT=$WORK/logs/${ARM}_${TAG}.stdout.log
MS_RUN_TAG=$TAG setsid nohup bash $ROOT/scripts/run_arm.sh $ARM $CYCLES > $STDOUT 2>&1 < /dev/null &
PID=$!
echo "$PID" > $WORK/logs/${ARM}_${TAG}.pid
echo "[launch] pid $PID"
echo "[launch] stdout      $STDOUT"
echo "[launch] launch logs $WORK/logs/$TAG/  (arm.log, train_cN.log, probe.log, watch.log, wandb/)  ->  $WORK/logs/latest"
sleep 20
echo "[launch] first lines:"; tail -n 15 $STDOUT
