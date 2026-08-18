#!/usr/bin/env bash
# The arm loop: prepare data (controller update) -> train K steps -> repeat.
# Usage: MS_ROOT=... run_arm.sh adaptive|static|teacher <n_cycles> [steps_per_cycle]
set -eu
ARM=$1; CYCLES=$2; K=${3:-10}
ROOT=${MS_ROOT:?set MS_ROOT to the mathscaffold checkout}
WORK=${MS_WORK:-$ROOT/runs/$ARM}
EXP=${MS_EXP:-questa_$ARM}
PY=${MS_PYTHON:-python3}
mkdir -p $WORK
# Every launch gets its own dated log directory (a retry/resume never appends into an
# older launch's files); $WORK/logs/latest always points at the newest launch.
RUN_TAG=${MS_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
LOGDIR=$WORK/logs/$RUN_TAG
mkdir -p $LOGDIR && ln -sfn $RUN_TAG $WORK/logs/latest
ARM_LOG=$LOGDIR/arm.log
# wandb's local files go under the launch's log dir, never into the repo checkout
# (tracked-file collisions on `git pull` seen live on B200)
export WANDB_DIR=${WANDB_DIR:-$LOGDIR/wandb}; mkdir -p $WANDB_DIR

# ---------------------------------------------------------------- preflight
# Everything that can make a launch die (or hang) silently is checked HERE, loudly,
# before any GPU work: a launch whose log stops after "[preflight]" lines names its
# own reason. (Seen live: a B200 launch that "showed nothing".)
pf() { echo "[preflight] $(date +%T) $*" | tee -a $ARM_LOG; }
die() { pf "FAIL: $*"; exit 2; }
pf "arm=$ARM cycles=$CYCLES steps/cycle=$K exp=$EXP launch=$RUN_TAG"
pf "logs=$LOGDIR (symlink: $WORK/logs/latest) wandb_dir=$WANDB_DIR"
pf "root=$ROOT work=$WORK"
pf "python=$($PY -c 'import sys;print(sys.executable, sys.version.split()[0])' 2>/dev/null || echo MISSING)"
$PY -c 'import sys;print(sys.executable)' >/dev/null 2>&1 || die "python '$PY' not runnable (activate the env or set MS_PYTHON)"
for m in verl vllm wandb math_verify pandas pyarrow openai; do
  $PY -c "import $m" 2>/dev/null || die "python module '$m' missing in $($PY -c 'import sys;print(sys.executable)') (pip/uv install it in this env)"
done
pf "modules ok: verl vllm wandb math_verify pandas pyarrow openai"
[ -d "${MS_MODEL:?set MS_MODEL to the base model dir}" ] || die "MS_MODEL dir not found: $MS_MODEL"
mkdir -p "${MS_CKPTS:?set MS_CKPTS to the ckpt root}" || die "cannot create MS_CKPTS=$MS_CKPTS"
IFS=',' read -ra _DF <<< "${MS_DATA:?set MS_DATA to comma-separated QuestA jsonl files}"
for f in "${_DF[@]}"; do [ -f "$f" ] || die "MS_DATA file not found: $f"; done
pf "model=$MS_MODEL ckpts=$MS_CKPTS data=${#_DF[@]} file(s)"
pf "gpus: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} MS_N_GPUS=${MS_N_GPUS:-8} visible=$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "$ARM" = "teacher" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "${AUTOSCAFFOLD_OPENAI_KEY_FILE:-/nonexistent}" ] && [ ! -f "$HOME/.openai_key" ]; then
    die "teacher arm needs OPENAI_API_KEY (or AUTOSCAFFOLD_OPENAI_KEY_FILE / ~/.openai_key)"
  fi
  pf "openai key: present (model ${MS_TEACHER_MODEL:-gpt-5.5})"
fi
if [ "${MS_WANDB:-0}" = "1" ] && [ "${WANDB_MODE:-online}" = "online" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    die "MS_WANDB=1 but wandb is not logged in: run 'wandb login' or export WANDB_API_KEY (an interactive login prompt would hang a nohup launch)"
  fi
  pf "wandb: logged in, entity=${MS_WANDB_ENTITY:-<default>} project=${MS_WANDB_PROJECT:-mathscaffold}"
else
  pf "wandb: off (MS_WANDB=${MS_WANDB:-0}, WANDB_MODE=${WANDB_MODE:-online})"
fi
pf "OK — entering cycle loop (logs: $LOGDIR/{arm,train_cN,probe,watch}.log; ckpts: $MS_CKPTS/$EXP)"

# ship log tails + liveness + GPU to wandb run <exp>_watch (Logs tab shows the arm's
# stdout live; status/alive drops to 0 with the last lines when the arm dies), so a
# crash is diagnosable from the browser without ssh. MS_WATCH=0 disables.
if [ "${MS_WANDB:-0}" = "1" ] && [ "${MS_WATCH:-1}" = "1" ]; then
  STDOUT_LOG=${MS_STDOUT_LOG:-$(readlink -f /proc/$$/fd/1 2>/dev/null || true)}
  [ -f "$STDOUT_LOG" ] || STDOUT_LOG=$ARM_LOG
  setsid nohup $PY $ROOT/scripts/wandb_watch.py --work $WORK --exp $EXP \
      --stdout-log "$STDOUT_LOG" --arm-log "$ARM_LOG" --arm-pid $$ --ckpts $MS_CKPTS \
      >> $LOGDIR/watch.log 2>&1 < /dev/null &
  pf "wandb watch: run ${EXP}_watch tails $STDOUT_LOG (local: $LOGDIR/watch.log)"
fi

# ---------------------------------------------------------------- cycle loop
# Resume: MS_START_CYCLE=c restarts at cycle c; MS_SKIP_PREPARE=1 additionally reuses
# the already-generated train_c$c.parquet (a training stage that died mid-way — the
# ratio state and parquet for that cycle are still valid, only the trainer needs to
# resume from its last checkpoint).
for ((c=${MS_START_CYCLE:-0}; c<CYCLES; c++)); do
  STEP=$(( (c + 1) * K ))
  RL=$WORK/rollouts_c$c.jsonl
  if [ "$c" = "${MS_START_CYCLE:-0}" ] && [ "${MS_SKIP_PREPARE:-0}" = "1" ] && [ -f $WORK/train_c$c.parquet ]; then
    echo "[resume] cycle $c: reusing existing train_c$c.parquet, skipping prepare" | tee -a $ARM_LOG
  else
    $PY $ROOT/scripts/prepare_cycle.py --arm $ARM --cycle $c \
        --state $WORK/ratio_state.json \
        --rollout-log $WORK/rollouts_c$((c-1)).jsonl \
        --out $WORK/train_c$c.parquet 2>&1 | tee -a $ARM_LOG
  fi
  # a training stage may hang (seen: vLLM/Ray stall at 0% GPU for an hour); retry the
  # stage up to 2 more times from its checkpoint before giving up on the cycle
  for attempt in 1 2 3; do
    bash $ROOT/scripts/stage_watchdog.sh $LOGDIR/train_c$c.log $EXP &
    WD=$!
    MATHSCAFFOLD_ROLLOUT_LOG=$RL MS_EXP=$EXP \
        bash $ROOT/scripts/train_stage.sh $STEP $WORK/train_c$c.parquet \
        2>&1 | tee -a $LOGDIR/train_c$c.log
    kill $WD 2>/dev/null || true
    LAST=$(cat $MS_CKPTS/$EXP/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
    [ "$LAST" -ge "$STEP" ] && break
    echo "[retry] cycle $c attempt $attempt ended with ckpt $LAST < $STEP; resuming" | tee -a $ARM_LOG
    # rows of the dead attempt would double-count in the next prepare: archive them
    [ -f $RL ] && mv $RL $WORK/rollouts_c$c.attempt$attempt.jsonl
  done
  # All 3 attempts exhausted without reaching STEP. Falling through here is what let a
  # dead arm march from cycle 2 to cycle 6 on 2026-08-18: prepare_cycle.py kept running
  # with zero outcomes, appending empty visit history and advancing ratio_state.json for
  # every later cycle. Stop instead.
  LAST=$(cat $MS_CKPTS/$EXP/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  if [ "$LAST" -lt "$STEP" ]; then
    echo "[arm] FAILED at cycle $c: ckpt $LAST < target $STEP after 3 attempts; stopping" \
        | tee -a $WORK/arm.log
    exit 1
  fi
  # hint-free probe every MS_PROBE_EVERY cycles (default 5 = 50 steps): AIME24/25,
  # HMMT25 -> probe.json (teacher preamble + wandb). Failure never stops training.
  if [ $(( (c + 1) % ${MS_PROBE_EVERY:-5} )) -eq 0 ]; then
    MS_EXP=$EXP bash $ROOT/scripts/probe_ckpt.sh $((c + 1)) \
        2>&1 | tee -a $LOGDIR/probe.log || echo "[probe] failed at cycle $((c+1)), continuing"
  fi
done
