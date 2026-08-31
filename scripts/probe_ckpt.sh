#!/usr/bin/env bash
# Hint-free probe of the latest checkpoint: merge FSDP->HF, serve with vLLM on EVERY
# visible GPU (one single-GPU server each, data-parallel), run eval_probe over the
# given sets, write $MS_WORK/probe.json (teacher preamble + wandb pick it up), stop
# the servers. Called by run_arm.sh every MS_PROBE_EVERY cycles; safe to run by hand.
set -eu
CYCLE=$1
SETS=${MS_PROBE_SETS:-aime24,aime25,hmmt25}
ROOT=${MS_ROOT:?}; WORK=${MS_WORK:-$ROOT/runs/${MS_ARM:-teacher}}; PY=${MS_PYTHON:-python3}
EXP=${MS_EXP:-questa_teacher}; CKPTS=${MS_CKPTS:?}/$EXP
PORT=${MS_PROBE_PORT:-8123}
# vLLM compiles kernels with ninja from the python env's bin; the scripts call $PY by path
# without activating the env, so put that bin dir on PATH (seen live: all 4 servers died
# with FileNotFoundError: 'ninja' and the probe timed out)
export PATH=$(dirname $(command -v $PY)):$PATH
STEP=$(cat $CKPTS/latest_checkpointed_iteration.txt)
CK=$CKPTS/global_step_$STEP
HF=$CK/hf
if [ ! -f $HF/config.json ]; then
  $PY -m verl.model_merger merge --backend fsdp --local_dir $CK/actor --target_dir $HF \
    || $PY ${MS_VERL_ROOT:-.}/scripts/model_merger.py merge --backend fsdp --local_dir $CK/actor --target_dir $HF
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json generation_config.json added_tokens.json; do
    [ -f $MS_MODEL/$f ] && [ ! -f $HF/$f ] && cp $MS_MODEL/$f $HF/ || true
  done
fi

# one single-GPU server per visible GPU via the shared pool (waits for the trainer to
# release the GPUs, picks free ports, setsid per server, fails fast with log tails)
source $ROOT/scripts/vllm_pool.sh
LOGPREFIX=$WORK/probe_vllm_c${CYCLE} MAXLEN=${MS_PROBE_MAXLEN:-40960} UTIL=0.85
ms_pool_start || { echo "[probe] FAIL: GPUs not free"; exit 3; }
trap ms_pool_stop EXIT
ms_pool_wait 1500 || { echo "[probe] FAIL: vLLM servers did not come up"; exit 3; }
echo "[probe] cycle $CYCLE step $STEP: servers $URLS"
$PY $ROOT/scripts/eval_probe.py --base-url "$URLS" --sets $SETS \
    --n ${MS_PROBE_N:-8} --ratios ${MS_PROBE_RATIOS:-0,50} \
    --max-tokens ${MS_PROBE_MAXTOK:-32768} \
    --temperature ${MS_PROBE_TEMP:-0.7} --top-p ${MS_PROBE_TOPP:-0.95} \
    --instruction ${MS_PROBE_INSTRUCTION:-official} \
    --out $WORK/probe.json --cycle $CYCLE 2>&1 | tee -a ${LOGDIR:-$WORK}/probe.log
echo "[probe] cycle $CYCLE step $STEP done -> $WORK/probe.json"
