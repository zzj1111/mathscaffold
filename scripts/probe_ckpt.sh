#!/usr/bin/env bash
# Hint-free probe of the latest checkpoint: merge FSDP->HF, serve with vLLM,
# run eval_probe over the given sets, write $MS_WORK/probe.json (teacher preamble +
# wandb pick it up), stop the server. Called by run_arm.sh every MS_PROBE_EVERY
# cycles; safe to run by hand.
set -eu
CYCLE=$1
SETS=${MS_PROBE_SETS:-aime24,aime25,hmmt25}
ROOT=${MS_ROOT:?}; WORK=${MS_WORK:?}; PY=${MS_PYTHON:-python3}
EXP=${MS_EXP:-questa_teacher}; CKPTS=${MS_CKPTS:?}/$EXP
PORT=${MS_PROBE_PORT:-8123}
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
$PY -m vllm.entrypoints.openai.api_server --model $HF --served-model-name actor \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 32768 \
    --host 127.0.0.1 --port $PORT > $WORK/probe_vllm_c$CYCLE.log 2>&1 &
VPID=$!
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 10
done
$PY $ROOT/scripts/eval_probe.py --base-url http://127.0.0.1:$PORT/v1 --sets $SETS \
    --n ${MS_PROBE_N:-32} --out $WORK/probe.json --cycle $CYCLE 2>&1 | tee -a $WORK/probe.log
kill $VPID 2>/dev/null; wait $VPID 2>/dev/null || true
echo "[probe] cycle $CYCLE step $STEP done -> $WORK/probe.json"
