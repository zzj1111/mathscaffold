#!/usr/bin/env bash
# Hint-free probe of the latest checkpoint: merge FSDP->HF, serve with vLLM on EVERY
# visible GPU (one single-GPU server each, data-parallel), run eval_probe over the
# given sets, write $MS_WORK/probe.json (teacher preamble + wandb pick it up), stop
# the servers. Called by run_arm.sh every MS_PROBE_EVERY cycles; safe to run by hand.
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

# one server per visible GPU (CUDA_VISIBLE_DEVICES, else all GPUs on the box)
GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')}
IFS=',' read -ra GARR <<< "$GPUS"
VPIDS=(); URLS=""
for i in "${!GARR[@]}"; do
  P=$((PORT + i))
  CUDA_VISIBLE_DEVICES=${GARR[$i]} $PY -m vllm.entrypoints.openai.api_server \
      --model $HF --served-model-name actor \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 32768 \
      --host 127.0.0.1 --port $P > $WORK/probe_vllm_c${CYCLE}_g${GARR[$i]}.log 2>&1 &
  VPIDS+=($!)
  URLS="$URLS,http://127.0.0.1:$P/v1"
done
URLS=${URLS#,}
echo "[probe] cycle $CYCLE step $STEP: ${#GARR[@]} vLLM servers on GPUs $GPUS"
stop_servers() { for p in "${VPIDS[@]}"; do kill $p 2>/dev/null || true; done; for p in "${VPIDS[@]}"; do wait $p 2>/dev/null || true; done; }
trap stop_servers EXIT
for i in "${!GARR[@]}"; do
  P=$((PORT + i))
  for t in $(seq 1 120); do
    curl -sf http://127.0.0.1:$P/v1/models >/dev/null 2>&1 && break; sleep 10
  done
done
$PY $ROOT/scripts/eval_probe.py --base-url "$URLS" --sets $SETS \
    --n ${MS_PROBE_N:-32} --out $WORK/probe.json --cycle $CYCLE 2>&1 | tee -a ${LOGDIR:-$WORK}/probe.log
echo "[probe] cycle $CYCLE step $STEP done -> $WORK/probe.json"
