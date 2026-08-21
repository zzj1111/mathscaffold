#!/usr/bin/env bash
# Shared vLLM server pool for the probes (sourced by probe_ckpt.sh and bare_probe.sh).
# One single-GPU server per visible GPU, robust against what broke probes live:
#   * GPUs still held by the exiting trainer (Ray/vLLM release lags)  -> wait, then start
#   * a port still bound by a previous probe's leftovers                -> pick free ports
#   * killing only the API server orphaned VLLM::EngineCore children    -> setsid + kill the
#     whole process group, then wait for the memory to actually come back
#   * servers that never come up while the caller proceeds anyway       -> fail fast with the
#     server log tail (exit 3); the arm loop logs "[probe] failed" and keeps training
# Usage (after setting PY, HF, LOGPREFIX, optional MAXLEN/UTIL):
#   source scripts/vllm_pool.sh; ms_pool_start; ...use $URLS...; (trap ms_pool_stop EXIT)

ms_gpu_list() {
  echo "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')}"
}

# wait until every visible GPU has < $1 MiB in use (default 3000), up to $2 s (default 600)
ms_wait_gpus_free() {
  local thr=${1:-3000} maxs=${2:-600} t=0 g used busy
  IFS=',' read -ra _G <<< "$(ms_gpu_list)"
  while :; do
    busy=""
    for g in "${_G[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1 | tr -d ' ')
      [ -n "$used" ] && [ "$used" -ge "$thr" ] && busy="$busy gpu$g=${used}MiB"
    done
    [ -z "$busy" ] && return 0
    if [ "$t" -ge "$maxs" ]; then
      echo "[pool] GPUs still busy after ${maxs}s:$busy — holders:" >&2
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -20 >&2
      return 1
    fi
    [ "$t" -eq 0 ] && echo "[pool] waiting for GPUs to free up:$busy" >&2
    sleep 10; t=$((t + 10))
  done
}

ms_free_port() {   # first free TCP port >= $1
  "$PY" - "$1" <<'EOF'
import socket, sys
p = int(sys.argv[1])
while p < 65000:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        p += 1
EOF
}

# starts one server per GPU; sets VPGIDS (process groups), VLOGS, URLS
ms_pool_start() {
  local port0=${PORT:-8123} i g port next
  IFS=',' read -ra _G <<< "$(ms_gpu_list)"
  VPGIDS=(); VLOGS=(); URLS=""
  export PATH=$(dirname $(command -v $PY)):$PATH     # vLLM needs ninja from the env's bin
  ms_wait_gpus_free "${POOL_FREE_THR:-3000}" "${POOL_FREE_WAIT:-600}" || return 3
  next=$port0
  for i in "${!_G[@]}"; do
    g=${_G[$i]}
    # ports are handed out monotonically within the pool: a server that has not bound
    # yet must not be offered the same free port again (seen live: 4 servers on one port)
    port=$(ms_free_port $next); next=$((port + 1))
    local log=${LOGPREFIX:-/tmp/vllm}_g${g}.log
    CUDA_VISIBLE_DEVICES=$g setsid "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$HF" --served-model-name actor \
        --tensor-parallel-size 1 --gpu-memory-utilization "${UTIL:-0.85}" --max-model-len "${MAXLEN:-40960}" \
        --enable-prefix-caching --host 127.0.0.1 --port "$port" > "$log" 2>&1 < /dev/null &
    VPGIDS+=($!); VLOGS+=("$log")
    URLS="$URLS,http://127.0.0.1:$port/v1"
  done
  URLS=${URLS#,}
  echo "[pool] ${#_G[@]} servers starting on GPUs $(ms_gpu_list): $URLS" >&2
}

# waits until every server answers /v1/models; on failure prints each dead server's log tail
ms_pool_wait() {
  local maxs=${1:-1500} t=0 u ok
  while :; do
    ok=1
    for u in ${URLS//,/ }; do curl -sf "$u/models" >/dev/null 2>&1 || ok=0; done
    [ $ok = 1 ] && { echo "[pool] all servers healthy (${t}s)" >&2; return 0; }
    # a server process that already exited will never become healthy: fail now
    local i
    for i in "${!VPGIDS[@]}"; do
      if ! kill -0 "${VPGIDS[$i]}" 2>/dev/null; then
        echo "[pool] FAIL: server ${VPGIDS[$i]} exited; log tail ${VLOGS[$i]}:" >&2
        grep -vE "^\s*$" "${VLOGS[$i]}" | tail -n 25 >&2
        return 3
      fi
    done
    if [ "$t" -ge "$maxs" ]; then
      echo "[pool] FAIL: servers not healthy after ${maxs}s; log tails:" >&2
      for i in "${!VLOGS[@]}"; do tail -n 8 "${VLOGS[$i]}" >&2; done
      return 3
    fi
    sleep 10; t=$((t + 10))
  done
}

# kills every server's whole process group (API server + EngineCore children), then waits
# for the GPU memory to come back so the next stage starts on clean GPUs
ms_pool_stop() {
  local p
  for p in "${VPGIDS[@]:-}"; do [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null || true; done
  sleep 5
  for p in "${VPGIDS[@]:-}"; do [ -n "$p" ] && kill -KILL -- "-$p" 2>/dev/null || true; done
  ms_wait_gpus_free "${POOL_FREE_THR:-3000}" 180 || echo "[pool] WARNING: GPU memory not released after stop" >&2
}
