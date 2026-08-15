#!/usr/bin/env bash
set -u
LOG=/home/zha00175/mathscaffold/runs/smoke_chain.log
mkdir -p /home/zha00175/mathscaffold/runs
log() { echo "$(date +%F\ %T) $*" >> $LOG; }
log "waiting for v3full exit at c2 boundary"
while true; do
  alive=0
  for pid in $(pgrep -u zha00175 python 2>/dev/null); do
    exp=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | grep -m1 "^ARM_EXP=") || true
    [ "$exp" = "ARM_EXP=alf_v3full_300" ] && alive=1
  done
  [ $alive = 0 ] && break
  sleep 45
done
log "v3full exited; purging requeue leftovers"
/home/zha00175/venv_verl/bin/python - <<'PY' >> $LOG 2>&1
import json
for path in ("/home/zha00175/alfscaffold/runs/exp/alf_v3full_300/state.json",
             "/home/zha00175/alfscaffold/runs/exp/alf_v3full_300/scaffold.json"):
    d = json.load(open(path))
    sc = d["scaffold"] if "scaffold" in d else d
    if sc.pop("requeue_next", None) is not None:
        print(f"purged {path}")
    json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
PY
rm -f /home/zha00175/alfscaffold/runs/exp/alf_v3full_300/restart.requested
sleep 20
log "launching math smoke"
bash /home/zha00175/mathscaffold/scripts/smoke_local.sh >> $LOG 2>&1 && log "SMOKE OK" || log "SMOKE FAILED"
