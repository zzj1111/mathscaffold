#!/usr/bin/env bash
# Kill a stalled training stage: if $LOG has not grown for STALL_MIN minutes, SIGTERM
# the TRAINER processes carrying MS_EXP=$EXP (ours only, verified by environ), so
# run_arm's retry loop resumes from the last checkpoint. Run in background beside the stage.
#
# STALL_MIN must exceed the longest healthy gap between two log writes: verl writes the
# log only at step end, and one 32K-response step (plus the end-of-stage checkpoint save)
# took 25-37 min live on the B200s. With the old 30-min default the watchdog killed two
# healthy stages at steps 79/80 and 72/73 (2026-08-23 01:2xZ / 01:4xZ), costing a 70->80
# re-walk each. Default: 90 min at MS_MAXRESP >= 32768, else 45 (MS_STALL_MIN overrides).
LOG=$1; EXP=$2
if [ -n "${MS_STALL_MIN:-}" ]; then STALL_MIN=$MS_STALL_MIN
elif [ "${MS_MAXRESP:-32768}" -ge 32768 ]; then STALL_MIN=90
else STALL_MIN=45; fi
last=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)

# our trainer-side python processes only: the wandb watcher (wandb_watch.py) also carries
# MS_EXP in its environ and must survive (killing it blinded the wandb status panels live)
stage_pids() {
  local p e
  for p in $(pgrep -u "$(id -un)" python 2>/dev/null); do
    e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -m1 "^MS_EXP=")
    [ "$e" = "MS_EXP=$EXP" ] || continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "wandb_watch.py" && continue
    echo $p
  done
}

while true; do
  sleep 120
  now=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  if [ "$now" != "$last" ]; then last=$now; continue; fi
  age=$(( ($(date +%s) - now) / 60 ))
  if [ "$age" -ge "$STALL_MIN" ]; then
    echo "[watchdog] $(date +%T) log idle ${age} min >= $STALL_MIN; killing stage of $EXP" | tee -a "$LOG"
    for p in $(stage_pids); do kill $p 2>/dev/null; done
    sleep 20
    for p in $(stage_pids); do kill -9 $p 2>/dev/null; done
    exit 0
  fi
done
