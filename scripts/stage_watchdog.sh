#!/usr/bin/env bash
# Kill a stalled training stage: if $LOG has not grown for STALL_MIN minutes, SIGTERM
# every process carrying MS_EXP=$EXP (ours only, verified by environ), so run_arm's
# retry loop resumes from the last checkpoint. Run in background beside the stage.
LOG=$1; EXP=$2; STALL_MIN=${MS_STALL_MIN:-30}
last=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
while true; do
  sleep 120
  now=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  if [ "$now" != "$last" ]; then last=$now; continue; fi
  age=$(( ($(date +%s) - now) / 60 ))
  if [ "$age" -ge "$STALL_MIN" ]; then
    echo "[watchdog] $(date +%T) log idle ${age} min >= $STALL_MIN; killing stage of $EXP" | tee -a "$LOG"
    for p in $(pgrep -u "$(id -un)" python 2>/dev/null); do
      e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -m1 "^MS_EXP=")
      [ "$e" = "MS_EXP=$EXP" ] && kill $p 2>/dev/null
    done
    sleep 20
    for p in $(pgrep -u "$(id -un)" python 2>/dev/null); do
      e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -m1 "^MS_EXP=")
      [ "$e" = "MS_EXP=$EXP" ] && kill -9 $p 2>/dev/null
    done
    exit 0
  fi
done
