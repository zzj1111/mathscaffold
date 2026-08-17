"""Ship the arm's logs and liveness to wandb so a crash is visible in the browser.

Runs beside run_arm.sh (started automatically after preflight when MS_WANDB=1).
Every --interval seconds it:
  * prints the NEW lines of the arm's stdout log (runs/<exp>.log) and arm.log to its
    own stdout -> wandb captures them into this run's "Logs" tab, live;
  * logs status/* (alive, cycle, ckpt_step, log idle minutes, GPU util/mem) so a flat
    line or alive=0 is visible on the run page;
  * on Traceback / FAIL / [retry] lines logs status/last_error (text table) and
    status/errors counter.
It outlives the arm: when the arm pid is gone it flushes the tail once more, logs
alive=0 with an exit_reason, and exits. Never raises — a watcher must not become the
thing that breaks the run.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ERR_PAT = re.compile(r"Traceback|Error|FAIL|\[retry\]|OutOfMemory|CUDA error|Killed", re.I)


def _tail_new(path, pos):
    """-> (new_text, new_pos); tolerant of a missing or truncated file."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "", pos
    if size < pos:
        pos = 0
    if size == pos:
        return "", pos
    with open(path, "rb") as f:
        f.seek(pos)
        data = f.read(size - pos)
    return data.decode("utf-8", "replace"), size


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout
        rows = [r.split(",") for r in out.strip().splitlines() if "," in r]
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            idx = [int(i) for i in vis.split(",") if i.strip().isdigit()]
            rows = [rows[i] for i in idx if i < len(rows)]
        if not rows:
            return {}
        return {"gpu/util_mean": sum(float(r[0]) for r in rows) / len(rows),
                "gpu/mem_used_gb_mean": sum(float(r[1]) for r in rows) / len(rows) / 1024}
    except Exception:
        return {}


def _int_file(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _mtime_age_min(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 60
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--stdout-log", required=True, help="the nohup log (runs/<exp>.log)")
    ap.add_argument("--arm-pid", type=int, required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-lines", type=int, default=200, help="lines echoed per tick")
    a = ap.parse_args()
    import wandb
    run = wandb.init(project=os.environ.get("MS_WANDB_PROJECT", "mathscaffold"),
                     entity=os.environ.get("MS_WANDB_ENTITY") or None,
                     id=f"{a.exp}_watch", name=f"{a.exp}_watch", resume="allow",
                     settings=wandb.Settings(init_timeout=60, _disable_stats=True,
                                             console="wrap"))
    print(f"[watch] started for pid {a.arm_pid}, work={a.work}", flush=True)
    pos_out = pos_arm = 0
    errors = 0
    grace = 2                       # extra ticks after the arm dies (final flush)
    while True:
        try:
            alive = _alive(a.arm_pid)
            for path, tag, key in ((a.stdout_log, "stdout", "pos_out"), (os.path.join(a.work, "arm.log"), "arm", "pos_arm")):
                text, newpos = _tail_new(path, pos_out if key == "pos_out" else pos_arm)
                if key == "pos_out":
                    pos_out = newpos
                else:
                    pos_arm = newpos
                if not text:
                    continue
                lines = text.splitlines()
                if len(lines) > a.max_lines:
                    print(f"[watch] ({tag}) ... {len(lines) - a.max_lines} lines skipped ...", flush=True)
                    lines = lines[-a.max_lines:]
                for ln in lines:
                    print(f"[{tag}] {ln}", flush=True)
                bad = [ln for ln in lines if ERR_PAT.search(ln)]
                if bad:
                    errors += len(bad)
                    tbl = wandb.Table(columns=["time", "source", "line"],
                                      data=[[time.strftime("%m-%d %H:%M:%S"), tag, ln[:500]] for ln in bad[-20:]])
                    run.log({"status/last_error": tbl, "status/errors": errors}, commit=False)
            cyc = None
            try:
                for ln in open(os.path.join(a.work, "arm.log"), errors="replace"):
                    m = re.search(r"\[prepare\] cycle (\d+)|\[resume\] cycle (\d+)", ln)
                    if m:
                        cyc = int(m.group(1) or m.group(2))
            except OSError:
                pass
            # the trainer's own wandb run drops points when a retry re-walks steps it
            # already logged (monotonic-step rule), so its curve looks frozen during a
            # resume; the tqdm bar in stdout is the truth of where training actually is
            tstep = None
            try:
                txt = open(a.stdout_log, errors="replace").read()[-200000:]
                bars = re.findall(r"Training Progress:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)", txt)
                if bars:
                    tstep = int(bars[-1][0])
            except OSError:
                pass
            metrics = {"status/alive": int(alive), "status/train_step": tstep,
                       "status/ckpt_step": _int_file(os.path.join(a.ckpts, a.exp, "latest_checkpointed_iteration.txt")),
                       "status/cycle": cyc,
                       "status/stdout_idle_min": _mtime_age_min(a.stdout_log)}
            metrics.update(_gpu())
            run.log({k: v for k, v in metrics.items() if v is not None})
            if not alive:
                grace -= 1
                if grace < 0:
                    tail = ""
                    try:
                        tail = "\n".join(open(a.stdout_log, errors="replace").read().splitlines()[-40:])
                    except OSError:
                        pass
                    print(f"[watch] arm pid {a.arm_pid} is gone; final tail:\n{tail}", flush=True)
                    run.log({"status/alive": 0,
                             "status/exit_reason": wandb.Table(columns=["last_lines"], data=[[tail[-4000:]]])})
                    run.finish()
                    return
        except Exception as e:          # never let the watcher die noisily
            print(f"[watch] tick error: {e!r}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
