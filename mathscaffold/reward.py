"""verl custom reward function + per-prompt outcome recorder.

Wire via verl config:
  custom_reward_function.path=<this file> custom_reward_function.name=compute_score
Score: 1.0 iff the extracted boxed answer verifies against ground truth
(Math-Verify), else 0. Every scored rollout appends one JSONL row
{qid, ratio, score} to $MATHSCAFFOLD_ROLLOUT_LOG so the controller can rebuild
per-problem group outcomes without touching trainer internals.

Verification runs in helper subprocesses with a hard wall-clock bound.
Why: verl's reward manager calls from worker THREADS, where Math-Verify's own
signal-based timeouts raise (found live: uniform zero reward), and with timeouts
disabled a single pathological parse/verify spun one core forever and stalled the
whole run (found live twice, both at the same step -> deterministic input). A
child that overruns is killed and respawned; that sample scores 0 and is flagged.
"""
from __future__ import annotations

import json
import os
import queue
import select
import subprocess
import sys
import threading

VERIFY_TIMEOUT_S = float(os.environ.get("MS_VERIFY_TIMEOUT", "20"))
VERIFY_WORKERS = int(os.environ.get("MS_VERIFY_WORKERS", "4"))
TAIL_CHARS = 3000

# Self-contained helper: one JSON request per line on stdin, one JSON reply per
# line on stdout. Timeouts stay disabled inside (the parent enforces the bound).
_CHILD = r"""
import json, sys, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from math_verify import parse, verify
for line in sys.stdin:
    try:
        req = json.loads(line)
        gold = parse(str(req["gt"]), parsing_timeout=None)
        pred = parse(str(req["sol"]), parsing_timeout=None)
        s = 1.0 if verify(gold, pred, timeout_seconds=None) else 0.0
    except Exception:
        s = 0.0
    sys.stdout.write(json.dumps({"s": s}) + "\n"); sys.stdout.flush()
"""


def _spawn():
    return subprocess.Popen([sys.executable, "-c", _CHILD], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)


class _VerifierPool:
    """N helper processes; a call checks one out, round-trips one request under a
    wall-clock bound, and returns it (or a fresh replacement if it had to be killed)."""

    def __init__(self, n, timeout_s):
        self.timeout_s = timeout_s
        self.idle = queue.Queue()
        self._lock = threading.Lock()
        self._n = n
        self._started = False

    def _ensure(self):
        with self._lock:
            if not self._started:
                for _ in range(self._n):
                    self.idle.put(_spawn())
                self._started = True

    def verify(self, sol, gt):
        """-> (score, timed_out)"""
        self._ensure()
        proc = self.idle.get()
        try:
            if proc.poll() is not None:
                proc = _spawn()
            proc.stdin.write(json.dumps({"gt": str(gt), "sol": str(sol)}) + "\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], self.timeout_s)
            if not ready:
                proc.kill()
                proc.wait()
                proc = _spawn()
                return 0.0, True
            line = proc.stdout.readline()
            if not line:                       # child died mid-request
                proc = _spawn()
                return 0.0, False
            return float(json.loads(line).get("s", 0.0)), False
        except (OSError, ValueError, BrokenPipeError):
            try:
                proc.kill()
            except OSError:
                pass
            proc = _spawn()
            return 0.0, False
        finally:
            self.idle.put(proc)


_POOL = _VerifierPool(VERIFY_WORKERS, VERIFY_TIMEOUT_S)


def _verify_inprocess(solution_str, ground_truth):
    """Fallback if helper processes cannot be spawned at all."""
    try:
        from math_verify import parse, verify
        gold = parse(str(ground_truth), parsing_timeout=None)
        pred = parse(str(solution_str), parsing_timeout=None)
        return 1.0 if verify(gold, pred, timeout_seconds=None) else 0.0
    except Exception:
        return 0.0


def _verify(solution_str, ground_truth):
    """-> (score, timed_out). Parse the answer-bearing tail of the response.

    Second chance when the tail scores 0 but the LAST \\boxed{} sits before it: a response
    that boxes its answer and then keeps writing (verification, alternative routes) would
    otherwise be marked wrong. Measured on 20,480 real rollouts: 0.45% of all of them, but
    the miss rate scales with length — 0% below 20K chars, 0.26% at 20-40K, 0.63% above
    40K — i.e. it is a silent penalty on long-but-correct answers, precisely the regime
    whose dynamics we are studying. The extra parse only runs for the ~6% of zeros at risk.
    """
    s = str(solution_str or "")
    tail = s[-TAIL_CHARS:]
    try:
        score, timed_out = _POOL.verify(tail, ground_truth)
    except Exception:
        score, timed_out = _verify_inprocess(tail, ground_truth), False
    if score <= 0 and len(s) > TAIL_CHARS:
        i = s.rfind("\\boxed{")
        if 0 <= i < len(s) - TAIL_CHARS:
            seg = s[max(0, i - 200): i + TAIL_CHARS]
            try:
                score2, t2 = _POOL.verify(seg, ground_truth)
            except Exception:
                score2, t2 = _verify_inprocess(seg, ground_truth), False
            if score2 > 0:
                return score2, timed_out or t2
    return score, timed_out


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw):
    score, timed_out = _verify(solution_str, ground_truth)
    log = os.environ.get("MATHSCAFFOLD_ROLLOUT_LOG")
    if log:
        try:
            with open(log, "a") as f:
                e = extra_info or {}
                # full text on disk (excerpting is the tool's job — get_traces
                # windows into it with head/tail/offset); a generous cap guards
                # against pathological runaways only
                cap = int(os.environ.get("MS_LOG_TEXT_CAP", "60000"))
                row = {"qid": e.get("qid"), "ratio": e.get("ratio"),
                       "text_inj": bool(e.get("text_inj")),
                       "score": score,
                       "text": str(solution_str or "")[:cap]}
                if timed_out:
                    row["verify_timeout"] = True
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass
    return score
