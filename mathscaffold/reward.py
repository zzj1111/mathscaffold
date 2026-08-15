"""verl custom reward function + per-prompt outcome recorder.

Wire via verl config:
  custom_reward_function.path=<this file> custom_reward_function.name=compute_score
Score: 1.0 iff the extracted boxed answer verifies against ground truth
(Math-Verify), else 0. Every scored rollout appends one JSONL row
{qid, ratio, score} to $MATHSCAFFOLD_ROLLOUT_LOG so the controller can rebuild
per-problem group outcomes without touching trainer internals.
"""
from __future__ import annotations

import json
import os


def _verify(solution_str, ground_truth):
    try:
        from math_verify import parse, verify
        return 1.0 if verify(parse(ground_truth), parse(solution_str)) else 0.0
    except Exception:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw):
    score = _verify(solution_str, ground_truth)
    log = os.environ.get("MATHSCAFFOLD_ROLLOUT_LOG")
    if log:
        try:
            with open(log, "a") as f:
                e = extra_info or {}
                head = int(os.environ.get("MS_LOG_HEAD_CHARS", "800"))
                tail = int(os.environ.get("MS_LOG_TAIL_CHARS", "400"))
                t = str(solution_str or "")
                excerpt = t if len(t) <= head + tail else (
                    t[:head] + "\n...[middle truncated]...\n" + t[-tail:])
                f.write(json.dumps({"qid": e.get("qid"), "ratio": e.get("ratio"),
                                    "text_inj": bool(e.get("text_inj")),
                                    "score": score, "text": excerpt},
                                   ensure_ascii=False) + "\n")
        except OSError:
            pass
    return score
