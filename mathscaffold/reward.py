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
                cap = int(os.environ.get("MS_LOG_TEXT_CHARS", "1200"))
                f.write(json.dumps({"qid": e.get("qid"), "ratio": e.get("ratio"),
                                    "topic": e.get("topic"),
                                    "text_inj": bool(e.get("text_inj")),
                                    "score": score,
                                    "text": str(solution_str or "")[:cap]},
                                   ensure_ascii=False) + "\n")
        except OSError:
            pass
    return score
