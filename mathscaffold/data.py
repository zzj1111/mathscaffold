"""QuestA jsonl -> verl parquet with per-problem hint-prefix ratios.

Faithful to QuestA's construction (add_prefix.py): the hint is the first r% (by
characters) of the post-</think> solution, prepended as '## Hint.'; rows whose
final answer does not appear in the solution text are dropped. r is PER PROBLEM
here (the adaptive arm's whole point); r=0 means a bare prompt.
"""
from __future__ import annotations

import json


def load_problems(jsonl_path):
    """-> list of {qid, problem, answer, solution}; qid is the stable line index."""
    out = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            gen = d.get("generation") or ""
            if gen[:1] == '"':
                gen = gen[1:-1]
            solution = gen.split("</think>")[-1]
            answer = str(d.get("answer") or "")
            if not answer or answer not in solution:
                continue
            out.append({"qid": f"q{i}", "problem": d["problem"],
                        "answer": answer, "solution": solution})
    return out


def hint_prompt(problem, solution, ratio):
    """QuestA's splice: first ratio% of solution chars as '## Hint.'; <10 chars = bare."""
    prefix = solution[: int(len(solution) * ratio / 100.0)]
    if len(prefix) < 10:
        return problem + "\n\n"
    return problem + "\n\n" + "## Hint." + prefix


def build_rows(problems, ratios, served_qids=None):
    """verl-style rows for the served problems at their current ratios."""
    rows = []
    for p in problems:
        if served_qids is not None and p["qid"] not in served_qids:
            continue
        r = float((ratios.get(p["qid"]) or {}).get("r", 0.0))
        rows.append({
            "data_source": "questa_math",
            "prompt": [{"role": "user", "content": hint_prompt(p["problem"], p["solution"], r)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "\\boxed{" + p["answer"] + "}"},
            "extra_info": {"qid": p["qid"], "ratio": r},
        })
    return rows


def write_parquet(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path)
    return path
