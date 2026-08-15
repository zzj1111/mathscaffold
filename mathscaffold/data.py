"""QuestA jsonl -> verl parquet with per-problem hint-prefix ratios.

Faithful to QuestA's construction (add_prefix.py): the hint is the first r% (by
characters) of the post-</think> solution, prepended as '## Hint.'; rows whose
final answer does not appear in the solution text are dropped. r is PER PROBLEM
here (the adaptive arm's whole point); r=0 means a bare prompt.
"""
from __future__ import annotations

import hashlib
import json


def load_problems(jsonl_paths):
    """-> list of {qid, problem, answer, solution} from one path or a
    comma-separated list (QuestA ships two stage files; we train one merged pool).
    qid = content hash of the normalized problem text: stable across files, file
    order, and reruns. Dedupe by the same key — QuestA's files repeat problems
    with different reference generations, and the two files overlap."""
    out = []
    seen = set()
    for jsonl_path in str(jsonl_paths).split(","):
        jsonl_path = jsonl_path.strip()
        if not jsonl_path:
            continue
        with open(jsonl_path) as f:
            for line in f:
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
                key = " ".join(str(d["problem"]).split())
                if key in seen:
                    continue
                seen.add(key)
                qid = "q" + hashlib.sha1(key.encode()).hexdigest()[:10]
                out.append({"qid": qid, "problem": d["problem"],
                            "answer": answer, "solution": solution})
    return out


BOX_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def hint_prompt(problem, solution, ratio):
    """QuestA's splice (their Fig. 4): problem, optional '## Hint.' prefix of the
    solution (first ratio% of characters; <10 chars = bare), then the boxed-answer
    instruction."""
    prefix = solution[: int(len(solution) * ratio / 100.0)]
    body = problem + "\n\n"
    if len(prefix) >= 10:
        body += "## Hint." + prefix + "\n\n"
    return body + BOX_INSTR


def build_rows(problems, state, served_qids=None, cycle=0):
    """verl-style rows for the served problems: per-problem hint ratio plus the
    general text scaffold (skill/example/plan notes) under one dose coin."""
    from . import textscaffold as TS
    probs = state.get("problems", state)
    text = state.get("text") or TS.empty_text()
    rows = []
    for p in problems:
        if served_qids is not None and p["qid"] not in served_qids:
            continue
        r = float((probs.get(p["qid"]) or {}).get("r", 0.0))
        body = hint_prompt(p["problem"], p["solution"], r)
        has_text = TS.coin(text, p["qid"], cycle)
        if has_text:
            body = TS.render(text) + "\n\n" + body
        rows.append({
            "data_source": "questa_math",
            "prompt": [{"role": "user", "content": body}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "\\boxed{" + p["answer"] + "}"},
            "extra_info": {"qid": p["qid"], "ratio": r, "text_inj": bool(has_text)},
        })
    return rows


def write_parquet(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path)
    return path
