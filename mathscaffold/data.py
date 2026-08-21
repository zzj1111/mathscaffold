"""QuestA jsonl -> verl parquet with per-problem hint-prefix ratios.

Faithful to QuestA's construction (add_prefix.py): the hint is the first r% (by
characters) of the post-</think> solution, prepended as '## Hint.'; rows whose
final answer does not appear in the solution text are dropped. r is PER PROBLEM
here (the adaptive arm's whole point); r=0 means a bare prompt.
"""
from __future__ import annotations

import hashlib
import json
import os


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

# MS_PROMPT_STYLE:
#   paper (default) — QuestA paper App. B.8, "Training prompt with partial solutions":
#       {Problem} ## Hint: {Partial Solution} Please reason step by step, and put your
#       final answer within \\boxed{}.
#     rendered as problem + "\n\n" + "## Hint." + prefix + "\n\n" + BOX_INSTR inside the
#     model's chat template (user turn). "## Hint." (period, no space) is the literal from
#     the released add_prefix.py; the paper's appendix prints "## Hint:". This is the
#     format the 2026-08 v1 arms trained with.
#   repo_raw — the released AReaL/datasets/add_prefix.py taken literally: no instruction,
#     and train_stage.sh feeds the string as RAW TEXT (identity chat template). Kept only
#     as a documented control: on OpenMath-Nemotron-1.5B it makes ~half the generations
#     never stop (repetition loops) — see AUTOSCAFFOLD.md 2026-08-21.
PROMPT_STYLE = os.environ.get("MS_PROMPT_STYLE", "paper")


def split_prefix(solution, ratio):
    """QuestA's split_prefix: the first ratio% of the solution by CHARACTERS, cut wherever
    that lands (mid-sentence is normal)."""
    return solution[: int(len(solution) * ratio / 100.0)]


def hint_prompt(problem, solution, ratio, style=None):
    style = style or PROMPT_STYLE
    prefix = split_prefix(solution, ratio)
    body = problem + "\n\n"
    if len(prefix) >= 10:
        body += "## Hint." + prefix
    if style == "repo_raw":
        return body
    if style != "paper":
        raise ValueError(f"unknown MS_PROMPT_STYLE {style!r} (paper|repo_raw)")
    return body + ("\n\n" if len(prefix) >= 10 else "") + BOX_INSTR


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
