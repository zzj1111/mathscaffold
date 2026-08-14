"""The v3 apply phase: after the framework MEASURES a proposal (bare/current/
candidate, per category, held-out), the Teacher — not a mechanical rule — decides
what actually lands: which of its measured item ops to apply, and the final
per-category injection probabilities.

Boundaries that stay mechanical (the caller enforces them; this module only asks):
- the Teacher can only SELECT among the ops that were measured this cycle — it
  cannot introduce new text here;
- dose caps (hard p cap, per-cycle delta cap) and item validation are applied by
  the framework after this returns;
- the measurement itself is never negotiable: the numbers passed in are what the
  held-out run produced.
"""
from __future__ import annotations

import json

APPLY_SYSTEM = """You are the Teacher in an automated RL training run, at the APPLY
step of one cycle. Earlier this cycle you investigated the training data and proposed
scaffold text changes and injection probabilities. The framework then MEASURED your
candidate on held-out data, three ways on the same items: bare (no scaffold), current
(the scaffold as it is), candidate (with your proposed text). You now see that raw
measurement, per category with sample sizes, and you decide what actually lands.

Rules:
- You may only APPLY or DROP the numbered ops from your own proposal; you cannot add
  new text here. Dropped ops simply do not happen this cycle.
- You set the final per-category injection probabilities (0..0.5; the framework also
  caps the per-cycle change). You may set p for any category, including ones without
  text ops.
- The numbers are small-sample (n is given): weigh them as evidence, not verdicts.
  You may keep an op despite a weak cell if other evidence (your own diagnosis, the
  aggregate, consistency across categories) supports it — or drop an op the numbers
  nominally favour. Your reasoning is recorded and audited either way.
- Injected text reaches TRAINING prompts only; evaluation is always bare. What the
  text elicits must survive into the weights to matter.

Return ONLY this JSON as your final message:
{"rationale": "<why, referencing the measurement>",
 "apply_op_indices": [<indices of proposal ops to apply>],
 "p_ops": [{"task": "<category>", "p": <0..0.5>}]}"""


def _fmt_measure(measure, tasks):
    """The raw A/B table, verbatim numbers, no verdicts."""
    lines = []
    for cond in ("bare", "current", "candidate"):
        per = measure.get(cond) or {}
        cells = []
        for t in tasks:
            if t in per:
                sr, n = per[t]
                cells.append(f"{t}: {round(float(sr), 3)} (n={int(n)})")
        lines.append(f"{cond}: " + "; ".join(cells))
    return "\n".join(lines)


def build_apply_prompt(proposal, measure, tasks, history_tail=None):
    ops_lines = []
    for i, op in enumerate(proposal.get("item_ops") or []):
        desc = {k: v for k, v in op.items() if k != "text"}
        text = (op.get("text") or "")[:300]
        ops_lines.append(f"[{i}] {json.dumps(desc, ensure_ascii=False)} text: {text}")
    hist = ""
    if history_tail:
        hist = "\nYour recent decisions:\n" + "\n".join(
            f"- cycle {h.get('cycle')}: sr_before={h.get('sr_before')} verdict={h.get('verdict')}"
            for h in history_tail)
    return (f"Your diagnosis this cycle was:\n{proposal.get('diagnosis','')[:1200]}\n\n"
            f"Your proposed ops:\n" + "\n".join(ops_lines) +
            f"\n\nYour proposed p_ops: {json.dumps(proposal.get('p_ops') or [], ensure_ascii=False)}\n\n"
            f"Held-out measurement (same items across conditions):\n"
            f"{_fmt_measure(measure, tasks)}\n{hist}\n\n"
            "Decide what to apply.")


def validate_application(raw, n_ops):
    """Coerce the Teacher's apply decision; None on malformed (caller no-ops)."""
    if not isinstance(raw, dict):
        return None
    try:
        idx = sorted({int(i) for i in (raw.get("apply_op_indices") or [])})
    except (TypeError, ValueError):
        return None
    if any(i < 0 or i >= n_ops for i in idx):
        return None
    p_ops = []
    for op in raw.get("p_ops") or []:
        try:
            p_ops.append({"task": str(op["task"]),
                          "p": max(0.0, min(0.5, float(op["p"])))})
        except (KeyError, TypeError, ValueError):
            return None
    return {"rationale": str(raw.get("rationale") or "")[:2000],
            "apply_op_indices": idx, "p_ops": p_ops}


def decide_application(client, proposal, measure, tasks, history_tail=None,
                       model="gpt-5.5", max_completion_tokens=3000):
    """(application|None, raw_text). One call, no tools — the investigation already
    happened; this is judgment over a small table."""
    user = build_apply_prompt(proposal, measure, tasks, history_tail)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": APPLY_SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens)
    text = r.choices[0].message.content or ""
    try:
        raw = json.loads(text)
    except ValueError:
        return None, text
    return validate_application(raw, len(proposal.get("item_ops") or [])), text
