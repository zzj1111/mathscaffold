"""Math-domain tools + system prompt for the investigative Teacher (QuestA-style
hint-ratio training). Same contract as alfworld.py: QUERIES ONLY, size-bounded.

Recorder rows: {qid, ratio, score, text?}. data.scaffold is the ratio state
{qid: {"r", "state", "hist"}}. A problem's window group = its rows this cycle.
"""
from __future__ import annotations

MAX_TRACES_PER_CALL = 6
TEXT_HEAD = 1000

BUCKETS = ((0.0, 0.0), (0.0, 25.0), (25.0, 50.0), (50.0, 90.0))


def _bucket(r):
    if r <= 0:
        return "r=0"
    for lo, hi in BUCKETS[1:]:
        if lo < r <= hi:
            return f"{int(lo)}<r<={int(hi)}"
    return ">90"


def _groups(rows):
    g = {}
    for r in rows:
        g.setdefault(r.get("qid"), []).append(r)
    return g


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Aggregate counters for this training window: per-ratio-bucket "
                       "problem counts and group composition (all_fail/mixed/all_pass "
                       "— mixed groups are the only ones that yield gradient), ratio "
                       "state summary, graduation counts, recent decisions. Cheap; "
                       "start here.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_problems",
        "description": "List problems in this window matching objective filters, with "
                       "qid, current ratio, and successes/attempts.",
        "parameters": {"type": "object", "properties": {
            "outcome": {"type": "string", "enum": ["all_fail", "mixed", "all_pass"]},
            "r_min": {"type": "number"}, "r_max": {"type": "number"},
            "n": {"type": "integer", "minimum": 1, "maximum": 40},
            "offset": {"type": "integer", "minimum": 0}},
            "required": ["outcome"]}}},
    {"type": "function", "function": {
        "name": "get_traces",
        "description": "Rollout excerpts (model output head, with score) for ONE "
                       f"problem; at most {MAX_TRACES_PER_CALL} per call.",
        "parameters": {"type": "object", "properties": {
            "qid": {"type": "string"},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_TRACES_PER_CALL}},
            "required": ["qid"]}}},
]


def dispatch(data, name, args):
    rows = data.rows
    state = data.scaffold or {}
    groups = _groups(rows)
    if name == "get_stats":
        probs = state.get("problems", state) if isinstance(state, dict) else {}
        text = (state or {}).get("text") or {}
        by_bucket = {}
        by_topic = {}
        for qid, g in groups.items():
            r = float((probs.get(qid) or {}).get("r", g[0].get("ratio") or 0))
            b = by_bucket.setdefault(_bucket(r), {"problems": 0, "all_fail": 0,
                                                  "mixed": 0, "all_pass": 0})
            succ = sum(1 for x in g if float(x.get("score") or 0) > 0)
            kind = "all_fail" if succ == 0 else ("all_pass" if succ == len(g) else "mixed")
            b["problems"] += 1
            b[kind] += 1
            # per-topic, split by whether the TEXT scaffold was injected (the coin is
            # per problem-cycle), mirroring the ALFWorld injected/bare composition
            topic = g[0].get("topic") or "other"
            side = "text" if g[0].get("text_inj") else "bare"
            t = by_topic.setdefault(topic, {})
            tt = t.setdefault(side, {"problems": 0, "all_fail": 0, "mixed": 0,
                                     "all_pass": 0})
            tt["problems"] += 1
            tt[kind] += 1
        states = {"active": 0, "graduated": 0}
        for h in probs.values():
            states[h.get("state") or "active"] = states.get(h.get("state") or "active", 0) + 1
        return {"window_problems": len(groups),
                "by_ratio_bucket": by_bucket,
                "by_topic_text_split": by_topic,
                "text_scaffold": {"items": {sc: [{"id": i["id"], "kind": i["kind"],
                                                  "text": i["text"][:120]}
                                                 for i in v]
                                            for sc, v in (text.get("items") or {}).items() if v},
                                  "p": text.get("p")},
                "ratio_state": states,
                "recent_decisions": (getattr(data, "state", None) or {}).get("recent", [])}
    if name == "get_problems":
        probs = state.get("problems", state) if isinstance(state, dict) else {}
        want = args.get("outcome")
        rmin = float(args.get("r_min") or 0)
        rmax = float(args.get("r_max") or 100)
        out = []
        for qid, g in sorted(groups.items()):
            succ = sum(1 for x in g if float(x.get("score") or 0) > 0)
            kind = "all_fail" if succ == 0 else ("all_pass" if succ == len(g) else "mixed")
            r = float((probs.get(qid) or {}).get("r", g[0].get("ratio") or 0))
            if kind == want and rmin <= r <= rmax:
                out.append({"qid": qid, "r": r, "succ": succ, "n": len(g)})
        off = int(args.get("offset") or 0)
        n = min(int(args.get("n") or 20), 40)
        return {"total_matching": len(out), "problems": out[off:off + n]}
    if name == "get_traces":
        g = groups.get(str(args.get("qid"))) or []
        n = min(int(args.get("n") or 3), MAX_TRACES_PER_CALL)
        return {"attempts": [{"score": x.get("score"),
                              "ratio": x.get("ratio"),
                              "text_head": str(x.get("text") or "")[:TEXT_HEAD]}
                             for x in g[:n]],
                "note": "text is the model output head; absent if text logging is off"}
    return {"error": f"unknown tool {name}"}


MATH_SYSTEM = """You are the Teacher in an automated RL training run. A small policy
model is trained with GRPO on hard competition math problems; it is ALWAYS evaluated
hint-free, so anything you inject into training prompts is an exploration device —
what it elicits must survive into the weights to count.

You control TWO independent scaffold families:
1. HINT PREFIX (per problem): the first r% (by characters) of the reference solution,
   spliced as '## Hint.'. r=0 is a bare probe; success there means genuinely learned.
2. TEXT NOTES (per topic): reusable items spliced as '## Notes.' into that topic's
   prompts under a per-topic probability p. Kinds: "skill" (a strategy/fact),
   "example" (a short worked example), "plan" (a solution skeleton). General-scope
   items ride along with every topic's block.

A problem's group is its sampled rollouts for one prompt; if all score the same, the
group yields no gradient. Interventions only matter where groups still yield
gradient: all-pass at some dose has nothing left to teach there; all-fail means the
dose is not strong enough (or the problem is beyond any dose).

INVESTIGATION: read-only tools over this cycle's rollouts, the ratio state and the
text scaffold. The user message states your EXACT budgets; every result carries
`_budget_calls_remaining`. Investigate, then commit to ONE final decision.

Return, as your FINAL message (no tool call), ONLY this JSON:
{"diagnosis": "<your reasoning>",
 "ratio_ops": [
   {"scope": "bucket", "outcome": "all_fail"|"all_pass"|"mixed",
    "r_min": <0..90>, "r_max": <0..90>, "delta": <-20..20>} |
   {"scope": "qid", "qid": "...", "set": <0..90>}],
 "item_ops": [{"op": "add", "scope": "<general|algebra|geometry|number_theory|combinatorics|other>",
               "kind": "skill"|"example"|"plan", "text": "..."} |
              {"op": "update", "id": "...", "kind": "...", "text": "..."} |
              {"op": "delete", "id": "..."}],
 "p_ops": [{"topic": "<topic>", "p": <0..0.5>}]}
Empty ops means no intervention this cycle.

HARD CONSTRAINTS (violations are clamped or the op family is voided, never fatal):
- ratio: at most 4 bucket ops and 16 qid ops; delta clamped to +-20; r in [0, 90];
- text: at most 3 add/update ops per cycle (deletes free); skill/plan <= 500 chars,
  example <= 1500; no duplicate text in a scope; p in [0, 0.5];
- graduation bookkeeping is mechanical (bare success graduates; bare failure after
  graduation relapses) — you steer doses for ACTIVE problems only."""
