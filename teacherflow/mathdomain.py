"""Math-domain tools + system prompt for the investigative Teacher (QuestA-style
hint-ratio training). Same contract as alfworld.py: QUERIES ONLY, size-bounded.

Recorder rows: {qid, ratio, score, text?}. data.scaffold is the ratio state
{qid: {"r", "state", "hist"}}. A problem's window group = its rows this cycle.
"""
from __future__ import annotations

MAX_TRACES_PER_CALL = 6
TEXT_HEAD = 1500
TEXT_TAIL = 800
WINDOW = 2500

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
        "description": "ONE problem in full (statement, reference solution, up to "
                       f"{MAX_TRACES_PER_CALL} rollout excerpts with scores) when qid is "
                       "given; or, with all_fail_batch=true and no qid, a compact sweep "
                       "of up to 4 ALL-FAIL problems (statement + 1 failed excerpt each) "
                       "— the fast way to look for the missing piece across the "
                       "zero-gradient set. Excerpts default to head+tail of each output; "
                       "pass char_offset to read a window from the MIDDLE of the outputs "
                       "(e.g. where the derivation goes wrong) — each attempt reports "
                       "its total length so you know where to look.",
        "parameters": {"type": "object", "properties": {
            "qid": {"type": "string"},
            "all_fail_batch": {"type": "boolean"},
            "offset": {"type": "integer", "minimum": 0,
                       "description": "for all_fail_batch: page through problems"},
            "char_offset": {"type": "integer", "minimum": 0,
                            "description": "for qid mode: start of a "
                                           f"{WINDOW}-char window into each output"},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_TRACES_PER_CALL}}}}},
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
            # split by whether the TEXT scaffold was injected (coin per
            # problem-cycle), mirroring the ALFWorld injected/bare composition
            side = "text" if g[0].get("text_inj") else "bare"
            tt = by_topic.setdefault(side, {"problems": 0, "all_fail": 0,
                                            "mixed": 0, "all_pass": 0})
            tt["problems"] += 1
            tt[kind] += 1
        states = {"active": 0, "graduated": 0}
        for h in probs.values():
            states[h.get("state") or "active"] = states.get(h.get("state") or "active", 0) + 1
        # last window's all-fail problems: fate now (escaped = >=1 success this window)
        prev = _groups(getattr(data, "prev_rows", []) or [])
        prev_fail = {q for q, g in prev.items()
                     if g and all(float(x.get("score") or 0) <= 0 for x in g)}
        reseen = {q for q in prev_fail if q in groups}
        escaped = {q for q in reseen
                   if any(float(x.get("score") or 0) > 0 for x in groups[q])}
        af_now = {q for q, g in groups.items()
                  if all(float(x.get("score") or 0) <= 0 for x in g)}
        af_text = sum(1 for q in af_now if groups[q][0].get("text_inj"))
        esc_text = sum(1 for q in escaped if groups[q][0].get("text_inj"))
        fate = {"all_fail_now": len(af_now),
                # text was in the prompt and did NOT unlock them (content question) vs
                # text never reached them (dose question)
                "all_fail_now_with_text": af_text,
                "all_fail_now_bare": len(af_now) - af_text,
                "last_window_all_fail": len(prev_fail), "reseen_now": len(reseen),
                "escaped": len(escaped), "escaped_with_text": esc_text,
                "still_all_fail": len(reseen - escaped),
                "escaped_ratios_now": sorted({float((probs.get(q) or {}).get("r", 0))
                                              for q in escaped})[:8]}
        return {"window_problems": len(groups),
                "by_ratio_bucket": by_bucket,
                "all_fail_fate": fate,
                "text_vs_bare": by_topic,
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
    if name == "get_traces" and args.get("all_fail_batch") and not args.get("qid"):
        probs = state.get("problems", state) if isinstance(state, dict) else {}
        meta_all = getattr(data, "problems", None) or {}
        af = sorted(q for q, g in groups.items()
                    if all(float(x.get("score") or 0) <= 0 for x in g))
        off = int(args.get("offset") or 0)
        out = []
        for q in af[off:off + 4]:
            m = meta_all.get(q) or {}
            g = groups[q]
            out.append({"qid": q, "r": float((probs.get(q) or {}).get("r", g[0].get("ratio") or 0)),
                        "text_inj": bool(g[0].get("text_inj")),
                        "problem": str(m.get("problem") or "")[:500],
                        "reference_tail": str(m.get("solution") or "")[-400:],
                        "one_failed_excerpt": str(g[0].get("text") or "")[-TEXT_TAIL:]})
        return {"total_all_fail": len(af), "offset": off, "problems": out}
    if name == "get_traces":
        qid = str(args.get("qid"))
        g = groups.get(qid) or []
        n = min(int(args.get("n") or 3), MAX_TRACES_PER_CALL)
        meta = (getattr(data, "problems", None) or {}).get(qid) or {}
        co = args.get("char_offset")
        def _excerpt(t):
            t = str(t or "")
            if co is not None:
                o = max(0, min(int(co), max(0, len(t) - 1)))
                return {"len": len(t), "window_start": o, "text": t[o:o + WINDOW]}
            if len(t) <= TEXT_HEAD + TEXT_TAIL:
                return {"len": len(t), "text": t}
            return {"len": len(t), "head": t[:TEXT_HEAD], "tail": t[-TEXT_TAIL:],
                    "note": f"middle {len(t) - TEXT_HEAD - TEXT_TAIL} chars omitted; "
                            f"use char_offset to read it"}
        return {"problem": str(meta.get("problem") or "")[:900],
                "reference_solution": str(meta.get("solution") or "")[:1100],
                "attempts": [{"score": x.get("score"), "ratio": x.get("ratio"),
                              **_excerpt(x.get("text"))} for x in g[:n]]}
    return {"error": f"unknown tool {name}"}


MATH_SYSTEM = """You are the Teacher in an automated RL training run. A small policy
model is trained with GRPO on hard competition math problems; it is ALWAYS evaluated
hint-free, so anything you inject into training prompts is an exploration device —
what it elicits must survive into the weights to count.

You control TWO independent scaffold families:
1. HINT PREFIX (per problem): the first r% (by characters) of the reference solution,
   spliced as '## Hint.'. r=0 is a bare probe; success there means genuinely learned.
2. TEXT NOTES (general): reusable items spliced as '## Notes.' into training
   prompts under ONE probability p. Kinds: "skill" (a strategy/fact), "example"
   (a short worked example), "plan" (a solution skeleton). Math problems do not
   factor into mechanical categories, so notes are global; per-problem targeting
   belongs to the hint ratio.

A problem's group is its sampled rollouts for one prompt; if all score the same, the
group yields no gradient. YOUR PRIMARY OBJECTIVE IS THE ALL-FAIL GROUP. Mixed groups
already carry gradient — plain RL learns those by itself. All-pass groups are already
learned at that dose. The one place RL cannot move on its own is the all-fail group:
zero successes, zero gradient, and it stays that way unless the dose changes the
sampling. Judge every intervention by whether it can turn all-fail groups into groups
with at least one success (that is when gradient appears): raise the hint ratio on
all-fail problems until they become mixed, then let RL learn them and anneal. Prefer
that over polishing problems that are already mostly solved. get_stats reports each
ratio bucket's all-fail/mixed/all-pass split, all-fail problems split by whether text
notes were in their prompt (bare = dose question; with text = content question), and
last window's all-fail problems' fate this window (escaped vs still all-fail) — what
unlocked escaped ones is your most direct evidence of the right dose. Use
get_traces(all_fail_batch=true) to sweep the zero-gradient set quickly, then
get_traces(qid) on the ones worth a full read.

INVESTIGATION: read-only tools over this cycle's rollouts, the ratio state and the
text scaffold. The user message states your EXACT budgets; every result carries
`_budget_calls_remaining`. Investigate, then commit to ONE final decision.

Return, as your FINAL message (no tool call), ONLY this JSON:
{"diagnosis": "<your reasoning>",
 "ratio_ops": [
   {"scope": "bucket", "outcome": "all_fail"|"all_pass"|"mixed",
    "r_min": <0..90>, "r_max": <0..90>, "delta": <-20..20>} |
   {"scope": "qid", "qid": "...", "set": <0..90>}],
 "item_ops": [{"op": "add", "scope": "general",
               "kind": "skill"|"example"|"plan", "text": "..."} |
              {"op": "update", "id": "...", "kind": "...", "text": "..."} |
              {"op": "delete", "id": "..."}],
 "p_ops": [{"p": <0..0.5>}]}
Empty ops means no intervention this cycle.

HARD CONSTRAINTS (violations are clamped or the op family is voided, never fatal):
- ratio: at most 4 bucket ops and 16 qid ops; delta clamped to +-20; r in [0, 90];
- text: at most 3 add/update ops per cycle (deletes free); skill/plan <= 500 chars,
  example <= 1500; no duplicates; ONE global p in [0, 0.5];
- graduation bookkeeping is mechanical (bare success graduates; bare failure after
  graduation relapses) — you steer doses for ACTIVE problems only."""
