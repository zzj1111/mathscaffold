"""The Teacher's data-access tools. QUERIES ONLY — no analysis, no classification.

Every result is JSON-safe and size-bounded. Filters accept only OBJECTIVE fields
(category, injected, success, whether an <answer> tag exists); nothing here interprets
a trajectory. Adding an interpreting tool is a design regression (see README).
"""
from __future__ import annotations

MAX_TRACES_PER_CALL = 10
O_TRIM = 300


def _has_answer(row):
    return any(str(s.get("a", "")).startswith("answer: ") for s in row.get("steps") or [])


def _trace(row):
    return {"qid": row.get("qid"), "gold": list(row.get("gold") or []),
            "injected": bool(row.get("injected")), "success": row.get("success"),
            "steps": [{"a": s.get("a"), "o": str(s.get("o") or "")[:O_TRIM],
                       "v": s.get("v", True)} for s in row.get("steps") or []]}


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Aggregate counters for the current window: per-category episode "
                       "counts and success split by injected/bare, complete-group counts "
                       "with all-fail/all-succeed totals, plus the current scaffold items, "
                       "p_task and the held-out eval trajectory. Cheap; start here.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_traces",
        "description": "Raw episode trajectories (searches, retrieved passages, answers, "
                       "gold) matching objective filters. Paged; at most "
                       f"{MAX_TRACES_PER_CALL} per call.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"},
            "success": {"type": "integer", "enum": [0, 1],
                        "description": "filter by episode outcome"},
            "injected": {"type": "boolean"},
            "has_answer_tag": {"type": "boolean",
                               "description": "whether the episode ever emitted <answer>"},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_TRACES_PER_CALL},
            "offset": {"type": "integer", "minimum": 0}},
            "required": ["category"]}}},
    {"type": "function", "function": {
        "name": "get_group",
        "description": "Every rollout of ONE question's group (same qid), so within-group "
                       "variance is visible: what the successful sibling did differently.",
        "parameters": {"type": "object", "properties": {
            "qid": {"type": "string"}}, "required": ["qid"]}}},
]


def dispatch(data, name, args):
    if name == "get_stats":
        out = {}
        for cat in data.categories():
            rows = [r for r in data.rows if r.get("data_source") == cat]
            d = {"episodes": len(rows)}
            for side in (True, False):
                sub = [r for r in rows if bool(r.get("injected")) == side]
                key = "injected" if side else "bare"
                d[key] = {"n": len(sub),
                          "success": round(sum(float(r.get("success") or 0) > 0
                                               for r in sub) / len(sub), 4) if sub else None}
            gs = [g for g in data.groups.values()
                  if g and g[0].get("data_source") == cat]
            complete = [g for g in gs if len(g) >= 2]
            allf = sum(1 for g in complete
                       if all(float(r.get("success") or 0) <= 0 for r in g))
            alls = sum(1 for g in complete
                       if all(float(r.get("success") or 0) > 0 for r in g))
            d["groups"] = {"complete": len(complete), "all_fail": allf,
                           "all_succeed": alls}
            out[cat] = d
        return {"per_category": out,
                "scaffold": {"items": data.scaffold.get("items"),
                             "p_task": data.scaffold.get("p_task")},
                "valid_seen_history": (data.state.get("decision_history") or [])
                                       and [{"cycle": h.get("cycle"),
                                             "sr_before": h.get("sr_before"),
                                             "verdict": h.get("verdict")}
                                            for h in data.state["decision_history"][-8:]]}
    if name == "get_traces":
        cat = args.get("category")
        rows = [r for r in data.rows if r.get("data_source") == cat]
        if "success" in args and args["success"] is not None:
            want = bool(args["success"])
            rows = [r for r in rows if (float(r.get("success") or 0) > 0) == want]
        if "injected" in args and args["injected"] is not None:
            rows = [r for r in rows if bool(r.get("injected")) == bool(args["injected"])]
        if "has_answer_tag" in args and args["has_answer_tag"] is not None:
            rows = [r for r in rows if _has_answer(r) == bool(args["has_answer_tag"])]
        off = int(args.get("offset") or 0)
        n = min(int(args.get("n") or 5), MAX_TRACES_PER_CALL)
        return {"total_matching": len(rows),
                "traces": [_trace(r) for r in rows[off:off + n]]}
    if name == "get_group":
        qid = str(args.get("qid"))
        for g in data.groups.values():
            if g and str(g[0].get("qid")) == qid:
                return {"rollouts": [_trace(r) for r in g]}
        return {"rollouts": [], "note": "qid not found in this window"}
    return {"error": f"unknown tool {name}"}
