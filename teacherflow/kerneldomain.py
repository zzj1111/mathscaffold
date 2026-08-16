"""Kernel-domain (CUDA/Triton, cudascaffold) tools + prompt for the investigative Teacher.

Recorder rows (cudaforge reward, CUDASCAFFOLD_ROLLOUT_LOG): {task_name, category, level,
data_source, injected, correctness, speedup, reward, fail_kind, fail_msg, code}. A group
is one task_name's candidates in the window (same prompt -> same GRPO group). Success is
correctness==1 (the reward may still be 0 under the rubric's hacking flag; both shown).

The system prompt = cudascaffold's own domain prompt (facts, item rules, exact JSON
grammar the arm validates) + the shared all-fail-first mechanism block + investigation
rules. Same contract as the other domains: read-only, budgeted, one JSON decision.
"""
from __future__ import annotations

MAX_TRACES_PER_CALL = 8
CODE_HEAD = 900
CODE_TAIL = 500


def _ok(r):
    return int(r.get("correctness") or 0) == 1


def _groups(rows):
    g = {}
    for r in rows:
        g.setdefault(r.get("task_name"), []).append(r)
    return g


def _trace(r):
    code = str(r.get("code") or "")
    if len(code) > CODE_HEAD + CODE_TAIL:
        code = code[:CODE_HEAD] + "\n...[middle truncated]...\n" + code[-CODE_TAIL:]
    return {"task": r.get("task_name"), "category": r.get("category"), "level": r.get("level"),
            "injected": bool(r.get("injected")), "correct": _ok(r),
            "speedup": r.get("speedup"), "reward": r.get("reward"),
            "fail_kind": r.get("fail_kind"), "fail_msg": r.get("fail_msg"),
            "code_excerpt": code}


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Aggregate counters for this window: per-category candidates and "
                       "correctness split by injected/bare with group composition "
                       "(all_fail/mixed/all_pass — mixed groups are the only ones that "
                       "yield gradient), failure-kind histogram, ALL-FAIL tracking "
                       "(new/recurring/escaped, per category, split by whether text reached "
                       "them), current scaffold and recent decisions. Cheap; start here.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_traces",
        "description": "Scored candidates matching objective filters: failure kind + error "
                       "message + code excerpt (head/tail). Paged; at most "
                       f"{MAX_TRACES_PER_CALL} per call (4 when all_fail_only).",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"},
            "correct": {"type": "integer", "enum": [0, 1]},
            "injected": {"type": "boolean"},
            "fail_kind": {"type": "string"},
            "all_fail_only": {"type": "boolean",
                              "description": "only candidates from ALL-FAIL groups (the "
                                             "zero-gradient tasks); capped at 4 per call"},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_TRACES_PER_CALL},
            "offset": {"type": "integer", "minimum": 0}}}}},
    {"type": "function", "function": {
        "name": "get_group",
        "description": "Every candidate of ONE task (task_name), so within-group variance "
                       "is visible: what a correct sibling did differently.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"}}, "required": ["task"]}}},
]


def dispatch(data, name, args):
    rows = data.rows
    groups = _groups(rows)
    if name == "get_stats":
        cats = sorted({r.get("category") for r in rows if r.get("category")})
        per = {}
        for cat in cats:
            sub = [r for r in rows if r.get("category") == cat]
            d = {"candidates": len(sub)}
            for side in (True, False):
                part = [r for r in sub if bool(r.get("injected")) == side]
                gpart = [g for t, g in groups.items()
                         if g and g[0].get("category") == cat and bool(g[0].get("injected")) == side]
                comp = {"all_pass": 0, "mixed": 0, "all_fail": 0}
                for g in gpart:
                    s = [_ok(x) for x in g]
                    comp["all_pass" if all(s) else ("all_fail" if not any(s) else "mixed")] += 1
                sp = [float(r.get("speedup") or 0) for r in part if _ok(r)]
                d["injected" if side else "bare"] = {
                    "n": len(part),
                    "correct_rate": round(sum(_ok(r) for r in part) / len(part), 4) if part else None,
                    "median_speedup_when_correct": round(sorted(sp)[len(sp) // 2], 3) if sp else None,
                    "groups": {"n": len(gpart), **comp}}
            fk = {}
            for r in sub:
                if not _ok(r):
                    fk[r.get("fail_kind") or "?"] = fk.get(r.get("fail_kind") or "?", 0) + 1
            d["failure_kinds"] = dict(sorted(fk.items(), key=lambda kv: -kv[1])[:8])
            per[cat] = d
        # all-fail tracking vs previous window
        def _fails(rs):
            g = _groups(rs)
            f, seen, side, cat_of = set(), set(), {}, {}
            for t, grp in g.items():
                if len(grp) < 2:
                    continue
                seen.add(t)
                cat_of[t] = grp[0].get("category")
                if not any(_ok(x) for x in grp):
                    f.add(t)
                    side[t] = bool(grp[0].get("injected"))
            return f, seen, side, cat_of
        cur_f, cur_seen, cur_side, cat_of = _fails(rows)
        prev_f, prev_seen, _, _ = _fails(getattr(data, "prev_rows", []) or [])
        recurring = sorted(cur_f & prev_f)
        escaped = sorted(prev_f & (cur_seen - cur_f))
        by_cat = {}
        blank = lambda: {"all_fail_now": 0, "recurring": 0, "all_fail_injected": 0, "all_fail_bare": 0}
        for t in cur_f:
            c = by_cat.setdefault(cat_of.get(t, "?"), blank())
            c["all_fail_now"] += 1
            c["all_fail_injected" if cur_side.get(t) else "all_fail_bare"] += 1
        for t in recurring:
            by_cat.setdefault(cat_of.get(t, "?"), blank())["recurring"] += 1
        sc = data.scaffold or {}
        return {"per_category": per,
                "all_fail_tracking": {
                    "all_fail_now": len(cur_f), "recurring_all_fail": len(recurring),
                    "escaped_since_last_window": len(escaped),
                    "last_window_all_fail_not_reseen": len(prev_f - cur_seen),
                    "by_category": by_cat, "recurring_tasks": recurring[:12],
                    "note": "recurring all-fail = zero gradient twice in a row; RL cannot move "
                            "these on its own. escaped = all-fail then, >=1 correct now. "
                            "all_fail_bare = text never reached them (dose question); "
                            "all_fail_injected = text was there and did not unlock them "
                            "(content question)."},
                "scaffold": {"items": sc.get("items"), "p_task": sc.get("p_task"),
                             "general_skill": sc.get("general_skill")},
                "recent_decisions": [{"cycle": h.get("cycle"), "sr_before": h.get("sr_before"),
                                      "verdict": h.get("verdict")}
                                     for h in ((data.state or {}).get("decision_history") or [])[-8:]]}
    if name == "get_traces":
        sub = list(rows)
        if args.get("category"):
            sub = [r for r in sub if r.get("category") == args["category"]]
        if args.get("correct") is not None:
            sub = [r for r in sub if _ok(r) == bool(args["correct"])]
        if args.get("injected") is not None:
            sub = [r for r in sub if bool(r.get("injected")) == bool(args["injected"])]
        if args.get("fail_kind"):
            sub = [r for r in sub if (r.get("fail_kind") or "") == args["fail_kind"]]
        n = min(int(args.get("n") or 4), MAX_TRACES_PER_CALL)
        if args.get("all_fail_only"):
            af = {t for t, g in groups.items() if len(g) >= 2 and not any(_ok(x) for x in g)}
            sub = [r for r in sub if r.get("task_name") in af]
            n = min(n, 4)
        off = int(args.get("offset") or 0)
        return {"total_matching": len(sub), "traces": [_trace(r) for r in sub[off:off + n]]}
    if name == "get_group":
        g = groups.get(str(args.get("task"))) or []
        return {"candidates": [_trace(r) for r in g[:MAX_TRACES_PER_CALL]],
                "note": None if g else "task not found in this window"}
    return {"error": f"unknown tool {name}"}


MECHANISM_BLOCK = """
A group is one task's candidates for one prompt; if all of them score the same, the group
yields no gradient. Injection can only shape behavior where groups still yield gradient:
where nearly all groups are all-pass, injected text has no signal left to shape and only
carries the train/eval distribution-shift cost.
YOUR PRIMARY OBJECTIVE IS THE ALL-FAIL GROUP. Mixed groups already carry gradient — plain
RL learns those by itself, and support there is at best a mild accelerant that has
repeatedly failed to survive into scaffold-free evaluation. All-pass groups are already
learned. The one place RL cannot move on its own is the ALL-FAIL group: zero correct
candidates, zero gradient, and it stays that way unless something changes the sampling.
Judge every intervention by whether it can turn all-fail groups into groups with at least
one correct candidate (that is when gradient appears), and prefer that over polishing
categories that are already mostly solved. get_stats reports all-fail tracking (new vs
recurring vs escaped, per category, split by whether text reached them): recurring all-fail
tasks are where support is the ONLY lever; escaped ones tell you what unlocked them;
all_fail_bare is a dose question, all_fail_injected a content question. Read the failed
candidates of all-fail groups (get_traces with all_fail_only=true, or get_group on a
recurring task) — their fail_kind / error message / code excerpt — to find the missing
piece before writing text. Text that names the specific missing move (the API misuse, the
shape/stride assumption, the launch/config mistake, the correctness check skipped) is what
can unlock them; generic advice cannot.

INVESTIGATION: you have read-only tools over this cycle's scored candidates plus aggregate
counters and your own decision history. The user message states your EXACT budgets (tool
calls / evidence characters); every result carries `_budget_calls_remaining`. Investigate,
then commit to ONE final decision (the JSON grammar below).
"""


def build_system(domain_prompt):
    """cudascaffold's own domain prompt (facts + item rules + exact JSON grammar) with the
    shared mechanism/investigation block inserted before its output-format section."""
    marker = "Return ONLY JSON:"
    i = domain_prompt.find(marker)
    if i == -1:
        return domain_prompt + "\n" + MECHANISM_BLOCK
    return domain_prompt[:i] + MECHANISM_BLOCK + "\n" + domain_prompt[i:]
