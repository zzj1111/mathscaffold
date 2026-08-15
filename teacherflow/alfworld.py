"""ALFWorld-domain tools + system prompt for the investigative Teacher.

Same contract as tools.py (QUERIES ONLY, size-bounded, objective filters), adapted to
the ALFWorld recorder schema: rows carry {uid, task_type, gamefile, injected,
success, steps:[{a, o, v}]}. A "group" is one game instance's rollouts (grouped by
gamefile within the window), so within-group variance is visible exactly as in the
Search domain.
"""
from __future__ import annotations

MAX_TRACES_PER_CALL = 8
O_TRIM = 240
STEP_CAP = 30           # ALFWorld episodes run to 50 steps; cap what one trace costs

CATEGORIES = ("pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
              "pick_clean_then_place_in_recep")


def _game(row):
    gf = str(row.get("gamefile") or "")
    # keep the informative tail: task dir + trial dir
    return "/".join(gf.rstrip("/").split("/")[-2:])


def _trace(row):
    steps = row.get("steps") or []
    return {"game": _game(row), "task_type": row.get("task_type"),
            "injected": bool(row.get("injected")), "success": row.get("success"),
            "n_steps": len(steps),
            "steps": [{"a": s.get("a"), "o": str(s.get("o") or "")[:O_TRIM],
                       "v": s.get("v", True)} for s in steps[:STEP_CAP]]}


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Aggregate counters for the current window: per-task-type "
                       "episode counts and success split by injected/bare, complete "
                       "game-groups with all-fail/all-succeed totals, plus the current "
                       "scaffold items, p_task and recent decision history. Cheap; "
                       "start here.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_traces",
        "description": "Raw episode trajectories (actions, environment observations, "
                       "validity flags) matching objective filters. Paged; at most "
                       f"{MAX_TRACES_PER_CALL} per call, first {STEP_CAP} steps each.",
        "parameters": {"type": "object", "properties": {
            "task_type": {"type": "string", "enum": list(CATEGORIES)},
            "success": {"type": "integer", "enum": [0, 1]},
            "injected": {"type": "boolean"},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_TRACES_PER_CALL},
            "offset": {"type": "integer", "minimum": 0}},
            "required": ["task_type"]}}},
    {"type": "function", "function": {
        "name": "get_group",
        "description": "Every rollout of ONE game instance (same gamefile suffix as "
                       "returned in `game`), so within-group variance is visible: what "
                       "the successful sibling did differently.",
        "parameters": {"type": "object", "properties": {
            "game": {"type": "string"}}, "required": ["game"]}}},
]


def dispatch(data, name, args):
    rows = data.rows
    if name == "get_stats":
        out = {}
        for cat in sorted({r.get("task_type") for r in rows if r.get("task_type")}):
            sub = [r for r in rows if r.get("task_type") == cat]
            d = {"episodes": len(sub)}
            # groups keyed by uid (one uid = one game instance's rollouts); the
            # injection coin is group-level, so a group is wholly injected or bare.
            groups = {}
            for r in sub:
                groups.setdefault(r.get("uid"), []).append(r)
            complete = [g for g in groups.values() if len(g) >= 2]
            for side in (True, False):
                part = [r for r in sub if bool(r.get("injected")) == side]
                key = "injected" if side else "bare"
                gpart = [g for g in complete if bool(g[0].get("injected")) == side]
                comp = {"all_succeed": 0, "mixed": 0, "all_fail": 0}
                for g in gpart:
                    succ = [float(r.get("success") or 0) > 0 for r in g]
                    comp["all_succeed" if all(succ) else
                         ("all_fail" if not any(succ) else "mixed")] += 1
                d[key] = {"n": len(part),
                          "success": round(sum(float(r.get("success") or 0) > 0
                                               for r in part) / len(part), 4) if part else None,
                          # mixed groups are the only ones that yield gradient
                          "groups": {"n": len(gpart), **comp}}
            d["groups"] = {"complete": len(complete),
                           "all_fail": sum(1 for g in complete
                                           if all(float(r.get("success") or 0) <= 0 for r in g)),
                           "all_succeed": sum(1 for g in complete
                                              if all(float(r.get("success") or 0) > 0 for r in g))}
            out[cat] = d
        # ALL-FAIL TRACKING: which games are all-fail this window, whether they
        # were also all-fail last time seen (recurring = RL is stuck on them, no
        # gradient at all), and whether last window's all-fail games escaped now.
        def _fail_games(rs):
            g = {}
            for r in rs:
                g.setdefault(r.get("gamefile"), []).append(r)
            fails, seen = set(), set()
            for gf, grp in g.items():
                if len(grp) < 2:
                    continue
                seen.add(gf)
                if all(float(r.get("success") or 0) <= 0 for r in grp):
                    fails.add(gf)
            return fails, seen
        cur_f, cur_seen = _fail_games(rows)
        prev_f, prev_seen = _fail_games(getattr(data, "prev_rows", []) or [])
        recurring = sorted(cur_f & prev_f)
        escaped = sorted(prev_f & (cur_seen - cur_f))
        still_unseen = len(prev_f - cur_seen)
        def _cat_of(gf):
            for c in CATEGORIES:
                if c in str(gf):
                    return c
            return "?"
        by_cat = {}
        for gf in cur_f:
            by_cat.setdefault(_cat_of(gf), {"all_fail_now": 0, "recurring": 0})
            by_cat[_cat_of(gf)]["all_fail_now"] += 1
        for gf in recurring:
            by_cat[_cat_of(gf)]["recurring"] += 1
        tracking = {
            "all_fail_now": len(cur_f),
            "recurring_all_fail": len(recurring),
            "escaped_since_last_window": len(escaped),
            "last_window_all_fail_not_reseen": still_unseen,
            "by_category": by_cat,
            "recurring_games": [_game({"gamefile": g}) for g in recurring[:12]],
            "note": "recurring all-fail = zero gradient twice in a row; RL cannot "
                    "move these on its own. escaped = all-fail last time, at least "
                    "one success now (whatever changed in between worked)."}
        return {"per_task_type": out,
                "all_fail_tracking": tracking,
                "scaffold": {"items": data.scaffold.get("items"),
                             "p_task": data.scaffold.get("p_task")},
                # held-out measurement of the CURRENT scaffold, injected (raw numbers;
                # absent until the loop has produced one)
                "last_injected_heldout": data.state.get("last_injected_eval"),
                "recent_decisions": [{"cycle": h.get("cycle"),
                                      "sr_before": h.get("sr_before"),
                                      "verdict": h.get("verdict")}
                                     for h in (data.state.get("decision_history") or [])[-8:]]}
    if name == "get_traces":
        cat = args.get("task_type")
        sub = [r for r in rows if r.get("task_type") == cat]
        if "success" in args and args["success"] is not None:
            want = bool(args["success"])
            sub = [r for r in sub if (float(r.get("success") or 0) > 0) == want]
        if "injected" in args and args["injected"] is not None:
            sub = [r for r in sub if bool(r.get("injected")) == bool(args["injected"])]
        off = int(args.get("offset") or 0)
        n = min(int(args.get("n") or 4), MAX_TRACES_PER_CALL)
        return {"total_matching": len(sub), "traces": [_trace(r) for r in sub[off:off + n]]}
    if name == "get_group":
        want = str(args.get("game") or "")
        grp = [r for r in rows if _game(r) == want or want in _game(r)]
        if not grp:
            return {"rollouts": [], "note": "game not found in this window"}
        return {"rollouts": [_trace(r) for r in grp[:MAX_TRACES_PER_CALL]]}
    return {"error": f"unknown tool {name}"}


ALF_SYSTEM = """You are the Teacher in an automated RL training run. A small policy
model is trained with GRPO-family RL to complete household tasks in the ALFWorld
text environment: at each step it sees an observation and admissible commands and
must output one action (go to X, take A from B, put A in/on B, open/close, clean/
heat/cool A with B, use lamp, etc.), up to 50 steps per episode; success is binary
task completion. Task categories: pick_and_place, pick_two_obj_and_place,
look_at_obj_in_light, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep,
pick_clean_then_place_in_recep.

Your job: decide, once per training cycle, whether to change the scaffold text that
is injected into TRAINING prompts only (the policy is always evaluated bare), and
the per-category injection probabilities p (hard cap 0.5, at most +/-0.2 change per
cycle). Because the loss conditions on the injected prompt during training but
evaluation is bare, text is an exploration device: what it elicits must survive
into the weights.

A group is one game instance's rollouts; if all of them score the same, the group
yields no gradient. Injection can only shape behavior where groups still yield
gradient: in categories where nearly all groups are all-succeed, injected text has
no signal left to shape and only carries the train/eval distribution-shift cost.
YOUR PRIMARY OBJECTIVE IS THE ALL-FAIL GROUP. Mixed groups already carry gradient
— plain RL learns those by itself, and support there is at best a mild accelerant
that has repeatedly failed to survive into hint-free evaluation. All-succeed groups
are already learned. The one place RL cannot move on its own is the ALL-FAIL group:
zero successes, zero gradient, and it stays that way unless something changes the
sampling. Judge every intervention by whether it can turn all-fail groups into
groups with at least one success (that is when gradient appears), and prefer that
over polishing categories that are already mostly solved. get_stats reports all-fail
tracking (new vs recurring vs escaped, per category): recurring all-fail games are
where support is the ONLY lever; escaped ones tell you what unlocked them. Read the
failed trajectories of all-fail groups (get_traces with success=0, get_group) to find
the missing piece before writing text — text that names the specific missing step or
object handling is what can unlock them, generic advice cannot.
Text changes are gated: an A/B on held-out games must show your candidate beating
the current scaffold, else it is rejected and any bundled p change dies with it.

INVESTIGATION: you have read-only tools over this cycle's training episodes (raw
trajectories with actions and environment observations), plus aggregate counters and
your own decision history. The user message states your EXACT budgets for this cycle
(a tool-call count and an evidence-character total), and every tool result carries
`_budget_calls_remaining` so you always know how much is left — plan your
investigation against it. Call whatever you need, in any order; then commit to ONE
final decision. If you need nothing, you may decide immediately. When the budget
runs out you must decide with what you have.

Return, as your FINAL message (no tool call), ONLY this JSON:
{"diagnosis": "<your reasoning>",
 "item_ops": [{"op": "add", "scope": "<general|pick_and_place|pick_two_obj_and_place|look_at_obj_in_light|pick_heat_then_place_in_recep|pick_cool_then_place_in_recep|pick_clean_then_place_in_recep>",
               "kind": "skill", "text": "..."} |
              {"op": "update", "id": "...", "text": "..."} | {"op": "delete", "id": "..."}],
 "p_ops": [{"task": "<category>", "p": <0..0.5>}]}
Empty item_ops and p_ops means no intervention this cycle. Keep any text you write
concise and concrete; it is spliced into training prompts and costs context there.

HARD CONSTRAINTS (violating ANY voids the whole action into a no-op):
- at most 3 add/update ops per cycle (deletes are free) — prioritize;
- "kind" must be "skill" or "example"; item text at most 500 characters;
- no duplicate text within a scope; update/delete must name an id that exists in
  the current scaffold (ids are visible via get_stats)."""
