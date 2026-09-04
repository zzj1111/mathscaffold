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
            "succ_min": {"type": "number", "minimum": 0, "maximum": 1},
            "succ_max": {"type": "number", "minimum": 0, "maximum": 1},
            "n": {"type": "integer", "minimum": 1, "maximum": 40},
            "offset": {"type": "integer", "minimum": 0}},
            "required": ["outcome"]}}},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Per-cycle record of PAST cycles (most recent last), computed from "
                       "each cycle's own rollout log: served problems, mean/quantiles of the "
                       "hint ratio in force, group composition (all_fail/mixed/all_pass) by "
                       "ratio bucket and by text-injected vs bare side, the hint-free probe "
                       "of that cycle (held-out and in-training pass@1) when recorded, and a "
                       "one-line summary of the decision taken after that cycle.",
        "parameters": {"type": "object", "properties": {
            "last_n": {"type": "integer", "minimum": 1, "maximum": 20}}}}},
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
        # REVISITS: problems rotate (each cycle serves a fresh slice of the pool; a
        # problem returns every ~pool/served cycles), so the dose you set today acts at
        # the problem's NEXT visit. This block is the readout: for problems in this
        # window that were seen before, their previous visit's (outcome, r) -> now.
        wc = getattr(data, "window_cycle", None)
        revisits = {}
        n_revisit = 0
        for qid, g in groups.items():
            hist = (probs.get(qid) or {}).get("hist") or []
            prev_e = [e for e in hist if wc is None or int(e.get("cycle", -1)) < int(wc)]
            if not prev_e:
                continue
            e = prev_e[-1]
            n_revisit += 1
            ps, pn = int(e.get("succ") or 0), int(e.get("n") or 0)
            pk = "all_fail" if ps == 0 else ("all_pass" if ps == pn else "mixed")
            succ = sum(1 for x in g if float(x.get("score") or 0) > 0)
            nk = "all_fail" if succ == 0 else ("all_pass" if succ == len(g) else "mixed")
            r_prev = float(e.get("r") or 0)
            r_now = float(g[0].get("ratio") or 0)
            key = f"prev {pk} @r={r_prev:g} -> now @r={r_now:g}"
            d = revisits.setdefault(key, {"problems": 0, "all_fail": 0, "mixed": 0, "all_pass": 0})
            d["problems"] += 1
            d[nk] += 1
        # response length and "no final answer" share: the failure profile measured at
        # step 270 was 3/4 committed wrong answers after ~2x longer reasoning than correct
        # ones and 1/4 responses cut at the token cap; these counters let that be read
        # per bucket every cycle instead of from a one-off dump.
        import re as _re
        length = {}
        for qid, g in groups.items():
            r = float((probs.get(qid) or {}).get("r", g[0].get("ratio") or 0))
            for side_key in (_bucket(r), "text" if g[0].get("text_inj") else "bare"):
                d = length.setdefault(side_key, {"n": 0, "chars_sum": 0, "no_boxed": 0,
                                                 "chars_sum_correct": 0, "n_correct": 0,
                                                 "chars_sum_wrong": 0, "n_wrong": 0})
                for x in g:
                    t = str(x.get("text") or "")
                    ok = float(x.get("score") or 0) > 0
                    d["n"] += 1; d["chars_sum"] += len(t)
                    d["no_boxed"] += 0 if _re.search(r"\\boxed\{", t) else 1
                    if ok:
                        d["n_correct"] += 1; d["chars_sum_correct"] += len(t)
                    else:
                        d["n_wrong"] += 1; d["chars_sum_wrong"] += len(t)
        length_out = {k: {"rollouts": d["n"],
                          "mean_chars": round(d["chars_sum"] / d["n"]) if d["n"] else None,
                          "mean_chars_correct": round(d["chars_sum_correct"] / d["n_correct"]) if d["n_correct"] else None,
                          "mean_chars_wrong": round(d["chars_sum_wrong"] / d["n_wrong"]) if d["n_wrong"] else None,
                          "no_final_answer_share": round(d["no_boxed"] / d["n"], 3) if d["n"] else None}
                      for k, d in length.items()}
        return {"window_problems": len(groups),
                "by_ratio_bucket": by_bucket,
                "response_length": {"note": "chars of the student's response; no_final_answer_share = "
                                            "rollouts with no \\boxed{} (mostly cut at the token cap)",
                                    **length_out},
                "all_fail_fate": fate,
                "revisits": {"note": "problems seen in an earlier cycle: previous visit's "
                                     "outcome@dose -> this visit's outcome split; the "
                                     "direct evidence for whether a dose change worked",
                             "n_revisited": n_revisit, "by_transition": revisits},
                "text_vs_bare": by_topic,
                "text_scaffold": {"items": {sc: [{"id": i["id"], "kind": i["kind"],
                                                  "text": i["text"][:120]}
                                                 for i in v]
                                            for sc, v in (text.get("items") or {}).items() if v},
                                  "p": text.get("p")},
                "ratio_state": states,
                "high_dose": (lambda thr, frac, pr: {"thr": thr,
                    "n_above": sum(1 for h in pr.values() if float(h.get("r") or 0) > thr),
                    "budget_n": int(frac * len(pr))} if thr > 0 and pr else None)(
                    float(__import__("os").environ.get("MS_HIGH_DOSE_R", "0") or 0),
                    float(__import__("os").environ.get("MS_HIGH_DOSE_FRAC", "0.10")),
                    state.get("problems", state) if isinstance(state, dict) else {}),
                "recent_decisions": (getattr(data, "state", None) or {}).get("recent", [])}
    if name == "get_history":
        import os, json as _json
        work = getattr(data, "work_dir", None)
        wc = getattr(data, "window_cycle", None)
        if not work or wc is None:
            return {"cycles": [], "note": "no work directory / window cycle known"}
        n = max(1, min(20, int(args.get("last_n") or 8)))
        probes = {}
        bp = os.path.join(work, "bare_probe.jsonl")
        if os.path.exists(bp):
            for ln in open(bp):
                try:
                    rec = _json.loads(ln)
                    probes[int(rec.get("cycle"))] = rec
                except (ValueError, TypeError):
                    continue
        recent = {int(e.get("cycle")): e for e in ((getattr(data, "state", None) or {}).get("recent") or [])
                  if e.get("cycle") is not None}
        out = []
        for k in range(max(0, int(wc) - n + 1), int(wc) + 1):
            f = os.path.join(work, f"rollouts_c{k}.jsonl")
            if not os.path.exists(f):
                continue
            rws = []
            with open(f) as fh:
                for ln in fh:
                    try:
                        r = _json.loads(ln)
                    except ValueError:
                        continue
                    rws.append({"qid": r.get("qid"), "score": r.get("score"),
                                "ratio": r.get("ratio"), "text_inj": r.get("text_inj")})
            gs = _groups(rws)
            comp = {}
            side = {"text": {"problems": 0, "all_fail": 0, "mixed": 0, "all_pass": 0},
                    "bare": {"problems": 0, "all_fail": 0, "mixed": 0, "all_pass": 0}}
            ratios = []
            for q, g in gs.items():
                r = float(g[0].get("ratio") or 0)
                ratios.append(r)
                succ = sum(1 for x in g if float(x.get("score") or 0) > 0)
                kind = "all_fail" if succ == 0 else ("all_pass" if succ == len(g) else "mixed")
                b = comp.setdefault(_bucket(r), {"problems": 0, "all_fail": 0, "mixed": 0, "all_pass": 0})
                b["problems"] += 1; b[kind] += 1
                sd = side["text" if g[0].get("text_inj") else "bare"]
                sd["problems"] += 1; sd[kind] += 1
            ratios.sort()
            qtl = (lambda f_: round(ratios[min(len(ratios) - 1, int(f_ * len(ratios)))], 1)) if ratios else (lambda f_: None)
            pr = probes.get(k + 1) or {}   # bare_probe records are labelled cycle+1 (probe after training cycle k)
            dec = recent.get(k + 1) or {}
            out.append({"cycle": k, "steps": f"{k*10+1}-{(k+1)*10}", "problems": len(gs),
                        "ratio_in_force": {"mean": round(sum(ratios)/len(ratios), 1) if ratios else None,
                                           "q25": qtl(0.25), "median": qtl(0.5), "q75": qtl(0.75)},
                        "by_ratio_bucket": comp, "text_vs_bare": side,
                        "hint_free_probe": {"heldout_pass1": (pr.get("heldout") or {}).get("pass1"),
                                            "train_pass1": (pr.get("train") or {}).get("pass1"),
                                            "n": pr.get("n")} if pr else None,
                        "decision_after": {"ratio_sets": dec.get("ratio_sets"), "item_ops": dec.get("item_ops"),
                                           "p_ops": dec.get("p_ops")} if dec else None})
        return {"cycles": out, "note": "steps assume K=10 per cycle; bucket keys: r=0, 0<r<=25, 25<r<=50, 50<r<=90."}
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
            frac = succ / len(g) if g else 0.0
            if args.get("succ_min") is not None and frac < float(args["succ_min"]):
                continue
            if args.get("succ_max") is not None and frac > float(args["succ_max"]):
                continue
            if kind == want and rmin <= r <= rmax:
                hist = ((probs.get(qid) or {}).get("hist") or [])[-3:]
                out.append({"qid": qid, "r": r, "succ": succ, "n": len(g),
                            "prev_visits": [{"cycle": e.get("cycle"), "r": e.get("r"),
                                             "succ": e.get("succ"), "n": e.get("n")}
                                            for e in hist]})
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

SERVING SCHEDULE (matters for what your ops can do): each cycle trains on a FRESH
slice of the problem pool in a fixed rotation, so a given problem comes back only every
few cycles. Consequences: (a) ratio_ops on this window's problems take effect at those
problems' NEXT visit, not next cycle — next cycle's window is different problems at
whatever dose they already carry; (b) text notes and p apply to next cycle's prompts immediately; (c) the
readout for a dose change is get_stats.revisits (previous visit's outcome@dose -> this
visit's outcome) and get_problems' prev_visits — judge earlier ratio decisions there,
not by next cycle's all-fail count.

A problem's group is its sampled rollouts for one prompt; if all score the same, the
group yields no gradient. YOUR PRIMARY OBJECTIVE IS THE ALL-FAIL GROUP. Mixed groups
carry gradient and RL learns them at the CURRENT dose — but success at a dose is not
success without it. All-pass groups carry no gradient either. The one place RL cannot
move on its own is the all-fail group: zero successes, zero gradient, and it stays that
way unless the dose changes the sampling. Whether, in which direction, by how much and
how fast to move any bucket's dose is your call, made from the evidence below — not from
a fixed schedule. get_stats reports each
ratio bucket's all-fail/mixed/all-pass split, all-fail problems split by whether text
notes were in their prompt (bare = dose question; with text = content question), and
last window's all-fail problems' fate this window (escaped vs still all-fail) — what
unlocked escaped ones is your most direct evidence of the right dose. Use
get_traces(all_fail_batch=true) to sweep the zero-gradient set quickly, then
get_traces(qid) on the ones worth a full read.
HINT-FREE READOUTS (in the user message when available): pass@1 on fixed held-out
competition sets the policy never trains on, each reported twice — hint-free, and
again on the SAME problems with a 50% solution-prefix hint (the keys ending _r50).
The hint-free number is what the policy can do on its own; hinted outcomes from
training groups cannot show that. Read the PAIR, not either half: if the hinted
number climbs while the hint-free one stalls or falls, the policy is learning to
CONTINUE hints, not to solve. Weigh that against what revisits show about which dose
changes actually produced escapes; both readouts are evidence, neither is a rule.

INVESTIGATION: read-only tools over this cycle's rollouts, the ratio state and the
text scaffold. The user message states your EXACT budgets; every result carries
`_budget_calls_remaining`. Investigate, then commit to ONE final decision.

Return, as your FINAL message (no tool call), ONLY this JSON:
{"diagnosis": "<your reasoning>",
 "ratio_ops": [
   {"scope": "where", "where": {"outcome": "all_fail"|"all_pass"|"mixed" (optional),
     "r_min": <0..90>, "r_max": <0..90>, "succ_min": <0..1>, "succ_max": <0..1>
     (all filters optional, AND-ed; succ_* = this window's success fraction)},
    "delta": <-20..20>}  — or "set": <0..90> instead of delta |
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
- ratio: at most 6 where ops, 4 bucket ops and 16 qid ops; delta clamped to +-20;
  r in [0, 90]; ops apply IN ORDER and compose (a later op sees earlier ops' result);
  where/bucket ops only ever touch THIS WINDOW's problems;
- text: at most 3 add/update ops per cycle (deletes free); skill/plan <= 500 chars,
  example <= 1500; no duplicates; ONE global p in [0, 0.5];
- graduation bookkeeping is mechanical ONLY for state: a bare success graduates, a
  bare failure returns the problem to the active set at its current dose (r=0). NO
  mechanism raises or lowers doses on its own — every dose move, in either direction,
  is yours."""
