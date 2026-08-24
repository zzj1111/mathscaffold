"""Regenerate the next cycle's train parquet from the ratio state + this cycle's
recorder rows. Run at every cycle boundary (the arm loop calls this).

Usage: prepare_cycle.py --arm adaptive|static --cycle N --out train.parquet
       [--switch-cycle 10] [--served 2048]
Serving: a rotating slice of `served` problems per cycle (deterministic epoch
order), so the controller always has fresh outcomes for exactly the served set.
"""
import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mathscaffold import controller as C, data as D

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default=os.environ.get("MS_DATA",
                "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-25-0-4.jsonl,"
                "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-50-0-4.jsonl"))
ap.add_argument("--state", default="ratio_state.json")
ap.add_argument("--rollout-log", default=None)
ap.add_argument("--arm", choices=["adaptive", "static", "teacher"], required=True)
ap.add_argument("--cycle", type=int, required=True)
# static arm (QuestA control): global R0 (50) until this cycle, then 25 — QuestA's
# Partial_50 -> Partial_50_25 two-stage schedule; set MS_SWITCH_CYCLE to half the run
ap.add_argument("--switch-cycle", type=int, default=int(os.environ.get("MS_SWITCH_CYCLE", "10")))
# served slice MUST equal steps_per_cycle x train_batch (default 8 x 128): the
# trainer consumes exactly that many prompts per cycle, and every served problem
# must produce outcomes for the controller
ap.add_argument("--served", type=int,
                default=int(os.environ.get("MS_K", "10")) * int(os.environ.get("MS_BS", "128")))
ap.add_argument("--out", required=True)
a = ap.parse_args()

problems = D.load_problems(a.jsonl)
state = C.load_state(a.state, problems)
# hard dose cap (MS_R_MAX, default 50): also folds down any state carried from a
# phase with a higher cap, so the cap is authoritative every cycle
_capped = 0
for _h in state.get("problems", {}).values():
    if float(_h.get("r") or 0) > C.R_MAX:
        _h["r"] = C.R_MAX
        _capped += 1
if _capped:
    print(f"[ctrl] dose cap {C.R_MAX:g}: folded {_capped} problems down")

outcomes = {}
inj_info = None
if a.rollout_log and os.path.exists(a.rollout_log):
    agg = collections.defaultdict(lambda: [0, 0, False])
    rows_inj = rows_tot = 0
    for line in open(a.rollout_log):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        e = agg[r["qid"]]
        e[0] += 1 if float(r.get("score") or 0) > 0 else 0
        e[1] += 1
        e[2] = e[2] or bool(r.get("text_inj"))
        rows_tot += 1
        rows_inj += 1 if r.get("text_inj") else 0
    outcomes = {q: (s, n) for q, (s, n, _) in agg.items()}
    # injected-vs-bare group composition: the Teacher's content-vs-dose evidence,
    # surfaced per cycle in wandb so scaffold usage is visible without reading logs
    comp = {True: {"all_fail": 0, "mixed": 0, "all_pass": 0},
            False: {"all_fail": 0, "mixed": 0, "all_pass": 0}}
    for q, (sc, n, inj) in agg.items():
        comp[inj]["all_fail" if sc == 0 else ("all_pass" if sc == n else "mixed")] += 1
    inj_info = {"rows_injected": rows_inj, "rows_total": rows_tot,
                "text": comp[True], "bare": comp[False]}

if a.arm == "adaptive":
    state, notes = C.adaptive_update(state, outcomes, a.cycle)
elif a.arm == "teacher":
    # mechanical bookkeeping first (graduation / relapse / bare-probe outcomes are
    # not the Teacher's to decide), then the investigative Teacher steers ratios;
    # unreachable/malformed degrades to the adaptive rule for this cycle.
    from mathscaffold import teacher as T
    from mathscaffold import textscaffold as TS
    probs = state["problems"]
    book = {q: (s_, n_) for q, (s_, n_) in outcomes.items()
            if probs.get(q, {}).get("r", 1) <= 0 or probs.get(q, {}).get("state") == "graduated"}
    state, notes = C.adaptive_update(state, book, a.cycle)
    # visit history for EVERY served problem (adaptive_update only books bare/graduated
    # ones): {cycle, r in effect, succ, n}. Problems rotate (a slice of the pool per
    # cycle, each problem back every ~pool/served cycles), so this is the only way the
    # Teacher can later see "all-fail at r=50 last visit -> mixed at r=70 now" — the
    # direct evidence for whether a dose change worked.
    for q, (s_, n_) in outcomes.items():
        h = probs.get(q)
        if h is None or q in book:
            continue
        h["hist"] = (h.get("hist") or [])[-11:] + [
            {"cycle": a.cycle - 1, "r": h.get("r"), "succ": s_, "n": n_}]
    probe_line = ""
    # per-cycle hint-free readouts (bare_probe.jsonl, one record per cycle): the
    # hint-dependence signal — hinted success rising while these stall means the policy
    # is learning to continue hints, not to solve
    bp = os.path.join(os.path.dirname(a.state) or ".", "bare_probe.jsonl")
    if os.path.exists(bp):
        try:
            recs = [json.loads(l) for l in open(bp) if l.strip()][-6:]
            if recs:
                last = recs[-1]
                def _fmt(k):
                    r_ = last.get(k) or {}
                    return f"{r_.get('pass1', float('nan')):.3f} (±{r_.get('stderr', 0):.3f})"
                trend = ", ".join(f"c{r_['cycle']}:{(r_.get('heldout') or {}).get('pass1', float('nan')):.2f}"
                                  for r_ in recs)
                probe_line += ("Hint-free pass@1 on 200 HELD-OUT training-distribution problems "
                               f"(never trained on, n={last.get('n', '?')}): {_fmt('heldout')}; on 200 "
                               f"IN-TRAINING problems: {_fmt('train')}; held-out trend by cycle: {trend}. ")
        except (OSError, ValueError) as _e:
            print(f"[prepare] bare_probe.jsonl unreadable: {_e}")
    pf = os.path.join(os.path.dirname(a.state) or ".", "probe.json")
    if os.path.exists(pf):
        try:
            pr = json.load(open(pf))
            # set scores only — probe.json also carries stderr / per_problem / n /
            # request_failures bookkeeping that must not be pasted into the prompt
            sets = {k: v for k, v in pr.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and k not in ("cycle", "n", "request_failures")}
            se = pr.get("stderr") or {}
            probe_line = ("Latest hint-free probes: "
                          + ", ".join(f"{k} pass@1 {v}" + (f" (±{se[k]})" if k in se else "")
                                      for k, v in sets.items())
                          + f" (measured at cycle {pr.get('cycle', '?')}). ")
        except (OSError, ValueError):
            pass
    if probe_line:
        print(f"[prepare] teacher readouts: {probe_line}")
    result, note, _ = T.decide(a.rollout_log or "", state, outcomes, a.cycle,
                               probe_line=probe_line, problems=problems,
                               transcript_dir=os.path.join(os.path.dirname(a.state) or ".",
                                                           "teacher_transcripts"))
    if result is None:
        state, notes2 = C.adaptive_update(state, outcomes, a.cycle)
        notes += [note] + notes2
    else:
        sets, item_ops, p_ops, tnote = result
        for qid, new_r in sets:
            probs[qid]["r"] = new_r
        state["text"], n1 = TS.apply_item_ops(state["text"], item_ops)
        state["text"], n2 = TS.apply_p_ops(state["text"], p_ops)
        notes += [tnote, f"teacher set {len(sets)} ratios"] + n1 + n2
else:
    state, notes = C.static_update(state, outcomes, a.cycle, a.switch_cycle)
for n in notes[:20]:
    print("[ctrl]", n)
print(f"[ctrl] {len(notes)} changes; outcomes for {len(outcomes)} problems")

# publish this cycle's controller/teacher state (no-op unless MS_WANDB=1)
try:
    from mathscaffold import wb
    wb.publish(os.path.dirname(a.state) or ".", a.arm, a.cycle, state, outcomes, notes,
               transcript_path=os.path.join(os.path.dirname(a.state) or ".",
                                            "teacher_transcripts", f"c{a.cycle}.json"),
               inj_info=inj_info)
except Exception as _e:
    print(f"[wandb] skipped: {_e}")

# the per-cycle bare probe's held-out set never enters training (MS_BARE_PROBE=0 keeps
# the old full-pool rotation). Note: the rotation is a shuffle of the served pool, so
# switching this on mid-run reshuffles it once.
heldout = set()
if os.environ.get("MS_BARE_PROBE", "1") != "0":
    _sets = os.environ.get("MS_BARE_SETS") or os.path.join(os.path.dirname(__file__), "..", "mathscaffold", "bare_probe_sets.json")
    try:
        heldout = set(json.load(open(_sets))["heldout"])
    except (OSError, ValueError, KeyError) as _e:
        print(f"[prepare] WARNING: no held-out set ({_e}); serving the full pool")
qids = [p["qid"] for p in problems if p["qid"] not in heldout]
random.Random(20260814).shuffle(qids)
lo = (a.cycle * a.served) % len(qids)
served = set((qids + qids)[lo:lo + a.served])
rows = D.build_rows(problems, state, served, cycle=a.cycle)
D.write_parquet(rows, a.out)
C.save_state(state, a.state)
# per-cycle snapshot of the state that produced this cycle's parquet: rolling an arm
# back to cycle N is then `cp ratio_state_cN.json ratio_state.json` (no replay needed)
try:
    import shutil
    shutil.copyfile(a.state, os.path.join(os.path.dirname(a.state) or ".", f"ratio_state_c{a.cycle}.json"))
except OSError as _e:
    print(f"[prepare] snapshot skipped: {_e}")
print(f"[prepare] cycle {a.cycle}: {len(rows)} rows -> {a.out}")
