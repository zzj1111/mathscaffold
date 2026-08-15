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
ap.add_argument("--switch-cycle", type=int, default=10)
# served slice MUST equal steps_per_cycle x train_batch (default 8 x 128): the
# trainer consumes exactly that many prompts per cycle, and every served problem
# must produce outcomes for the controller
ap.add_argument("--served", type=int,
                default=int(os.environ.get("MS_K", "10")) * int(os.environ.get("MS_BS", "128")))
ap.add_argument("--out", required=True)
a = ap.parse_args()

problems = D.load_problems(a.jsonl)
state = C.load_state(a.state, problems)

outcomes = {}
if a.rollout_log and os.path.exists(a.rollout_log):
    agg = collections.defaultdict(lambda: [0, 0])
    for line in open(a.rollout_log):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        agg[r["qid"]][0] += 1 if float(r.get("score") or 0) > 0 else 0
        agg[r["qid"]][1] += 1
    outcomes = {q: (s, n) for q, (s, n) in agg.items()}

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
    probe_line = ""
    pf = os.path.join(os.path.dirname(a.state) or ".", "probe.json")
    if os.path.exists(pf):
        try:
            pr = json.load(open(pf))
            probe_line = ("Latest hint-free probes: "
                          + ", ".join(f"{k} pass@1 {v}" for k, v in pr.items()
                                      if k != "cycle")
                          + f" (measured at cycle {pr.get('cycle', '?')}). ")
        except (OSError, ValueError):
            pass
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
                                            "teacher_transcripts", f"c{a.cycle}.json"))
except Exception as _e:
    print(f"[wandb] skipped: {_e}")

qids = [p["qid"] for p in problems]
random.Random(20260814).shuffle(qids)
lo = (a.cycle * a.served) % len(qids)
served = set((qids + qids)[lo:lo + a.served])
rows = D.build_rows(problems, state, served, cycle=a.cycle)
D.write_parquet(rows, a.out)
C.save_state(state, a.state)
print(f"[prepare] cycle {a.cycle}: {len(rows)} rows -> {a.out}")
