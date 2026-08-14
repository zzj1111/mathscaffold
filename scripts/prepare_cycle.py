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
                "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-25-0-4.jsonl"))
ap.add_argument("--state", default="ratio_state.json")
ap.add_argument("--rollout-log", default=None)
ap.add_argument("--arm", choices=["adaptive", "static", "teacher"], required=True)
ap.add_argument("--cycle", type=int, required=True)
ap.add_argument("--switch-cycle", type=int, default=10)
ap.add_argument("--served", type=int, default=2048)
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
    book = {q: (s_, n_) for q, (s_, n_) in outcomes.items()
            if state.get(q, {}).get("r", 1) <= 0 or state.get(q, {}).get("state") == "graduated"}
    state, notes = C.adaptive_update(state, book, a.cycle)
    sets, note, _ = T.decide(a.rollout_log or "", state, outcomes, a.cycle,
                             transcript_dir=os.path.join(os.path.dirname(a.state) or ".",
                                                         "teacher_transcripts"))
    if sets is None:
        state, notes2 = C.adaptive_update(state, outcomes, a.cycle)
        notes += [note] + notes2
    else:
        for qid, new_r in sets:
            state[qid]["r"] = new_r
        notes += [note, f"teacher set {len(sets)} ratios"]
else:
    state, notes = C.static_update(state, outcomes, a.cycle, a.switch_cycle)
for n in notes[:20]:
    print("[ctrl]", n)
print(f"[ctrl] {len(notes)} changes; outcomes for {len(outcomes)} problems")

qids = [p["qid"] for p in problems]
random.Random(20260814).shuffle(qids)
lo = (a.cycle * a.served) % len(qids)
served = set((qids + qids)[lo:lo + a.served])
rows = D.build_rows(problems, state, served)
D.write_parquet(rows, a.out)
C.save_state(state, a.state)
print(f"[prepare] cycle {a.cycle}: {len(rows)} rows -> {a.out}")
