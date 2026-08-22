"""Rebuild the teacher arm's ratio_state.json as it stood right after the prepare of
cycle --upto, WITHOUT calling the teacher: replays prepare_cycle.py's state transitions
from the per-cycle rollout logs (rollouts_c{N}.jsonl) and the recorded teacher decisions
(teacher_transcripts/c{N}.json["decision"]). Use it to roll an arm back to a healthy
cycle while discarding the decisions the teacher made on degenerate rollouts later.

  python scripts/replay_state.py --work runs/teacher_v3 --upto 5 \
      --out runs/teacher_v3/ratio_state.json            # after moving the old one aside

Also rewrites teacher_transcripts/history.json to the entries <= --upto (the teacher's
rolling memory must not carry the discarded cycles' diagnoses). The transition code is
mirrored line by line from prepare_cycle.py; keep the two in sync.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mathscaffold import controller as C  # noqa: E402
from mathscaffold import data as D  # noqa: E402
from mathscaffold import teacher as T  # noqa: E402
from mathscaffold import textscaffold as TS  # noqa: E402


def outcomes_from(rollout_log):
    agg = collections.defaultdict(lambda: [0, 0])
    if not rollout_log or not os.path.exists(rollout_log):
        return {}
    for line in open(rollout_log):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        e = agg[r["qid"]]
        e[0] += 1 if float(r.get("score") or 0) > 0 else 0
        e[1] += 1
    return {q: (s, n) for q, (s, n) in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--upto", type=int, required=True, help="last cycle whose prepare is replayed")
    ap.add_argument("--jsonl", default=os.environ.get("MS_DATA",
                    "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-25-0-4.jsonl,"
                    "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-50-0-4.jsonl"))
    ap.add_argument("--out", required=True, help="where to write the rebuilt state")
    ap.add_argument("--no-history", action="store_true", help="do not rewrite history.json")
    a = ap.parse_args()

    problems = D.load_problems(a.jsonl)
    tdir = os.path.join(a.work, "teacher_transcripts")
    tmp = a.out + ".replay_tmp.json"
    if os.path.exists(tmp):
        os.remove(tmp)
    recent = []
    for c in range(a.upto + 1):
        state = C.load_state(tmp, problems)          # fresh at c=0, then the saved one
        for h in state.get("problems", {}).values():
            if float(h.get("r") or 0) > C.R_MAX:
                h["r"] = C.R_MAX
        outcomes = outcomes_from(os.path.join(a.work, f"rollouts_c{c - 1}.jsonl") if c > 0 else None)
        probs = state["problems"]
        book = {q: (s_, n_) for q, (s_, n_) in outcomes.items()
                if probs.get(q, {}).get("r", 1) <= 0 or probs.get(q, {}).get("state") == "graduated"}
        state, notes = C.adaptive_update(state, book, c)
        probs = state["problems"]
        for q, (s_, n_) in outcomes.items():
            h = probs.get(q)
            if h is None or q in book:
                continue
            h["hist"] = (h.get("hist") or [])[-11:] + [{"cycle": c - 1, "r": h.get("r"), "succ": s_, "n": n_}]
        tpath = os.path.join(tdir, f"c{c}.json")
        decision = None
        if os.path.exists(tpath):
            try:
                decision = json.load(open(tpath)).get("decision")
            except (OSError, ValueError):
                decision = None
        if decision is None:
            state, notes2 = C.adaptive_update(state, outcomes, c)
            print(f"[replay] cycle {c}: no decision -> mechanical fallback ({len(notes2)} notes)")
        else:
            for qid, (succ, n) in outcomes.items():
                if qid in probs:
                    probs[qid]["_outcome"] = ("all_fail" if succ == 0 else ("all_pass" if succ == n else "mixed"))
            sets, item_ops, p_ops, note = T.normalize(decision, state)
            for qid, new_r in sets:
                probs[qid]["r"] = new_r
            state["text"], n1 = TS.apply_item_ops(state["text"], item_ops)
            state["text"], n2 = TS.apply_p_ops(state["text"], p_ops)
            recent.append({"cycle": c, "probe": None, "ratio_sets": len(sets), "item_ops": len(item_ops),
                           "p_ops": len(p_ops), "diagnosis": (decision.get("diagnosis") or "")[:240]})
            print(f"[replay] cycle {c}: {note}; {len(sets)} ratios set, p={state['text'].get('p')}")
        C.save_state(state, tmp)
    os.replace(tmp, a.out)
    rs = collections.Counter(round(float(h["r"])) for h in state["problems"].values())
    print(f"[replay] wrote {a.out}: r histogram {dict(sorted(rs.items()))}, text p {state['text'].get('p')}, "
          f"items {[i['id'] for i in state['text']['items'].get('general', [])]}")
    if not a.no_history:
        hp = os.path.join(tdir, "history.json")
        if os.path.exists(hp):
            os.replace(hp, hp + ".pre_replay")
        json.dump(recent[-12:], open(hp, "w"), ensure_ascii=False)
        print(f"[replay] history.json rewritten with {len(recent[-12:])} entries (old one kept as .pre_replay)")


if __name__ == "__main__":
    main()
