"""Fix the two 200-problem sets for the per-cycle hint-free (bare) probe.

  heldout: sampled from the full pool and EXCLUDED from training (prepare_cycle.py
           drops these qids from the serving rotation) — measures generalization
           within the training distribution;
  train:   a disjoint sample that stays in training — the heldout/train gap is the
           memorization component.

Both are qid lists (content hashes of the problem text, stable across machines), so
the same file serves every machine. Written once and committed:
  python scripts/make_bare_probe_sets.py --n 200 --seed 0
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mathscaffold import data as D  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mathscaffold",
                           "bare_probe_sets.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.environ.get("MS_DATA",
                    "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-25-0-4.jsonl,"
                    "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-50-0-4.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    problems = D.load_problems(a.jsonl)
    qids = sorted(p["qid"] for p in problems)
    rng = random.Random(a.seed)
    pick = rng.sample(qids, 2 * a.n)
    sets = {"heldout": sorted(pick[: a.n]), "train": sorted(pick[a.n:]),
            "n": a.n, "seed": a.seed, "pool_size": len(qids)}
    with open(a.out, "w") as f:
        json.dump(sets, f, indent=1)
    print(f"pool {len(qids)} -> heldout {len(sets['heldout'])}, train {len(sets['train'])} -> {a.out}")


if __name__ == "__main__":
    main()
