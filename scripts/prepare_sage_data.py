"""Build the SAGE-bench training set from open-r1/OpenR1-Math-220k (default split).

Mirrors SAGE's (arXiv 2602.03143) data prep: start from the ~94k default split, keep
rows whose DeepSeek-R1 generation is Math-Verify-correct (the dataset ships these
flags as `correctness_math_verify`), keep traces shorter than --max-trace-tokens
(8192) under the target model's tokenizer, then sample --n (15000, seed 0).

Output jsonl rows are {problem, answer, generation} — the format mathscaffold's
data.load_problems expects (solution = the part after </think>; rows whose answer
string does not appear there are dropped by the loader, so we pre-apply the same
check and report the count). Also writes the bare-probe qid sets (200 held-out +
200 in-train) for this pool next to the jsonl.

  python scripts/prepare_sage_data.py --model Qwen/Qwen3-4B-Instruct-2507 \
      --out data/sage15k/openr1_sage15k.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="tokenizer used for the trace-length filter")
    ap.add_argument("--max-trace-tokens", type=int, default=8192)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-n", type=int, default=200, help="bare-probe set size (per set)")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ds = load_dataset(a.dataset, split="train")
    print(f"{a.dataset}: {len(ds)} rows", flush=True)

    kept, seen = [], set()
    n_noverify = n_long = n_nocontain = 0
    for i, row in enumerate(ds):
        if i % 10000 == 0:
            print(f"  {i}/{len(ds)} scanned, {len(kept)} kept", flush=True)
        gens = row.get("generations") or []
        flags = row.get("correctness_math_verify") or []
        gen = next((g for g, ok in zip(gens, flags) if ok), None)
        if gen is None:
            n_noverify += 1
            continue
        if len(tok(gen, add_special_tokens=False)["input_ids"]) >= a.max_trace_tokens:
            n_long += 1
            continue
        answer = str(row.get("answer") or "")
        solution = gen.split("</think>")[-1]
        if not answer or answer not in solution:
            n_nocontain += 1
            continue
        key = " ".join(str(row["problem"]).split())
        if key in seen:
            continue
        seen.add(key)
        kept.append({"problem": row["problem"], "answer": answer, "generation": gen,
                     "_qid": "q" + hashlib.sha1(key.encode()).hexdigest()[:10]})
    print(f"kept {len(kept)} (dropped: {n_noverify} unverified, {n_long} >= {a.max_trace_tokens} tokens, "
          f"{n_nocontain} answer-not-in-solution)", flush=True)

    rng = random.Random(a.seed)
    if len(kept) > a.n:
        kept = rng.sample(kept, a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_qid"}, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} rows -> {a.out}")

    sets_path = os.path.splitext(a.out)[0] + ".bare_probe_sets.json"
    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_bare_probe_sets.py"),
                           "--jsonl", a.out, "--n", str(a.probe_n), "--seed", "0", "--out", sets_path])
    print(f"bare-probe sets -> {sets_path}  (export MS_BARE_SETS={sets_path})")


if __name__ == "__main__":
    main()
