"""AIME24/25 + MATH500 probe against a served checkpoint (hint-free, always).
Eval protocol per QuestA/PieceHint: temp 0.7, top-p 0.95, n samples, mean pass@1
via Math-Verify. Usage: eval_probe.py --base-url http://host:port/v1 --set aime24
Datasets: HuggingFaceH4/aime_2024 etc. — downloaded on first use."""
import argparse, json

from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--set", default="aime24",
                choices=["aime24", "aime25", "hmmt25", "math500"])
ap.add_argument("--sets", default=None,
                help="comma list to run several in one server session, e.g. aime24,aime25,hmmt25")
ap.add_argument("--out", default=None, help="write {set: mean_pass1} JSON here")
ap.add_argument("--cycle", type=int, default=None)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=30720)
a = ap.parse_args()

from datasets import load_dataset
SRC = {"aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
       "aime25": ("MathArena/aime_2025", "train", "problem", "answer"),
       "hmmt25": ("MathArena/hmmt_feb_2025", "train", "problem", "answer"),
       "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer")}

import openai
cli = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=7200)
from math_verify import parse, verify

def one(item, qk, ak):
    ok = 0
    for _ in range(a.n):
        r = cli.chat.completions.create(
            model="actor", temperature=0.7, top_p=0.95, max_tokens=a.max_tokens,
            messages=[{"role": "user", "content": item[qk] +
                       "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}])
        try:
            ok += 1 if verify(parse("\\boxed{" + str(item[ak]) + "}", parsing_timeout=None),
                              parse((r.choices[0].message.content or "")[-3000:],
                                    parsing_timeout=None),
                              timeout_seconds=None) else 0
        except Exception:
            pass
    return ok / a.n

results = {}
for name_ in (a.sets.split(",") if a.sets else [a.set]):
    name, split, qk, ak = SRC[name_]
    ds = load_dataset(name, split=split)
    with ThreadPoolExecutor(max_workers=8) as ex:
        scores = list(ex.map(lambda it: one(it, qk, ak), ds))
    results[name_] = round(sum(scores) / len(scores), 4)
    print(json.dumps({"set": name_, "n_problems": len(scores), "mean_pass1": results[name_]}), flush=True)
if a.out:
    payload = dict(results)
    if a.cycle is not None:
        payload["cycle"] = a.cycle
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print("wrote", a.out)
