"""AIME24/25 + MATH500 probe against a served checkpoint (hint-free, always).
Eval protocol per QuestA/PieceHint: temp 0.7, top-p 0.95, n samples, mean pass@1
via Math-Verify. Usage: eval_probe.py --base-url http://host:port/v1 --set aime24
Datasets: HuggingFaceH4/aime_2024 etc. — downloaded on first use."""
import argparse, json

from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--set", default="aime24", choices=["aime24", "aime25", "math500"])
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=30720)
a = ap.parse_args()

from datasets import load_dataset
SRC = {"aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
       "aime25": ("math-ai/aime25", "test", "problem", "answer"),
       "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer")}
name, split, qk, ak = SRC[a.set]
ds = load_dataset(name, split=split)

import openai
cli = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=7200)
from math_verify import parse, verify

def one(item):
    ok = 0
    for _ in range(a.n):
        r = cli.chat.completions.create(
            model="actor", temperature=0.7, top_p=0.95, max_tokens=a.max_tokens,
            messages=[{"role": "user", "content": item[qk]}])
        try:
            ok += 1 if verify(parse("\\boxed{" + str(item[ak]) + "}"),
                              parse(r.choices[0].message.content or "")) else 0
        except Exception:
            pass
    return ok / a.n

with ThreadPoolExecutor(max_workers=8) as ex:
    scores = list(ex.map(one, ds))
print(json.dumps({"set": a.set, "n_problems": len(scores),
                  "mean_pass1": sum(scores) / len(scores)}))
