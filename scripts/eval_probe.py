"""AIME24/25 + MATH500 probe against a served checkpoint (hint-free, always).
Eval protocol per QuestA/PieceHint: temp 0.7, top-p 0.95, n samples, mean pass@1
via Math-Verify. Usage: eval_probe.py --base-url http://host:port/v1 --set aime24
Datasets: HuggingFaceH4/aime_2024 etc. — downloaded on first use."""
import argparse, json

from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True,
                help="one URL, or comma-separated URLs (one vLLM per GPU) — problems are spread round-robin")
ap.add_argument("--set", default="aime24",
                choices=["aime24", "aime25", "hmmt25", "math500", "amc23"])
ap.add_argument("--sets", default=None,
                help="comma list to run several in one server session, e.g. aime24,aime25,hmmt25")
ap.add_argument("--out", default=None, help="write {set: mean_pass1} JSON here")
ap.add_argument("--cycle", type=int, default=None)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--instruction", choices=["probe", "official"], default="probe",
                help="probe = 'Please reason step by step...' suffix (our training-arm protocol); "
                     "official = the OpenMath-Nemotron model-card prefix (NeMo-Skills wording)")
ap.add_argument("--max-tokens", type=int, default=30720)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.95)
a = ap.parse_args()

from datasets import load_dataset
SRC = {"aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
       "aime25": ("MathArena/aime_2025", "train", "problem", "answer"),
       "hmmt25": ("MathArena/hmmt_feb_2025", "train", "problem", "answer"),
       "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
       "amc23": ("math-ai/amc23", "test", "question", "answer")}

import openai
clis = [openai.OpenAI(base_url=u.strip(), api_key="EMPTY", timeout=7200)
        for u in a.base_url.split(",") if u.strip()]
from math_verify import parse, verify

def one(idx_item, qk, ak):
    # one API call with n completions (vLLM samples them in a single batch) on the
    # client for this problem's slot — spreads load across the per-GPU servers
    idx, item = idx_item
    cli = clis[idx % len(clis)]
    content = (("Solve the following math problem. Make sure to put the answer "
                "(and only answer) inside \\boxed{}.\n\n" + item[qk])
               if a.instruction == "official" else
               (item[qk] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."))
    # a transient server error on one problem must not abort the whole probe: retry,
    # then score the problem 0 and count it (reported at the end; a nonzero count
    # means the numbers are a lower bound, not a clean measurement)
    r = None
    for attempt in range(4):
        try:
            r = cli.chat.completions.create(
                model="actor", temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, n=a.n,
                messages=[{"role": "user", "content": content}])
            break
        except Exception as e:          # noqa: BLE001
            print(f"[probe] request error on problem {idx} (attempt {attempt + 1}): {e!r}"[:300], flush=True)
            time.sleep(15)
    if r is None:
        ERRORS.append(idx)
        return 0.0
    ok = 0
    gold = parse("\\boxed{" + str(item[ak]) + "}", parsing_timeout=None)
    for ch in r.choices:
        txt = ch.message.content or ""
        hit = False
        try:
            hit = bool(verify(gold, parse(txt[-3000:], parsing_timeout=None),
                              timeout_seconds=None))
        except Exception:
            pass
        # same second chance as the training reward (mathscaffold/reward.py): a response
        # that boxes its answer and then keeps writing past the 3000-char tail would
        # otherwise be scored wrong — a length-dependent false negative (0% below 20K
        # chars, 0.63% above 40K on real rollouts). Keep probe and reward identical.
        if not hit and len(txt) > 3000:
            i = txt.rfind("\\boxed{")
            if 0 <= i < len(txt) - 3000:
                try:
                    hit = bool(verify(gold, parse(txt[max(0, i - 200): i + 3000],
                                                  parsing_timeout=None), timeout_seconds=None))
                except Exception:
                    pass
        ok += 1 if hit else 0
    return ok / a.n

ERRORS = []
PER_PROBLEM = {}
STDERR = {}
results = {}
for name_ in (a.sets.split(",") if a.sets else [a.set]):
    name, split, qk, ak = SRC[name_]
    ds = load_dataset(name, split=split)
    with ThreadPoolExecutor(max_workers=4 * len(clis)) as ex:
        scores = list(ex.map(lambda it: one(it, qk, ak), enumerate(ds)))
    results[name_] = round(sum(scores) / len(scores), 4)
    PER_PROBLEM[name_] = [round(x, 4) for x in scores]
    # binomial standard error of the set mean under resampling with the same problems:
    # sqrt(sum p_i(1-p_i)/n) / N — the noise floor to read any two probes against
    se = (sum(x * (1 - x) for x in scores) / a.n) ** 0.5 / len(scores)
    STDERR[name_] = round(se, 4)
    print(json.dumps({"set": name_, "n_problems": len(scores), "mean_pass1": results[name_],
                      "stderr": STDERR[name_], "request_failures": len(ERRORS)}), flush=True)
if a.out:
    payload = dict(results)
    payload["stderr"] = STDERR
    payload["per_problem"] = PER_PROBLEM
    payload["n"] = a.n
    payload["request_failures"] = len(ERRORS)
    if a.cycle is not None:
        payload["cycle"] = a.cycle
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print("wrote", a.out)
