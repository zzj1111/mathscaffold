"""AIME24/25 + MATH500 probe against a served checkpoint (hint-free, always).
Eval protocol per QuestA/PieceHint: temp 0.7, top-p 0.95, n samples, mean pass@1
via Math-Verify. Usage: eval_probe.py --base-url http://host:port/v1 --set aime24
Datasets: HuggingFaceH4/aime_2024 etc. — downloaded on first use."""
import argparse, json

from concurrent.futures import ThreadPoolExecutor

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from mathscaffold import reward as R  # noqa: E402  (subprocess-guarded verifier)

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
    gt = "\\boxed{" + str(item[ak]) + "}"
    for ch in r.choices:
        # Verify through the TRAINING reward's subprocess pool (reward._verify): same
        # scoring path as training — including the last-boxed second chance — AND the same
        # wall-clock guard (MS_VERIFY_TIMEOUT, default 20s; an overrunning child is killed
        # and that sample scores 0).
        # In-process parse/verify has NO timeout, and sympy holds the GIL while it works,
        # so a single pathological expansion freezes every thread in this ThreadPoolExecutor
        # and the whole probe hangs. Seen live on B200: 5h50m wedged inside a factorial
        # _recursive, aime24 already scored but discarded, aime25/hmmt25 never ran.
        try:
            s, timed_out = R._verify(ch.message.content or "", gt)
        except Exception:
            s, timed_out = 0.0, False
        if timed_out:
            TIMEOUTS.append(idx)
        ok += 1 if s > 0 else 0
    return ok / a.n

ERRORS = []
TIMEOUTS = []   # samples whose verification hit the wall clock; scored 0, probe continues
PER_PROBLEM = {}
STDERR = {}
results = {}
# Each set used to get its own pool of 4-per-server, so a 3-set probe paid three tails
# of the longest generation — and each wave inside a set paid one as well. Submit every
# problem of every set together and let vLLM's continuous batching schedule the lot.
TASKS = []
for name_ in (a.sets.split(",") if a.sets else [a.set]):
    name, split, qk, ak = SRC[name_]
    ds = load_dataset(name, split=split)
    TASKS += [(name_, i, item, qk, ak) for i, item in enumerate(ds)]
_workers = int(_os.environ.get("MS_EVAL_WORKERS", "0")) or len(TASKS)
with ThreadPoolExecutor(max_workers=max(1, min(_workers, len(TASKS) or 1))) as ex:
    _all = list(ex.map(lambda t: one((t[1], t[2]), t[3], t[4]), TASKS))
_by_set = {}
for _t, _sc in zip(TASKS, _all):
    _by_set.setdefault(_t[0], []).append(_sc)
for name_, scores in _by_set.items():
    results[name_] = round(sum(scores) / len(scores), 4)
    PER_PROBLEM[name_] = [round(x, 4) for x in scores]
    # binomial standard error of the set mean under resampling with the same problems:
    # sqrt(sum p_i(1-p_i)/n) / N — the noise floor to read any two probes against
    se = (sum(x * (1 - x) for x in scores) / a.n) ** 0.5 / len(scores)
    STDERR[name_] = round(se, 4)
    print(json.dumps({"set": name_, "n_problems": len(scores), "mean_pass1": results[name_],
                      "stderr": STDERR[name_], "request_failures": len(ERRORS), "verify_timeouts": len(TIMEOUTS)}), flush=True)
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
