"""AIME24/25 + MATH500 probe against a served checkpoint (hint-free, always).
Eval protocol per QuestA/PieceHint: temp 0.7, top-p 0.95, n samples, mean pass@1
via Math-Verify. Usage: eval_probe.py --base-url http://host:port/v1 --set aime24
Datasets: HuggingFaceH4/aime_2024 etc. — downloaded on first use."""
import argparse, json, re

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
ap.add_argument("--dump", default=None, help="directory: write <set>[_rR].jsonl with every sample's "
                "response, finish_reason, last boxed answer and verdict (failure analysis)")
ap.add_argument("--ratios", default="0",
                help="comma list of hint ratios, e.g. '0,50'. A ratio >0 prepends the first r%% of the\n                      set's reference solution as a '## Hint.' block, the same construction training\n                      uses. Sets with no reference solution silently run ratio 0 only.")
a = ap.parse_args()

from datasets import load_dataset
# (name, split, question_key, answer_key, solution_key or None). hmmt25 reads FlagEval's
# copy rather than MathArena's: verified identical (30/30 answers and 30/30 problem texts
# match) but it also carries a reference solution, which is what a hinted ratio needs.
# aime25 has no reference solution in any cached copy, so it can only be probed bare.
SRC = {"aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer", "solution"),
       "aime25": ("MathArena/aime_2025", "train", "problem", "answer", None),
       "hmmt25": ("FlagEval/HMMT_2025", "train", "question", "answer", "solution"),
       "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer", "solution"),
       "amc23": ("math-ai/amc23", "test", "question", "answer", None)}

def clean_prefix(sol, ratio):
    """The first ratio% of the reference solution, by characters, as training builds it —
    but cut to the FIRST solution only. These pages concatenate several alternative
    solutions, each ending in \\boxed{answer} plus an author signature, so a flat ratio%
    of the whole page lands AFTER a conclusion for most problems and hands over the
    answer: measured on aime24, the answer appears verbatim in 30/30 raw solutions.
    Cutting at the first conclusion marker first drops that to 1/30 at both 25%% and 50%%."""
    m = re.search(r"\\boxed\{|\\framebox\{|~[A-Za-z0-9_]{3,}", sol)
    core = sol[: m.start()] if m else sol
    return core[: int(len(core) * ratio / 100.0)]

import openai
clis = [openai.OpenAI(base_url=u.strip(), api_key="EMPTY", timeout=7200)
        for u in a.base_url.split(",") if u.strip()]
from math_verify import parse, verify

def one(idx_item, qk, ak, sk=None, ratio=0, dump_key=None):
    # one API call with n completions (vLLM samples them in a single batch) on the
    # client for this problem's slot — spreads load across the per-GPU servers
    idx, item = idx_item
    cli = clis[idx % len(clis)]
    q = item[qk]
    if ratio and sk:
        pre = clean_prefix(str(item[sk] or ""), ratio)
        if len(pre) >= 10:
            q = q + "\n\n## Hint." + pre
    content = (("Solve the following math problem. Make sure to put the answer "
                "(and only answer) inside \\boxed{}.\n\n" + q)
               if a.instruction == "official" else
               (q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."))
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
    recs = []
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
        if a.dump:
            txt = ch.message.content or ""
            boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", txt)
            recs.append({"correct": bool(s > 0), "timed_out": bool(timed_out),
                         "finish_reason": getattr(ch, "finish_reason", None),
                         "chars": len(txt), "last_boxed": boxed[-1] if boxed else None,
                         "n_boxed": len(boxed), "response": txt})
    if a.dump:
        DUMP.setdefault(dump_key or ("r%d" % ratio), []).append({"idx": idx, "gold": str(item[ak]), "ratio": ratio,
                                               "problem": q if not (ratio and sk) else item[qk],
                                               "samples": recs})
    return ok / a.n

ERRORS = []
DUMP = {}
TIMEOUTS = []   # samples whose verification hit the wall clock; scored 0, probe continues
PER_PROBLEM = {}
STDERR = {}
results = {}
# Each set used to get its own pool of 4-per-server, so a 3-set probe paid three tails
# of the longest generation — and each wave inside a set paid one as well. Submit every
# problem of every set together and let vLLM's continuous batching schedule the lot.
# A set is probed once per requested ratio. The bare run keeps the plain set name, so
# probe.json's historical keys ("aime24") still mean the hint-free number; a hinted run
# is reported alongside as "aime24_r50". The DIFFERENCE is the point: hinted accuracy
# climbing while bare accuracy stalls is the policy learning to continue hints rather
# than to solve, which no single number can show.
RATIOS = [int(x) for x in str(a.ratios).split(",") if x.strip() != ""]
LEAK = {}
TASKS = []
for name_ in (a.sets.split(",") if a.sets else [a.set]):
    name, split, qk, ak, sk = SRC[name_]
    ds = load_dataset(name, split=split)
    for r in RATIOS:
        if r and not sk:
            print(f"[probe] {name_}: no reference solution in this source, skipping ratio {r}", flush=True)
            continue
        key = name_ if r == 0 else f"{name_}_r{r}"
        if r:
            # report, do not assume: if the answer survives into the hint the number
            # measures copying, not hint use
            n_leak = sum(1 for it in ds
                         if str(it[ak]).strip() and str(it[ak]).strip() in clean_prefix(str(it[sk] or ""), r))
            LEAK[key] = n_leak
            if n_leak:
                print(f"[probe] {key}: answer appears in the hint for {n_leak}/{len(ds)} problems", flush=True)
        TASKS += [(key, i, item, qk, ak, sk, r) for i, item in enumerate(ds)]
_workers = int(_os.environ.get("MS_EVAL_WORKERS", "0")) or len(TASKS)
with ThreadPoolExecutor(max_workers=max(1, min(_workers, len(TASKS) or 1))) as ex:
    def _run(t):
        return one((t[1], t[2]), t[3], t[4], t[5], t[6], dump_key=t[0])
    _all = list(ex.map(_run, TASKS))
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
if a.dump:
    _os.makedirs(a.dump, exist_ok=True)
    for _k, _v in DUMP.items():
        with open(_os.path.join(a.dump, f"{_k}.jsonl"), "w") as f:
            for _rec in sorted(_v, key=lambda x: x["idx"]):
                f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
    print("dumped", {k: len(v) for k, v in DUMP.items()}, "->", a.dump)
if a.out:
    payload = dict(results)
    payload["stderr"] = STDERR
    payload["per_problem"] = PER_PROBLEM
    payload["n"] = a.n
    payload["request_failures"] = len(ERRORS)
    if LEAK:
        payload["answer_in_hint"] = LEAK
    if a.cycle is not None:
        payload["cycle"] = a.cycle
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print("wrote", a.out)
