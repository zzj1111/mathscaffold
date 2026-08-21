"""Hint-free pass rate of a checkpoint on the TRAINING problems (diagnostic).

For every problem of a QuestA jsonl: prompt = the training prompt at ratio 0 (paper
template: problem + boxed instruction, no hint, no notes) through the model's chat
template; sampling = the TRAINING distribution (temperature 1.0, top_p 1.0, max
response = the training cap); n samples per problem; scored by the training reward
(mathscaffold.reward.compute_score). Per-problem rows go to <out>.jsonl (resumable);
a summary with the pass-count histogram goes to <out>.summary.json.

  python scripts/eval_bare_trainset.py --model-dir <hf dir> --gpus 0,1 \
      --jsonl /mnt/data1/zha00175/math_prep/questa_12k/OpenR1-50-0-4.jsonl \
      --n 8 --out /mnt/data1/zha00175/math_analysis/bare_base_openr1_50
"""
import argparse
import collections
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mathscaffold import data as D  # noqa: E402
from mathscaffold import reward as R  # noqa: E402


def start_servers(model_dir, gpus, port0, log_prefix):
    procs, urls = [], []
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    for i, g in enumerate(gpus):
        port = port0 + i
        e = dict(env, CUDA_VISIBLE_DEVICES=g)
        p = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
             "--served-model-name", "actor", "--tensor-parallel-size", "1",
             "--gpu-memory-utilization", "0.85", "--max-model-len", "32768",
             "--enable-prefix-caching", "--host", "127.0.0.1", "--port", str(port)],
            env=e, stdout=open(f"{log_prefix}.vllm_g{g}.log", "w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        procs.append(p)
        urls.append(f"http://127.0.0.1:{port}/v1")
    return procs, urls


def wait_healthy(clis, procs, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(p.poll() is not None for p in procs):
            raise RuntimeError("a vLLM server died; see .vllm_g*.log")
        try:
            for c in clis:
                c.models.list()
            return
        except Exception:
            time.sleep(10)
    raise RuntimeError("vLLM not healthy in time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gpus", default="", help="comma list; one vLLM server per GPU (omit with --base-url)")
    ap.add_argument("--base-url", default="", help="comma list of running vLLM servers; no servers are spawned")
    ap.add_argument("--jsonl", default="/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-50-0-4.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.0, help="hint ratio (solution-prefix %%); 0 = bare")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("MS_MAXRESP", "24000")))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--port0", type=int, default=8300)
    ap.add_argument("--limit", type=int, default=0, help="first N problems (file order)")
    ap.add_argument("--sample", type=int, default=0, help="random subset of N problems (fixed --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qids", default=None, help="json list of qids to restrict to")
    ap.add_argument("--save-text", action="store_true", help="keep full generations in the rows")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()

    problems = D.load_problems(a.jsonl)
    if a.limit:
        problems = problems[: a.limit]
    if a.sample:
        import random
        problems = random.Random(a.seed).sample(problems, a.sample)
    if a.qids:
        keep = set(json.load(open(a.qids)))
        problems = [p for p in problems if p["qid"] in keep]
    rows_path = a.out + ".jsonl"
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            try:
                done.add(json.loads(line)["qid"])
            except (ValueError, KeyError):
                continue
    todo = [p for p in problems if p["qid"] not in done]
    print(f"{len(problems)} problems, {len(done)} already done, {len(todo)} to run; n={a.n} "
          f"temp={a.temperature} max_tokens={a.max_tokens}", flush=True)

    gpus = [g for g in a.gpus.split(",") if g]
    # SIGTERM must run the finally: block below, or the vLLM servers (own sessions)
    # outlive us and hold the GPUs (seen live)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.base_url:
        procs, urls = [], [u.strip() for u in a.base_url.split(",") if u.strip()]
    else:
        if not gpus:
            sys.exit("need --gpus (spawn servers) or --base-url (use running servers)")
        procs, urls = start_servers(a.model_dir, gpus, a.port0, a.out)
    import openai
    clis = [openai.OpenAI(base_url=u, api_key="EMPTY", timeout=3600, max_retries=2) for u in urls]
    try:
        if procs:
            wait_healthy(clis, procs)
            print(f"{len(clis)} servers up on GPUs {gpus}", flush=True)
        else:
            for c in clis:
                c.models.list()
            print(f"using {len(clis)} running servers", flush=True)
        lock = __import__("threading").Lock()
        t0 = time.time()
        counter = {"done": 0}

        def one(ip):
            i, p = ip
            prompt = D.hint_prompt(p["problem"], p["solution"], a.ratio, style="paper")
            cli = clis[i % len(clis)]
            r = None
            for attempt in range(4):
                try:
                    r = cli.chat.completions.create(
                        model="actor", temperature=a.temperature, top_p=1.0,
                        max_tokens=a.max_tokens, n=a.n,
                        messages=[{"role": "user", "content": prompt}])
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] {p['qid']} attempt {attempt + 1}: {e!r}"[:200], flush=True)
                    time.sleep(15)
            if r is None:
                return None
            gt = "\\boxed{" + p["answer"] + "}"
            scores, lens, fins = [], [], []
            for ch in r.choices:
                txt = ch.message.content or ""
                s = R.compute_score("questa_math", txt, gt, {"qid": p["qid"], "ratio": a.ratio, "text_inj": False})
                s = float(s if not isinstance(s, dict) else s.get("score", 0))
                scores.append(1 if s > 0 else 0)
                lens.append(len(txt))
                fins.append(ch.finish_reason)
            row = {"qid": p["qid"], "n": a.n, "pass": sum(scores), "scores": scores,
                   "finish": fins, "char_len": lens, "answer": p["answer"],
                   "problem": p["problem"][:300]}
            if a.save_text:
                row["texts"] = [ch.message.content or "" for ch in r.choices]
                row["problem_full"] = p["problem"]
                row["solution"] = p["solution"]
            with lock:
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counter["done"] += 1
                if counter["done"] % 50 == 0:
                    el = time.time() - t0
                    print(f"[{counter['done']}/{len(todo)}] {el / 60:.0f} min elapsed, "
                          f"eta {el / counter['done'] * (len(todo) - counter['done']) / 60:.0f} min", flush=True)
            return row

        with ThreadPoolExecutor(max_workers=8 * len(clis)) as ex:
            list(ex.map(one, enumerate(todo)))
    finally:
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), 15)
            except Exception:
                pass

    rows = [json.loads(l) for l in open(rows_path)]
    hist = collections.Counter(r["pass"] for r in rows)
    n = rows[0]["n"] if rows else a.n
    summary = {
        "ratio": a.ratio, "model_dir": a.model_dir, "jsonl": a.jsonl, "n": n, "n_problems": len(rows),
        "mean_pass1": round(sum(r["pass"] for r in rows) / (n * len(rows)), 4),
        "solved_any": round(sum(r["pass"] > 0 for r in rows) / len(rows), 4),
        "solved_all": round(sum(r["pass"] == n for r in rows) / len(rows), 4),
        "pass_hist": {str(k): hist[k] for k in range(n + 1)},
        # SE of the set mean treating per-problem pass rates as iid draws (between- plus
        # within-problem spread; conservative for a fixed set, and well-defined at n=1)
        "stderr": round((statistics.pvariance([r["pass"] / n for r in rows]) / len(rows)) ** 0.5, 4) if len(rows) > 1 else 0.0,
        "truncated_frac": round(sum(f == "length" for r in rows for f in r["finish"]) / (n * len(rows)), 4),
        "mean_chars": round(sum(sum(r["char_len"]) for r in rows) / (n * len(rows))),
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
