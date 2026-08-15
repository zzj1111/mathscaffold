"""One-off: tag each problem with a topic (algebra/geometry/number_theory/
combinatorics/other) — the category scopes for text-form scaffold. Batched
(20/call), resumable (skips qids already in the output file)."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mathscaffold import data as D
from mathscaffold.teacher import _client, MODEL

OUT = sys.argv[1] if len(sys.argv) > 1 else "topics.json"
problems = D.load_problems(os.environ.get("MS_DATA",
    "/mnt/data1/zha00175/math_prep/questa_12k/OpenR1-25-0-4.jsonl"))
done = {}
if os.path.exists(OUT):
    done = json.load(open(OUT))
todo = [p for p in problems if p["qid"] not in done]
print(f"{len(problems)} problems, {len(done)} tagged, {len(todo)} to go", flush=True)
cli = _client()
TOPICS = ("algebra", "geometry", "number_theory", "combinatorics", "other")
for i in range(0, len(todo), 20):
    batch = todo[i:i + 20]
    listing = "\n\n".join(f"[{p['qid']}] {p['problem'][:400]}" for p in batch)
    r = cli.chat.completions.create(model=MODEL, max_completion_tokens=1200, messages=[
        {"role": "user", "content":
         "Classify each competition math problem into exactly one topic from "
         f"{list(TOPICS)}. Reply ONLY a JSON object mapping id to topic.\n\n" + listing}])
    txt = r.choices[0].message.content or "{}"
    try:
        tags = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
    except ValueError:
        continue
    for qid, t in tags.items():
        if t in TOPICS:
            done[str(qid)] = t
    with open(OUT, "w") as f:
        json.dump(done, f)
    print(f"tagged {len(done)}/{len(problems)}", flush=True)
print("DONE", flush=True)
