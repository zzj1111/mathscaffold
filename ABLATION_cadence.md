# Ablation: how often should the Teacher change the scaffold?

For whoever is running this. Read the whole thing before starting — the point of the
experiment is in §2 and §6, and if those are not held to, the runs are wasted.

---

## 1. The question

Our Teacher currently decides doses **every 10 training steps**. That number was picked
because it seemed reasonable, never because we measured it.

SAGE decides **every step**, for every problem, with no delay at all: it samples a group,
sees it came out all-wrong, and raises the hint on the spot.

So we do not know which of these two things our system is actually buying:

- **the decision** — a strong model looking at the evidence and choosing well, or
- **the freshness** — simply reacting sooner.

This ablation varies only the cadence and holds the decision-maker fixed.

---

## 2. The arms

Everything is the sagebench setup (Qwen3-4B, SAGE's 15k problems, SAGE's RL
hyperparameters). The **only** difference between arms is how many training steps pass
between two Teacher decisions.

| Arm | steps per decision | cycles to run | total steps |
|---|---|---|---|
| **K5** | 5 | 80 | 400 |
| **K10** | 10 | 40 | 400 | 
| **K20** | 20 | 20 | 400 |

K10 already exists and is at 350 steps — it needs **5 more cycles** to reach 400, not a
fresh run. K5 and K20 are new runs from scratch.

400 steps is chosen so all three divide evenly. Do not change it to a nicer-looking number
for one arm.

---

## 3. Commands

Fill in these three lines once for whichever box you are on, then everything below is
copy-paste:

```
export MS_ROOT=/path/to/mathscaffold
export MS_PYTHON=/path/to/env/bin/python3
export WANDB_API_KEY=...   OPENAI_API_KEY=...
```

**K5 — 80 cycles of 5 steps:**

```
cd $MS_ROOT && git pull
MS_EXP=sagebench_teacher_k5 MS_WORK=$MS_ROOT/runs/sagebench_teacher_k5 \
MS_WANDB_RUN_ID=sagebench_teacher_k5 MS_PROBE_EVERY=20 MS_BARE_EVERY=1 \
bash scripts/launch_sagebench.sh teacher 80 5
```

**K20 — 20 cycles of 20 steps:**

```
cd $MS_ROOT && git pull
MS_EXP=sagebench_teacher_k20 MS_WORK=$MS_ROOT/runs/sagebench_teacher_k20 \
MS_WANDB_RUN_ID=sagebench_teacher_k20 MS_PROBE_EVERY=5 MS_BARE_EVERY=1 \
bash scripts/launch_sagebench.sh teacher 20 20
```

**K10 — continue the existing run 5 more cycles** (do not start it over; it has state):

```
cd $MS_ROOT && git pull
MS_START_CYCLE=35 bash scripts/launch_sagebench.sh teacher 40 10
```

The third argument to `launch_sagebench.sh` is the steps-per-decision. `MS_PROBE_EVERY` is
counted in **cycles**, which is why it differs per arm — 20/10/5 all mean "a full benchmark
probe every 100 training steps". Check the `[preflight]` line the launcher prints: it
reports `steps/cycle` and `total`, and those must match the table in §2.

Run the arms **one at a time** unless you have 16 GPUs. They each want all 8.

---

## 4. What varies, and what must not

**Varies** — all of these are consequences of the treatment, and that is fine:

- how often the Teacher is called
- how big a window it sees each time (K × 128 prompts)
- how many decisions it makes over the run, and therefore its total budget to move doses

**Must be identical across arms:**

- model, dataset, seed, and every RL hyperparameter
- total training steps (400)
- a full benchmark probe every 100 steps
- exactly one fresh scaffold-free probe before each Teacher decision

Do not "improve" anything while these are running. If you find yourself editing a config,
stop and ask first — a second changed variable makes all three runs unreadable.

---

## 5. What to collect

Per arm, at the end:

- `runs/<arm>/probe.json` — benchmark scores per probe
- `runs/<arm>/bare_probe.jsonl` — scaffold-free scores per cycle
- the wandb `*_arm` run, which already carries `ratio/mean`, `ratio/frac_zero`,
  `groups/frac_zero_grad` per cycle
- the Teacher transcripts under the run's transcript directory
- wall-clock per arm, and the number of Teacher calls made

---

## 6. How to read it

**Do not read the benchmark scores first.** AIME24 and AIME25 have 30 problems each; at
n=32 the standard error is about **1.0 point**, so anything under ~2 points apart is
noise. Three arms at 400 steps will very likely land inside each other's error bars on
AIME, and that on its own tells you nothing.

The readouts that actually answer the question are the mechanism ones:

- **`groups/frac_zero_grad`** — the fraction of groups still producing no gradient. This is
  the number the scaffold exists to reduce. In our 1.5B runs it fell from 0.44 to 0.33 in
  the first 100 steps and then stopped moving for 300 more.
- **`ratio/frac_zero`** — how much of the problem pool the Teacher actually got off zero
  dose.
- **`ratio/mean`** — how much total dose it put into the pool.

Compare these three curves across arms at matched steps.

**Three possible outcomes, and what each one means:**

1. **All three are the same.** Cadence is not the lever. Stop trying to make the loop
   faster; the thing to attack is what the Teacher is told and what it can see.
2. **K5 clearly better than K20.** Freshness matters. That is an argument for building the
   per-step trigger — we already compute the all-wrong signal every step for dynamic
   filtering and currently throw it away.
3. **K20 better than K5.** More decisions is worse: the Teacher is over-reacting to noisy
   short windows. That is an argument for *longer* windows, not shorter.

---

## 7. One confound you should know about

The Teacher's per-decision limits (how many bulk and single-problem changes one decision
may contain) are **per decision, not per step**. So K5 gets four times as many decisions
as K20 over the run, and therefore four times the total capacity to move doses.

We are leaving it that way on purpose: "deciding more often" naturally means "intervening
more", and separating them costs another two runs.

But it means a K5 win is ambiguous between *fresher decisions* and *more total
intervention*. If K5 does win, the follow-up is to scale the per-decision quotas inversely
with K and re-run — and that follow-up should be planned, not improvised.

---

## 8. If something breaks

- **A training stage dies.** The arm retries it from its checkpoint up to three times on
  its own. If it exhausts them the arm stops deliberately rather than limping on. Restart
  with `MS_START_CYCLE=<the cycle it died in>`; the launcher prints how.
- **A probe hangs.** Known failure. It has a timeout. If a probe runs longer than about an
  hour, note the cycle and say so — do not kill the training to fix it, probes failing is
  survivable and training is not.
- **GPUs not free at launch.** Preflight refuses to start. Something from a previous run is
  still holding memory; find it before starting, do not force past the check.
- **Anything else.** Write down what you saw and ask. Guessing at a fix mid-run costs more
  than a day of waiting.
