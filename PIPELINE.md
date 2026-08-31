# How the system works

A plain description of the framework, for drawing a diagram from. No results, no numbers,
no error handling — just what happens and in what order.

Words used below: an **item** is one task we train on. A **group** is the several answers we
sample for that item. A **hint** is anything we add to a training prompt to help.

---

## 1. The problem we are solving

We train a model with RL. For each item we sample a group of answers and score them.

The model learns from the *differences* inside the group. Some answers are right, some are
wrong, so it learns to do more of what worked.

If every answer in the group is wrong, there is no difference. There is nothing to learn
from, and the item is effectively skipped.

This does not fix itself. The model stays bad at that item, because it never learns from
it. Training longer does not help — the training is what is not happening.

So we put a hint in the prompt during training. Now a couple of answers come out right,
the group has differences again, and the model learns.

That leaves one question, which is the whole framework:

**How big should the hint be?**

- Too small, and every answer is still wrong. Nothing changed.
- Too big, and the model learns "when I see a hint, keep writing it." That scores well in
  training. At test time there is no hint and it can do nothing.

The right size is different for every item, and it changes as the model gets better. So
somebody has to keep deciding it. That somebody is the **Teacher**: a large model we call
once every N training steps.

---

## 2. What the Teacher may and may not do

**May:** change what goes into the training prompt.

**May not:** touch the loss, the optimizer, the reward function, or any RL setting.

That separation is deliberate. A hint changes *what the model tries*. It never changes
*how we score what it did*.

And we never put a hint in an evaluation prompt. Evaluation is always bare. That is the
only way to tell "it learned the task" apart from "it got good at following hints."

---

## 3. The loop

```
      hint settings  ──>  build training data  ──>  train  ──>  what happened
            ▲                                                         │
            │                                                         │
          checks  <──  Teacher's decision  <──  Teacher's queries  <──┘
            ▲                                          ▲
            └──────────  test with no hints  ──────────┘
```

Three stages, one after another, repeating. Nothing runs in parallel.

```
each cycle:
    A. DECIDE   look at the last batch of training, set hint sizes, build the next data
    B. TRAIN    run STEPS_PER_CYCLE steps of RL on it
    C. MEASURE  test the new model with no hints
```

---

## 4. What gets saved between cycles

These files are what make it a loop instead of a straight line.

| What | Contains | Written in | Read in |
|---|---|---|---|
| **Hint settings** | For each item: how big its hint is now, whether it has graduated, and a log of its past visits. Also the shared note library and how often notes get used. | DECIDE | DECIDE |
| **Training log** | One line per answer the model produced: which item, what score, what hint was in its prompt | TRAIN | DECIDE |
| **Checkpoint** | The model and its optimizer | TRAIN | TRAIN, MEASURE |
| **No-hint scores** | How the model does with no help, one record per cycle | MEASURE | DECIDE |
| **Benchmark scores** | Public benchmark results | MEASURE | DECIDE |
| **Teacher's notebook** | Every query the Teacher made, what came back, and what it decided | DECIDE | DECIDE |

---

## 5. Stage A — DECIDE

**Reads:** hint settings, the last training log, both score files.
**Writes:** the next training data, updated hint settings.

**Step 1. Sort the last batch into three piles.**

Take every answer from the last training log and group them by item. Each item lands in
one of three piles:

- **all wrong** — no learning happened here. This is the pile the framework exists for.
- **mixed** — some right, some wrong. RL is already learning this one on its own.
- **all right** — already solved.

**Step 2. Apply the automatic rules.**

Some changes are not judgement calls, so the Teacher does not make them. The main one: an
item that succeeds *with no hint* graduates out of the hinted pool, and comes back if it
later fails.

**Step 3. Write down what happened to each item.**

For every item served this round, save a line: which cycle, what hint size it actually had,
how it did. Items rotate in and out, so this log is the only way a future cycle can find
out whether a hint change worked. (See §9.)

**Step 4. Get a decision.**

Either from the Teacher (§7), or from a fixed rule if this is a control arm (§11).

**Step 5. Run the decision through the checks (§8) and apply whatever survives.**

**Step 6. Build the next training data.**

Take this cycle's slice of items from a rotating queue, put each item's hint into its
prompt, write the file.

**If no valid decision came back:** leave the hint settings alone and carry on.

---

## 6. Two kinds of hint

There are two independent things the Teacher can adjust. A domain can use either or both.

**Hint size** — how much help one specific item gets. A single number. Zero means no hint
at all. Set per item.

**Note library** — a small shared collection of reusable notes, plus how often they get
attached. Everyone draws from the same library; the difference between items is only
whether a note got attached this time.

Two rules hold for both:

- Hints go into **training** prompts only. Never into evaluation prompts.
- Whether an item gets a note is decided **when the data file is built**, once per item per
  cycle. So all the answers sampled for one prompt saw the same thing. That keeps each
  group internally consistent, which is what the group-relative scoring needs.

---

## 7. How the Teacher decides

The Teacher is a large model given read-only access and a budget. It looks around, then
returns one decision. It cannot change anything directly.

**It is always told:** what is being trained; that evaluation is hint-free, so a hint only
counts if what it produces sticks in the weights; what the two hint types are and their
limits; and that items rotate, so a hint size set now takes effect at that item's *next*
appearance.

**It is told each cycle:** which cycle just finished, the current no-hint scores and
benchmark scores, and exactly how many queries it may make.

**Then it loops:**

```
until it gives an answer, or runs out of budget:
    it either asks for data, or gives its decision
    if it asks: it gets back a bounded chunk of JSON, plus how much budget is left
    when the budget runs out: it is told to decide with what it has
```

**What it can ask for.** Three queries. They report facts only and never interpret
anything — interpreting is the Teacher's job, and a query that did it for them would be
answering the question we are trying to ask.

| Query | Gives back |
|---|---|
| **Summary** | Counts. How many items in each pile, broken down by hint size. Which of the all-wrong items had a note attached (a *what the note says* problem) and which did not (a *hint too small* problem). What happened to last round's all-wrong items — did they come back, did any escape. For items seen before, the pair "last time: this outcome at this hint size → this time: this outcome." The current note library. What the Teacher decided recently. |
| **Item list** | One line per item — id, current hint size, how many answers were right — filtered by pile, hint size, or success rate. Capped per call. |
| **Actual answers** | One item in full: the task, the reference material, and several of the model's actual answers with their scores. Or a quick sweep across several all-wrong items. Capped per call. |

**What it returns.** One JSON object: its reasoning in plain text, plus three lists of
changes — hint sizes, notes, note frequency. All three empty means "change nothing."

**What it remembers.** Its last few decisions are shown back to it next cycle.

---

## 8. The checks

The Teacher's answer is a request. It goes through checks before anything is saved.

| Check | Rule |
|---|---|
| Readable | If the output will not parse, the whole thing is dropped |
| Step size | One decision may move any hint size by at most `MAX_DELTA`. Asking for an absolute value counts as a target to move toward, not a way around the limit |
| Range | Hint sizes are clamped to `0..R_MAX` |
| How many changes | Caps on how many bulk and single-item changes one decision may contain; extras are dropped in order |
| Graduated items | Never get a hint again |
| Big-hint budget | Only `HIGH_DOSE_FRAC` of all items may sit above `HIGH_DOSE_R`. Items already there keep their setting and can still be adjusted. Only newcomers use up the budget, and once it is full, further newcomers get clamped down to the threshold |
| Notes | Limits on library size, note length, and edits per cycle. No duplicates. An id being edited has to exist |
| Frequency | Hard ceiling `P_MAX` |

Two things worth showing in the diagram:

- The checks live **outside** the Teacher. It cannot skip them.
- When anything goes wrong, the fallback is **change nothing** — not "fall back to the
  automatic rule." Falling back to a rule would mean the Teacher arm silently ran a
  different arm's behaviour, and we would not be able to tell.

---

## 9. Hints act late

Each cycle trains on a fresh slice of items, so an item only comes back around every few
cycles.

**A hint size set this cycle does nothing next cycle.** It applies the next time that item
shows up. Next cycle is a different set of items carrying whatever hint sizes they already
had.

Two things follow:

- You cannot judge a hint change by next cycle's numbers. You judge it by the before/after
  pair recorded in §5 step 3 — "last visit: all wrong at this size → this visit: mixed."
  That is why that log exists.
- Note changes are global, so they *do* apply immediately. The two hint types have
  different lag.

This is the least obvious thing about the system. Draw it explicitly.

---

## 10. Stage B — TRAIN, and Stage C — MEASURE

**TRAIN** runs `STEPS_PER_CYCLE` steps of RL on the data DECIDE just built, writes every
answer to the training log, and saves a checkpoint. It can optionally throw away groups
where every answer scored the same and pull in more items until the batch is full of
groups that actually teach something.

**MEASURE** tests that checkpoint **with no hints**, on two fixed sets of items:

- Items from the same distribution the model trains on, but never trained on. Plus items
  that *are* in training. The first says whether it generalizes. **The difference between
  the two says whether it is just memorizing what it has seen.**
- Public benchmarks, on their own schedule.

Both go into the next Teacher call. This is what answers the question the whole system is
built around: *is it learning to solve, or learning to follow hints?* If hinted results
keep improving while the no-hint numbers sit still, it is the second one.

If a measurement fails, training keeps going; the next cycle just has stale numbers.

---

## 11. Control arms

The loop is the same for all three. Only step 4 of DECIDE changes. They run side by side.

| Arm | Who sets the hints |
|---|---|
| **teacher** | The Teacher (§7) |
| **adaptive** | A fixed rule based on which pile the item landed in |
| **static** | Nobody. Hints never change from their starting values |

Teacher vs adaptive asks whether the Teacher's judgement beats a rule. Both vs static asks
whether hinting beats not hinting.

---

## 12. Adding a new domain

The whole Teacher loop, the checks, and the timing are reused as-is. A new domain supplies
four things:

1. **What a logged answer contains.**
2. **How answers group into an item, and what counts as a right answer.**
3. **What a hint is** — what the size number means, and what a note looks like.
4. **The domain's own paragraph** for the Teacher's instructions, which gets pasted in
   front of the shared ones.

Domains differ a lot on point 3. One grades *how much of a written solution* is shown.
Another grades *how much procedural advice* is attached. Which of those works better is
one of the things being studied, not something the framework fixes.

---

## Appendix — the knobs

Nothing above is a hard-coded number. These are the names.

| Name | Controls |
|---|---|
| `STEPS_PER_CYCLE` | Training steps between two Teacher calls |
| `GROUP_SIZE` | Answers sampled per item |
| `MAX_TOOL_CALLS`, `EVIDENCE_BUDGET` | How much the Teacher may look at |
| `MAX_ROWS_PER_CALL`, `MAX_TRACES_PER_CALL` | How much one query returns |
| `RECENT_DECISIONS` | How many past decisions it is shown |
| `MAX_DELTA`, `R_MAX` | Biggest hint change per decision; largest hint allowed |
| `MAX_BUCKET_OPS`, `MAX_WHERE_OPS`, `MAX_QID_OPS` | How many changes per decision |
| `HIGH_DOSE_R`, `HIGH_DOSE_FRAC` | What counts as a big hint, and how many may have one |
| `P_MAX`, `MAX_EDITS_PER_CYCLE` | Note frequency ceiling; note edits per decision |
| `PROBE_EVERY` | How often to measure |
| `SERVE_MULT` | How many extra items to put in each cycle's slice |
