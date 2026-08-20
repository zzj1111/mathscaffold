"""Per-problem hint-ratio controllers.

State file: {qid: {"r": float, "state": "active"|"graduated",
                   "hist": [{"cycle", "r", "succ", "n"}, ...]}}

adaptive_update: evidence-driven —
  all-fail  -> r += UP   (pull the problem back into the gradient band; cap R_MAX)
  all-pass  -> r -= DOWN (anneal toward bare; floor 0)
  mixed     -> hold      (it is producing gradient; leave it alone)
  r == 0    : bare probe semantics — any success graduates, all-fail relapses to R0.
static_update: QuestA's global two-stage schedule (50 until switch_step, then 25).
"""
from __future__ import annotations

import json
import os

R0 = float(os.environ.get("MS_R0", "25"))
UP = float(os.environ.get("MS_UP", "15"))
DOWN = float(os.environ.get("MS_DOWN", "15"))
R_MAX = float(os.environ.get("MS_R_MAX", "50"))
RELAPSE_R = float(os.environ.get("MS_RELAPSE_R", "25"))


def load_state(path, problems):
    try:
        with open(path) as f:
            st = json.load(f)
        if "problems" not in st:           # v1 flat shape -> wrap
            st = {"problems": st}
    except OSError:
        st = {"problems": {p["qid"]: {"r": R0, "state": "active", "hist": []}
                           for p in problems}}
    if "text" not in st:
        from . import textscaffold as TS
        st["text"] = TS.empty_text()
    return st


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def adaptive_update(state, outcomes, cycle):
    """outcomes: {qid: (succ, n)} from the recorder. Returns (state, notes)."""
    notes = []
    probs = state.get("problems", state)
    for qid, (succ, n) in outcomes.items():
        h = probs.get(qid)
        if not h or n == 0:
            continue
        h["hist"] = (h.get("hist") or [])[-11:] + [
            {"cycle": cycle, "r": h["r"], "succ": succ, "n": n}]
        if h.get("state") == "graduated":
            if succ == 0:
                h.update(state="active", r=RELAPSE_R)
                notes.append(f"{qid}: relapsed -> r={RELAPSE_R}")
            continue
        if h["r"] <= 0:
            if succ >= 1:
                h["state"] = "graduated"
                notes.append(f"{qid}: GRADUATED (bare {succ}/{n})")
            else:
                h["r"] = RELAPSE_R
                notes.append(f"{qid}: bare probe failed -> r={RELAPSE_R}")
        elif succ == 0:
            new = min(R_MAX, h["r"] + UP)
            if new != h["r"]:
                notes.append(f"{qid}: all-fail -> r={new}")
            h["r"] = new
        elif succ == n:
            h["r"] = max(0.0, h["r"] - DOWN)
            notes.append(f"{qid}: all-pass -> r={h['r']}")
        # mixed: hold
    return state, notes


def static_update(state, outcomes, cycle, switch_cycle):
    """QuestA control: global 50 -> 25 at switch_cycle, no per-problem logic."""
    r = R0 if cycle < switch_cycle else 25.0
    for h in state.get("problems", state).values():
        h["r"] = r
    return state, [f"static schedule: all r={r}"]
