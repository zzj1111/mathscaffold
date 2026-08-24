"""The investigative Teacher for the math arm — SAME workflow as the ALFWorld and
Search arms (teacherflow: budgeted read-only tools -> one JSON decision), with the
math decision space (per-problem / per-bucket ratio ops).

Mechanical bookkeeping (graduation on bare success, relapse on bare failure) runs
FIRST via controller.adaptive bookkeeping; the Teacher then steers ratios of active
problems. Malformed output or unreachable API degrades to the mechanical rule."""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHERFLOW_PATH = os.environ.get(
    "TEACHERFLOW_PATH",
    _REPO if os.path.isdir(os.path.join(_REPO, "teacherflow")) else
    os.path.join(os.path.expanduser("~"), "teacherflow"))
MODEL = os.environ.get("MS_TEACHER_MODEL", "gpt-5.5")
MAX_BUCKET_OPS, MAX_QID_OPS, MAX_WHERE_OPS = 4, 16, 6
R_MAX = float(os.environ.get("MS_R_MAX", "50"))


def _client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        # same contract as the ALFWorld/Search arms: a key file whose path is in
        # AUTOSCAFFOLD_OPENAI_KEY_FILE (raw key, or an OPENAI_API_KEY= line)
        candidates = [os.environ.get("AUTOSCAFFOLD_OPENAI_KEY_FILE"),
                      os.path.expanduser("~/.openai_key")]
        for p in candidates:
            if not p:
                continue
            try:
                txt = open(p).read()
            except OSError:
                continue
            for line in txt.splitlines():
                if line.strip().startswith("OPENAI_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    break
            key = key or txt.strip()
            if key:
                break
    return OpenAI(api_key=key, timeout=600, max_retries=2)


def normalize(decision, state):
    """Clamp/validate ratio_ops -> (sets, item_ops, p_ops, note). Text-op validation
    happens at apply time (textscaffold enforces its own contract)."""
    if not isinstance(decision, dict):
        return [], [], [], "non-dict -> no-op"
    probs = state.get("problems", state)
    sets = {}
    buckets = qids = wheres = 0
    for op in decision.get("ratio_ops") or []:
        if not isinstance(op, dict):
            continue
        if op.get("scope") == "bucket" and buckets < MAX_BUCKET_OPS:
            buckets += 1
            want = op.get("outcome")
            lo, hi = float(op.get("r_min") or 0), float(op.get("r_max") or R_MAX)
            delta = max(-20.0, min(20.0, float(op.get("delta") or 0)))
            for qid, h in probs.items():
                if h.get("state") == "graduated" or h.get("_outcome") != want:
                    continue
                cur = float(sets.get(qid, h.get("r") or 0))
                if lo <= cur <= hi:
                    sets[qid] = max(0.0, min(R_MAX, cur + delta))
        elif op.get("scope") == "where" and wheres < MAX_WHERE_OPS:
            # SQL-style bulk op over THIS WINDOW's problems: filter by outcome and/or
            # r range and/or this window's success fraction, then delta or set.
            wheres += 1
            w = op.get("where") or {}
            delta = op.get("delta")
            rset = op.get("set")
            if delta is None and rset is None:
                continue
            lo, hi = float(w.get("r_min") or 0), float(w.get("r_max") or R_MAX)
            for qid, h in probs.items():
                if h.get("state") == "graduated" or h.get("_outcome") is None:
                    continue
                if w.get("outcome") and h.get("_outcome") != w["outcome"]:
                    continue
                cur = float(sets.get(qid, h.get("r") or 0))
                if not (lo <= cur <= hi):
                    continue
                sf = h.get("_succ_frac")
                if w.get("succ_min") is not None and (sf is None or sf < float(w["succ_min"])):
                    continue
                if w.get("succ_max") is not None and (sf is None or sf > float(w["succ_max"])):
                    continue
                try:
                    new_r = (float(rset) if rset is not None
                             else cur + max(-20.0, min(20.0, float(delta))))
                except (TypeError, ValueError):
                    break
                sets[qid] = max(0.0, min(R_MAX, new_r))
        elif op.get("scope") == "qid" and qids < MAX_QID_OPS:
            qids += 1
            qid = str(op.get("qid") or "")
            h = probs.get(qid)
            if h and h.get("state") != "graduated" and op.get("set") is not None:
                try:
                    sets[qid] = max(0.0, min(R_MAX, float(op["set"])))
                except (TypeError, ValueError):
                    pass
    return (list(sets.items()), list(decision.get("item_ops") or []),
            list(decision.get("p_ops") or []),
            f"ok ({wheres} where, {buckets} bucket, {qids} qid ops -> {len(sets)} problems; "
            f"{len(decision.get('item_ops') or [])} item, {len(decision.get('p_ops') or [])} p)")


def decide(rollout_log, state, outcomes, cycle, probe_line="", transcript_dir=None,
           problems=None):
    """One Teacher decision. Mutates nothing; returns (sets, note, transcript).
    problems: list from data.load_problems — statements/reference solutions become
    visible in get_traces. Cross-cycle memory: a rolling history.json beside the
    transcripts is surfaced as recent_decisions and appended to on every decision."""
    if TEACHERFLOW_PATH not in sys.path:
        sys.path.insert(0, TEACHERFLOW_PATH)
    from teacherflow import mathdomain as MD
    from teacherflow.data import RunData
    from teacherflow.workflow import investigate_and_propose

    probs = state.get("problems", state)
    # window annotations are per-cycle SCRATCH: they persist in the saved state, so
    # stale ones from earlier cycles must be wiped or bucket/where ops accumulate
    # across windows (seen live on v4: one "mixed -5" op hit 4174 problems, 3x the
    # window, parking thousands at r=45 without evidence at their own revisit)
    for h in probs.values():
        h.pop("_outcome", None)
        h.pop("_succ_frac", None)
    for qid, (succ, n) in outcomes.items():
        if qid in probs:
            probs[qid]["_outcome"] = ("all_fail" if succ == 0 else
                                      ("all_pass" if succ == n else "mixed"))
            probs[qid]["_succ_frac"] = round(succ / n, 3) if n else None
    data = RunData(rollout_log, scaffold_path=None, state_path=None)
    data.scaffold = state
    data.problems = {p["qid"]: p for p in (problems or [])}
    hist_path = os.path.join(transcript_dir, "history.json") if transcript_dir else None
    recent = []
    if hist_path and os.path.exists(hist_path):
        try:
            recent = json.load(open(hist_path))[-6:]
        except (OSError, ValueError):
            recent = []
    data.state = {"recent": recent}
    data.window_cycle = cycle - 1        # the rollouts being judged are cycle-1's
    preamble = (f"Cycle {cycle} just finished training. {probe_line}"
                "Investigate as you see fit, then decide.")
    try:
        rmax = str(int(R_MAX))
        system = (MD.MATH_SYSTEM.replace("0..90", "0.." + rmax)
                  .replace("[0, 90]", "[0, " + rmax + "]"))
        decision, transcript = investigate_and_propose(
            _client(), data, model=MODEL, user_preamble=preamble,
            tools=MD, system=system)
    except Exception as e:
        return None, f"teacher unreachable ({str(e)[:120]}) -> mechanical fallback", []
    # (note: returns (sets, item_ops, p_ops, note) via normalize on success)
    if transcript_dir:
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            with open(os.path.join(transcript_dir, f"c{cycle}.json"), "w") as f:
                json.dump({"decision": decision, "transcript": transcript,
                           "preamble": preamble, "system_prompt_chars": len(system)}, f,
                          ensure_ascii=False, indent=1)
        except OSError:
            pass
    if decision is None:
        return None, "malformed final output -> mechanical fallback", transcript
    result = normalize(decision, state)
    if hist_path:
        try:
            sets, item_ops, p_ops, note = result
            recent.append({"cycle": cycle, "probe": probe_line.strip() or None,
                           "ratio_sets": len(sets), "item_ops": len(item_ops),
                           "p_ops": len(p_ops),
                           "diagnosis": (decision.get("diagnosis") or "")[:240]})
            json.dump(recent[-12:], open(hist_path, "w"), ensure_ascii=False)
        except OSError:
            pass
    return result, "decided", transcript
