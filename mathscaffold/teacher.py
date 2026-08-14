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

TEACHERFLOW_PATH = os.environ.get(
    "TEACHERFLOW_PATH", os.path.join(os.path.expanduser("~"), "teacherflow"))
MODEL = os.environ.get("MS_TEACHER_MODEL", "gpt-5.5")
MAX_BUCKET_OPS, MAX_QID_OPS = 4, 16


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
    """Clamp/validate ratio_ops -> list of (qid, new_r). Unknown/graduated qids drop."""
    if not isinstance(decision, dict):
        return [], "non-dict -> no-op"
    sets = {}
    buckets = qids = 0
    for op in decision.get("ratio_ops") or []:
        if not isinstance(op, dict):
            continue
        if op.get("scope") == "bucket" and buckets < MAX_BUCKET_OPS:
            buckets += 1
            want = op.get("outcome")
            lo, hi = float(op.get("r_min") or 0), float(op.get("r_max") or 90)
            delta = max(-20.0, min(20.0, float(op.get("delta") or 0)))
            for qid, h in state.items():
                if h.get("state") == "graduated" or h.get("_outcome") != want:
                    continue
                if lo <= float(h.get("r") or 0) <= hi:
                    sets[qid] = max(0.0, min(90.0, float(h["r"]) + delta))
        elif op.get("scope") == "qid" and qids < MAX_QID_OPS:
            qids += 1
            qid = str(op.get("qid") or "")
            h = state.get(qid)
            if h and h.get("state") != "graduated" and op.get("set") is not None:
                try:
                    sets[qid] = max(0.0, min(90.0, float(op["set"])))
                except (TypeError, ValueError):
                    pass
    return list(sets.items()), f"ok ({buckets} bucket, {qids} qid ops -> {len(sets)} problems)"


def decide(rollout_log, state, outcomes, cycle, probe_line="", transcript_dir=None):
    """One Teacher decision. Mutates nothing; returns (sets, note, transcript)."""
    if TEACHERFLOW_PATH not in sys.path:
        sys.path.insert(0, TEACHERFLOW_PATH)
    from teacherflow import mathdomain as MD
    from teacherflow.data import RunData
    from teacherflow.workflow import investigate_and_propose

    for qid, (succ, n) in outcomes.items():
        if qid in state:
            state[qid]["_outcome"] = ("all_fail" if succ == 0 else
                                      ("all_pass" if succ == n else "mixed"))
    data = RunData(rollout_log, scaffold_path=None, state_path=None)
    data.scaffold = state
    data.state = {}
    preamble = (f"Cycle {cycle} just finished training. {probe_line}"
                "Investigate as you see fit, then decide.")
    try:
        decision, transcript = investigate_and_propose(
            _client(), data, model=MODEL, user_preamble=preamble,
            tools=MD, system=MD.MATH_SYSTEM)
    except Exception as e:
        return None, f"teacher unreachable ({str(e)[:120]}) -> mechanical fallback", []
    if transcript_dir:
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            with open(os.path.join(transcript_dir, f"c{cycle}.json"), "w") as f:
                json.dump({"decision": decision, "transcript": transcript}, f,
                          ensure_ascii=False, indent=1)
        except OSError:
            pass
    if decision is None:
        return None, "malformed final output -> mechanical fallback", transcript
    sets, note = normalize(decision, state)
    return sets, note, transcript
