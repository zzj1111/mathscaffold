"""The investigation loop: tools within budget, then one validated decision.

client_factory() -> an object with .chat.completions.create(...) (OpenAI-compatible).
Everything is auditable: the transcript records each tool call, its args, and a size
digest of what came back; the caller persists it beside the decision.
"""
from __future__ import annotations

import json

from . import tools as T
from .prompts import SYSTEM

MAX_TOOL_CALLS = 10
MAX_EVIDENCE_CHARS = 60_000


def _digest(result):
    s = json.dumps(result, ensure_ascii=False)
    return {"chars": len(s),
            "keys": sorted(result)[:6] if isinstance(result, dict) else None}


def _validate(decision):
    if not isinstance(decision, dict):
        return None
    out = {"diagnosis": str(decision.get("diagnosis") or "")[:2000],
           "item_ops": [], "p_ops": []}
    for op in decision.get("item_ops") or []:
        if isinstance(op, dict) and op.get("op") in ("add", "update", "delete"):
            out["item_ops"].append(op)
    for op in decision.get("p_ops") or []:
        try:
            key = "task" if "task" in op else "topic"
            out["p_ops"].append({key: str(op[key]),
                                 "p": max(0.0, min(0.5, float(op["p"])))})
        except (KeyError, TypeError, ValueError):
            continue
    # pass through untouched: the arm-side normalize() owns cleaning and the cap
    if decision.get("requeue_ops"):
        out["requeue_ops"] = list(decision["requeue_ops"])
    if decision.get("ratio_ops"):
        out["ratio_ops"] = list(decision["ratio_ops"])   # math domain; arm-side clamps
    return out


def investigate_and_propose(client, data, model="gpt-5.5",
                            max_tool_calls=MAX_TOOL_CALLS,
                            max_evidence_chars=MAX_EVIDENCE_CHARS,
                            user_preamble="", tools=None, system=None):
    """Returns (decision|None, transcript). decision None = malformed final output,
    which the caller treats as a no-op (never a crash), mirroring the v1 contract.

    `tools`/`system` select the domain (default: the Search-R1 module/prompt this
    repo was born with; pass teacherflow.alfworld's for the ALFWorld arm)."""
    T_mod = tools if tools is not None else T
    sys_prompt = system if system is not None else SYSTEM
    budget_note = (f"Budgets this cycle: at most {max_tool_calls} tool calls and "
                   f"{max_evidence_chars} characters of returned evidence. Excess calls "
                   f"are refused, not queued.")
    messages = [{"role": "system", "content": sys_prompt}]
    content = user_preamble or ("A new cycle just finished training. Investigate as "
                                "you see fit, then decide.")
    messages.append({"role": "user", "content": content + "\n\n" + budget_note})
    transcript = []
    evidence_chars = 0
    for _ in range(max_tool_calls + 1):
        allow_tools = (len([t for t in transcript if t.get("tool")]) < max_tool_calls
                       and evidence_chars < max_evidence_chars)
        r = client.chat.completions.create(
            model=model, messages=messages,
            tools=T_mod.TOOL_SPECS if allow_tools else None,
            max_completion_tokens=4000)
        msg = r.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        if calls:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                # The budget is checked per ROUND but one round may batch several calls
                # (observed live: 13 calls slipped past a cap of 10). Enforce per CALL:
                # excess calls in the same round get a refusal instead of data.
                used = len([t for t in transcript if t.get("tool")])
                if used >= max_tool_calls:
                    result = {"error": "tool-call budget exhausted; decide with what you have"}
                else:
                    result = T_mod.dispatch(data, c.function.name, args)
                if isinstance(result, dict):
                    result = {**result,
                              "_budget_calls_remaining": max(0, max_tool_calls - used - 1)}
                payload = json.dumps(result, ensure_ascii=False)
                if evidence_chars + len(payload) > max_evidence_chars:
                    result = {"error": "evidence budget exhausted; decide with what you have"}
                    payload = json.dumps(result)
                evidence_chars += len(payload)
                transcript.append({"tool": c.function.name, "args": args,
                                   "result_digest": _digest(result)})
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": payload})
            continue
        text = (msg.content or "").strip()
        if not text:
            # observed live (alf v3 c12'): the API returns an empty final message and
            # a whole cycle's decision degrades to no-op. One explicit retry.
            transcript.append({"final": "", "note": "empty final; retrying once"})
            messages.append({"role": "user",
                             "content": "Your last message was empty. Reply now with "
                                        "ONLY the final JSON decision."})
            r = client.chat.completions.create(model=model, messages=messages,
                                               max_completion_tokens=4000)
            text = (r.choices[0].message.content or "").strip()
        transcript.append({"final": text[:4000]})
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, transcript
        try:
            return _validate(json.loads(text[start:end + 1])), transcript
        except ValueError:
            return None, transcript
    transcript.append({"final": "", "note": "loop budget exhausted without a decision"})
    return None, transcript
