"""Text-form scaffold for math: the full form space beside the hint prefix.

items: {"general"|topic: [{"id","kind","text"}]}, kinds skill|example|plan;
p: per-topic injection probability (general items ride along, ALFWorld semantics).
The coin is drawn AT DATA BUILD, once per problem per cycle (all rollouts of a
prompt share it, so groups stay homogeneous). Deterministic per (cycle, qid).
"""
from __future__ import annotations

import copy
import hashlib
import re

TOPICS = ("algebra", "geometry", "number_theory", "combinatorics", "other")
SCOPES = ("general",) + TOPICS
KINDS = ("skill", "example", "plan")
MAX_ITEMS_PER_SCOPE = 8
MAX_CHARS = {"skill": 500, "plan": 500, "example": 1500}
MAX_EDITS_PER_CYCLE = 3
P_MAX = 0.5


def empty_text():
    return {"items": {s: [] for s in SCOPES}, "p": {t: 0.0 for t in TOPICS}, "next_n": 1}


def _norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def apply_item_ops(text, ops):
    """Validated apply -> (new_text, notes). Caps enforced; invalid op voids ALL
    (mirrors the ALFWorld contract: prioritize, don't spray)."""
    tx = copy.deepcopy(text)
    edits = 0
    notes = []
    for op in ops or []:
        kind_op = op.get("op")
        if kind_op == "delete":
            for s in SCOPES:
                tx["items"][s] = [i for i in tx["items"][s] if i["id"] != op.get("id")]
            notes.append(f"delete:{op.get('id')}")
            continue
        if edits >= MAX_EDITS_PER_CYCLE:
            return text, [f"voided: more than {MAX_EDITS_PER_CYCLE} add/update ops"]
        edits += 1
        k = op.get("kind") or "skill"
        body = str(op.get("text") or "").strip()
        if k not in KINDS or not body or len(body) > MAX_CHARS[k]:
            return text, [f"voided: bad kind/text in {kind_op}"]
        if kind_op == "add":
            s = op.get("scope")
            if s not in SCOPES:
                return text, [f"voided: bad scope {s!r}"]
            if len(tx["items"][s]) >= MAX_ITEMS_PER_SCOPE:
                return text, [f"voided: scope {s} full"]
            if any(_norm(i["text"]) == _norm(body) for i in tx["items"][s]):
                return text, ["voided: duplicate text"]
            iid = f"m{tx['next_n']}"
            tx["next_n"] += 1
            tx["items"][s].append({"id": iid, "kind": k, "text": body})
            notes.append(f"add:{s}:{iid}")
        elif kind_op == "update":
            hit = None
            for s in SCOPES:
                for i in tx["items"][s]:
                    if i["id"] == op.get("id"):
                        hit = i
            if hit is None:
                return text, [f"voided: unknown id {op.get('id')!r}"]
            hit["text"] = body
            hit["kind"] = k
            notes.append(f"update:{op.get('id')}")
        else:
            return text, [f"voided: unknown op {kind_op!r}"]
    return tx, notes


def apply_p_ops(text, p_ops):
    notes = []
    for op in p_ops or []:
        t = op.get("topic")
        if t in TOPICS:
            try:
                text["p"][t] = max(0.0, min(P_MAX, float(op.get("p"))))
                notes.append(f"p:{t}={text['p'][t]}")
            except (TypeError, ValueError):
                pass
    return text, notes


def render(text, topic):
    parts = [i["text"] for i in text["items"].get("general", [])]
    parts += [i["text"] for i in text["items"].get(topic or "other", [])]
    if not parts:
        return ""
    return "## Notes.\n" + "\n".join(f"- {t}" for t in parts)


def coin(text, qid, topic, cycle):
    p = float(text["p"].get(topic or "other", 0.0))
    if p <= 0 or not render(text, topic):
        return False
    h = hashlib.sha1(f"{cycle}:{qid}".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 2**32) < p
