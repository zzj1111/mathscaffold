"""Per-cycle wandb publisher for the arm loop (separate run from the trainer's).

Logs, on the arm's own x-axis `cycle`: hint-ratio distribution, group composition
(all_fail / mixed / all_pass), all-fail fate (escaped / still), graduations, text
scaffold size and dose, and — for the teacher arm — the decision itself as a Table
row (diagnosis, ops counts, ratio moves, item texts) so every decision is readable
in the UI. Enabled by MS_WANDB=1; entity/project from env; failures never break the
loop."""
from __future__ import annotations

import json
import os


def publish(work, arm, cycle, state, outcomes, notes, transcript_path=None, inj_info=None):
    if os.environ.get("MS_WANDB", "0") != "1":
        return
    try:
        import wandb
        run = wandb.init(project=os.environ.get("MS_WANDB_PROJECT", "mathscaffold"),
                         entity=os.environ.get("MS_WANDB_ENTITY") or None,
                         id=f"{os.environ.get('MS_EXP', arm)}_arm", resume="allow",
                         name=f"{os.environ.get('MS_EXP', arm)}_arm", reinit=True,
                         settings=wandb.Settings(init_timeout=60, _disable_stats=True))
        probs = state.get("problems", {})
        text = state.get("text") or {}
        rs = [float(h.get("r") or 0) for h in probs.values()]
        comp = {"all_fail": 0, "mixed": 0, "all_pass": 0}
        for q, (s, n) in outcomes.items():
            comp["all_fail" if s == 0 else ("all_pass" if s == n else "mixed")] += 1
        payload = {
            "cycle": cycle,
            "ratio/mean": sum(rs) / len(rs) if rs else 0,
            "ratio/frac_zero": sum(1 for r in rs if r <= 0) / len(rs) if rs else 0,
            "ratio/frac_ge50": sum(1 for r in rs if r >= 50) / len(rs) if rs else 0,
            "groups/all_fail": comp["all_fail"], "groups/mixed": comp["mixed"],
            "groups/all_pass": comp["all_pass"],
            "groups/frac_zero_grad": ((comp["all_fail"] + comp["all_pass"]) /
                                      max(1, sum(comp.values()))),
            "state/graduated": sum(1 for h in probs.values() if h.get("state") == "graduated"),
            "text/items": sum(len(v) for v in (text.get("items") or {}).values()),
            "text/p": float((text.get("p") or {}).get("general", 0)),
            "ctrl/n_changes": len(notes or []),
        }
        # scaffold USAGE: realized injection fraction, injected-vs-bare group
        # composition (the content-vs-dose evidence), dose histogram, and the
        # current text items verbatim
        if inj_info:
            tot = max(1, inj_info["rows_total"])
            payload["scaffold/frac_rows_injected"] = inj_info["rows_injected"] / tot
            for side in ("text", "bare"):
                c = inj_info[side]; n = max(1, sum(c.values()))
                payload[f"scaffold/{side}_problems"] = sum(c.values())
                payload[f"scaffold/all_fail_rate_{side}"] = c["all_fail"] / n
            payload["scaffold/all_fail_rate_gap"] = (payload["scaffold/all_fail_rate_bare"]
                                                     - payload["scaffold/all_fail_rate_text"])
        if rs:
            payload["ratio/hist"] = wandb.Histogram(rs)
            cnt = {}
            for r_ in rs:
                cnt[int(round(r_))] = cnt.get(int(round(r_)), 0) + 1
            for r_, n_ in sorted(cnt.items()):
                if n_ >= 5:
                    payload[f"ratio/n_at_{r_}"] = n_
        try:
            itbl = wandb.Table(columns=["scope", "id", "kind", "p_effective", "text"])
            pmap = text.get("p") or {}
            for sc, items in (text.get("items") or {}).items():
                for it in items:
                    itbl.add_data(sc, it.get("id"), it.get("kind"),
                                  float(pmap.get(sc, pmap.get("general", 0)) or 0),
                                  str(it.get("text") or "")[:800])
            payload["scaffold/items"] = itbl
        except Exception:
            pass
        # probe results if present
        pf = os.path.join(work, "probe.json")
        if os.path.exists(pf):
            try:
                pr = json.load(open(pf))
                for k, v in pr.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "cycle":
                        payload[f"probe/{k}"] = float(v)
                for k, v in (pr.get("stderr") or {}).items():
                    payload[f"probe/{k}_stderr"] = float(v)
            except (OSError, ValueError, TypeError):
                pass
        # per-cycle bare (hint-free) probe on held-out / in-training 200-problem sets
        bpf = os.path.join(work, "bare_probe.json")
        if os.path.exists(bpf):
            try:
                bp = json.load(open(bpf))
                for which in ("heldout", "train"):
                    for k in ("pass1", "stderr", "solved_any", "truncated", "mean_chars"):
                        if k in (bp.get(which) or {}):
                            payload[f"bare/{which}_{k}"] = float(bp[which][k])
                payload["bare/gap_train_minus_heldout"] = float(bp.get("gap_train_minus_heldout", 0))
                payload["bare/step"] = float(bp.get("step", 0))
            except (OSError, ValueError, TypeError):
                pass
        # teacher decision as a table row
        if transcript_path and os.path.exists(transcript_path):
            try:
                d = json.load(open(transcript_path))
                dec = d.get("decision") or {}
                tools = [t["tool"] for t in d.get("transcript", []) if t.get("tool")]
                tbl = wandb.Table(columns=["cycle", "n_tool_calls", "tools", "diagnosis",
                                           "ratio_ops", "item_ops", "p_ops"])
                tbl.add_data(cycle, len(tools), ",".join(tools),
                             (dec.get("diagnosis") or "")[:2000],
                             json.dumps(dec.get("ratio_ops") or [], ensure_ascii=False)[:2000],
                             json.dumps(dec.get("item_ops") or [], ensure_ascii=False)[:4000],
                             json.dumps(dec.get("p_ops") or [], ensure_ascii=False))
                payload["teacher/decision"] = tbl
                payload["teacher/n_tool_calls"] = len(tools)
                payload["teacher/n_item_ops"] = len(dec.get("item_ops") or [])
                payload["teacher/n_ratio_ops"] = len(dec.get("ratio_ops") or [])
            except (OSError, ValueError):
                pass
        run.define_metric("cycle")
        for k in payload:
            if k != "cycle":
                run.define_metric(k, step_metric="cycle")
        run.log(payload)
        run.finish()
    except Exception as e:  # observability must never break the loop
        print(f"[wandb] publish skipped: {type(e).__name__}: {str(e)[:120]}")
