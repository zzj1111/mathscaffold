"""Read-only view over one training run's files. Pure loads, no verl, no GPU.

The window is the caller's business: pass `tail_rows` to bound how much of the
recorder the Teacher may see (one cycle's worth by default at the call site).
"""
from __future__ import annotations

import collections
import json


class RunData:
    def __init__(self, rollout_log, scaffold_path=None, state_path=None, tail_rows=None):
        rows = []
        try:
            with open(rollout_log) as f:
                lines = f.read().strip().splitlines()
            prev_lines = []
            if tail_rows:
                prev_lines = lines[-2 * int(tail_rows):-int(tail_rows)]
                lines = lines[-int(tail_rows):]
            # the window BEFORE this one, so tools can tell recurring all-fail
            # groups (RL cannot escape on its own) from first-time ones
            self.prev_rows = []
            for ln in prev_lines:
                try:
                    self.prev_rows.append(json.loads(ln))
                except ValueError:
                    continue
            for ln in lines:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    continue
        except OSError:
            pass
        if not hasattr(self, "prev_rows"):
            self.prev_rows = []
        self.rows = rows
        self.groups = collections.OrderedDict()
        for r in rows:
            uid = r.get("uid")
            if uid is None:
                continue
            self.groups.setdefault(uid, []).append(r)
        self.scaffold = self._load(scaffold_path) or {}
        self.state = self._load(state_path) or {}

    @staticmethod
    def _load(path):
        if not path:
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def categories(self):
        return sorted({r.get("data_source") for r in self.rows if r.get("data_source")})
