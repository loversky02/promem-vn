"""Label each injected reminder necessary vs wasted — the ProMem analogue of
napmem-vn's unnecessary-call probe, so "spam" is measured directly (not only via
score).

Trace row = (step, injected, action, new_unlocks). PROXY definition — honest and
correlational; we cannot observe the counterfactual offline:
  necessary := a new achievement unlocks within `window` steps at/after the inject
               (the reminder plausibly led to progress)
  wasted    := the inject fired but no achievement follows within `window`
               (pure context cost, no visible payoff)

This mirrors napmem's offline proxy, which was itself flagged misspecified — so
treat `wasted_rate` as a SPAM PROXY, not ground truth. Its value is the trend:
as a gate gets more aggressive, wasted_rate should rise toward the spam cliff.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InjectAudit:
    injects: int
    necessary: int
    wasted: int

    @property
    def wasted_rate(self) -> float:
        return self.wasted / self.injects if self.injects else 0.0

    def merge(self, other: "InjectAudit") -> "InjectAudit":
        return InjectAudit(self.injects + other.injects,
                           self.necessary + other.necessary,
                           self.wasted + other.wasted)


def audit_trace(trace, window: int = 5) -> InjectAudit:
    """Audit one episode trace. `trace` is a list of
    (step, injected, action, new_unlocks) rows (tuples or lists)."""
    n = len(trace)
    injects = necessary = 0
    for i, row in enumerate(trace):
        injected = bool(row[1])
        if not injected:
            continue
        injects += 1
        # progress at/after the inject, within `window` steps (inclusive of i)
        if any(int(trace[j][3]) > 0 for j in range(i, min(n, i + window + 1))):
            necessary += 1
    return InjectAudit(injects=injects, necessary=necessary, wasted=injects - necessary)


def aggregate(traces, window: int = 5) -> InjectAudit:
    """Fold audit_trace over many episode traces (each a list of rows)."""
    total = InjectAudit(0, 0, 0)
    for tr in traces:
        total = total.merge(audit_trace(tr, window))
    return total
