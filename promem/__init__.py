"""ProMem: a proactive memory agent that runs beside an unmodified action agent.

Reproduction of "Remember When It Matters: Proactive Memory Agent for
Long-Horizon Agents" (arXiv 2607.08716), probed on automem-vn's Crafter.

Public surface:
  MemoryBank  - the structured bank (facts / attempts / open subgoals)
  Gate        - the inject-vs-silent policy
  run_episode - the 3-arm episode runner (no-mem / full-context / active-injection)
"""

from .memory.bank import MemoryBank
from .inject.gate import Gate
from .runner import run_episode, EpisodeResult, ARMS

__all__ = ["MemoryBank", "Gate", "run_episode", "EpisodeResult", "ARMS"]
