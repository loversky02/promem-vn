"""LLMMemoryAgent — the faithful two-phase ProMem memory agent (an LLM), per §3.3.

Runs at the first step and every `interval` steps (the paper's fixed-interval trigger):
  Phase 1 (manage):    emit bank-edit commands (SAVE_KNOWLEDGE / SAVE_PROCEDURAL /
                       UPDATE_STATUS / DELETE) → update the StructuredBank.
  Phase 2 (intervene): decide REMIND <text> or SILENT for the next action-agent call.

`backend(system, user) -> str` — reuse automem.meta.driver.client_from_env (9router).
Unlike the rule-based Gate, the inject-vs-silent decision here is made by an LLM
conditioned on the whole curated bank — this is what the paper actually does.
"""

from __future__ import annotations

from .bank import StructuredBank

PHASE1_SYS = (
    "You maintain a structured MEMORY BANK for a long-horizon agent playing Crafter. "
    "You do NOT play the game; you curate durable execution state the agent is likely to "
    "forget. Given the task, a recent trajectory window, and the current bank, output "
    "bank-edit commands, ONE PER LINE, from:\n"
    "  SAVE_KNOWLEDGE <stable fact: resource seen + where, what unlocks what, a constraint>\n"
    "  SAVE_PROCEDURAL <an attempt and its outcome: what was tried, did it work>\n"
    "  UPDATE_STATUS <one line: current goal / open subgoals / what is stuck>\n"
    "  DELETE <entry-id>\n"
    "Save only NEW, decision-relevant items; avoid duplicating existing entries. "
    "Output NONE if nothing should change."
)

PHASE2_SYS = (
    "You decide whether to proactively REMIND a Crafter agent of something it has likely "
    "forgotten. Given the memory bank and the recent trajectory, output EXACTLY one line:\n"
    "  REMIND <one short, specific, memory-grounded reminder>\n"
    "  SILENT\n"
    "Remind ONLY when a remembered fact, open subgoal, or failed attempt is about to be "
    "ignored or repeated. Do NOT give generic strategy, restate the current observation, "
    "or remind every step. Prefer SILENT."
)


class LLMMemoryAgent:
    def __init__(self, backend, interval: int = 5, window: int = 8):
        self.backend = backend
        self.bank = StructuredBank()
        self.interval = max(1, interval)
        self.window = window
        self.phase1_calls = 0
        self.phase2_calls = 0

    def is_trigger(self, step: int) -> bool:
        return step == 0 or step % self.interval == 0

    def _traj(self, trajectory: list[str]) -> str:
        rows = trajectory[-self.window:]
        return "\n".join(rows) if rows else "(start)"

    # -- phases -----------------------------------------------------------
    def manage(self, task: str, trajectory: list[str]) -> None:
        """Phase 1 only — update the bank (used by both full-bank and active arms)."""
        user = (f"TASK: {task}\n\nRECENT TRAJECTORY:\n{self._traj(trajectory)}\n\n"
                f"CURRENT BANK:\n{self.bank.render_for_memory_agent()}\n\n"
                "Output bank-edit commands (one per line) or NONE.")
        out = self.backend(PHASE1_SYS, user) or ""
        self.phase1_calls += 1
        self._apply(out)

    def intervene(self, trajectory: list[str], obs: str) -> str | None:
        """Phase 2 only — reminder vs silent (active arm)."""
        user = (f"MEMORY BANK:\n{self.bank.render_for_memory_agent()}\n\n"
                f"RECENT TRAJECTORY:\n{self._traj(trajectory)}\n\n"
                f"CURRENT OBSERVATION:\n{obs}\n\nREMIND <text> or SILENT?")
        out = self.backend(PHASE2_SYS, user) or ""
        self.phase2_calls += 1
        return self._parse_intervention(out)

    def step(self, task: str, trajectory: list[str], obs: str) -> str | None:
        """Both phases (active-injection). Returns a reminder string or None."""
        self.manage(task, trajectory)
        return self.intervene(trajectory, obs)

    # -- parsing ----------------------------------------------------------
    def _apply(self, text: str) -> None:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            up = line.upper()
            if up.startswith("SAVE_KNOWLEDGE"):
                self.bank.save_knowledge(line[len("SAVE_KNOWLEDGE"):].strip(": ").strip())
            elif up.startswith("SAVE_PROCEDURAL"):
                self.bank.save_procedural(line[len("SAVE_PROCEDURAL"):].strip(": ").strip())
            elif up.startswith("UPDATE_STATUS"):
                self.bank.update_status(line[len("UPDATE_STATUS"):].strip(": ").strip())
            elif up.startswith("DELETE"):
                self.bank.delete(line[len("DELETE"):].strip(": ").strip())

    @staticmethod
    def _parse_intervention(text: str) -> str | None:
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("REMIND"):
                return s[len("REMIND"):].strip(": ").strip() or None
            if s.upper().startswith("SILENT"):
                return None
        return None  # default to silent when unparseable
