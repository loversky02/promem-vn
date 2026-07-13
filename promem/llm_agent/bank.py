"""StructuredBank — the paper's memory bank B_t = (status, knowledge, procedural).

Faithful to arXiv 2607.08716 §3.2:
  * status     — the memory agent's PRIVATE progress model; never shown to the action agent.
  * knowledge  — stable facts (resource locations, what unlocks what, constraints).
  * procedural — attempts + outcomes (what was tried, whether it worked).

The bank is edited by the LLM memory agent's Phase-1 tool-style commands, not free-form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9_]+")


@dataclass
class Entry:
    id: str
    content: str


class StructuredBank:
    def __init__(self) -> None:
        self.status: str = ""
        self.knowledge: list[Entry] = []
        self.procedural: list[Entry] = []
        self._n = 0

    def _nid(self, p: str) -> str:
        self._n += 1
        return f"{p}{self._n}"

    # -- edit commands (Phase 1) ------------------------------------------
    def save_knowledge(self, content: str) -> str | None:
        content = content.strip()
        if not content:
            return None
        self.knowledge.append(Entry(self._nid("k"), content))
        return self.knowledge[-1].id

    def save_procedural(self, content: str) -> str | None:
        content = content.strip()
        if not content:
            return None
        self.procedural.append(Entry(self._nid("p"), content))
        return self.procedural[-1].id

    def update_status(self, content: str) -> None:
        self.status = content.strip()

    def delete(self, entry_id: str) -> bool:
        entry_id = entry_id.strip()
        before = len(self.knowledge) + len(self.procedural)
        self.knowledge = [e for e in self.knowledge if e.id != entry_id]
        self.procedural = [e for e in self.procedural if e.id != entry_id]
        return before != len(self.knowledge) + len(self.procedural)

    # -- rendering --------------------------------------------------------
    def render(self, include_status: bool) -> str:
        lines: list[str] = []
        if include_status:
            lines.append(f"STATUS: {self.status or '(none)'}")
        lines.append("KNOWLEDGE:")
        lines += [f"  [{e.id}] {e.content}" for e in self.knowledge] or ["  (none)"]
        lines.append("PROCEDURAL:")
        lines += [f"  [{e.id}] {e.content}" for e in self.procedural] or ["  (none)"]
        return "\n".join(lines)

    def render_for_memory_agent(self) -> str:
        return self.render(include_status=True)

    def render_for_action_agent(self) -> str:
        # status stays PRIVATE — the action agent never sees the memory agent's working model
        return self.render(include_status=False)

    def tokens(self) -> int:
        return len(_WORD.findall(self.render_for_action_agent().lower()))
