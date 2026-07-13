"""MemoryBank: the structured state the proactive memory agent maintains from
the recent trajectory — the thing that fights *behavioral state decay*.

Three fields, per the paper's description of a "structured memory bank":
  * facts         - durable world/task facts observed (unlocked achievements,
                    inventory, resources seen nearby)
  * attempts      - actions tried + whether they made progress (to catch loops)
  * open_subgoals - achievements whose prerequisites are met but not yet unlocked

Kept dependency-light (stdlib only) and env-agnostic: `observe()` reads the text
observation + Crafter's `info["achievements"]`, so the bank never imports crafter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Crafter tech tree (approximate, honest): achievement -> prerequisite
#     achievements that must already be unlocked before it is reachable. ---
PREREQS: dict[str, list[str]] = {
    "collect_wood": [],
    "place_table": ["collect_wood"],
    "make_wood_pickaxe": ["place_table"],
    "make_wood_sword": ["place_table"],
    "collect_stone": ["make_wood_pickaxe"],
    "place_stone": ["collect_stone"],
    "place_furnace": ["collect_stone"],
    "make_stone_pickaxe": ["collect_stone", "place_table"],
    "make_stone_sword": ["collect_stone", "place_table"],
    "collect_coal": ["make_wood_pickaxe"],
    "collect_iron": ["make_stone_pickaxe"],
    "make_iron_pickaxe": ["collect_iron", "collect_coal", "place_furnace", "place_table"],
    "make_iron_sword": ["collect_iron", "collect_coal", "place_furnace", "place_table"],
    "collect_diamond": ["make_iron_pickaxe"],
    "collect_drink": [],
    "collect_sapling": [],
    "place_plant": ["collect_sapling"],
    "eat_plant": ["place_plant"],
    "eat_cow": [],
    "defeat_zombie": [],
    "defeat_skeleton": [],
    "wake_up": [],
}

# Priority along the "tech spine" first (unlocks the most downstream), then the
# opportunistic side achievements. Used to pick THE current subgoal to surface.
PRIORITY: list[str] = [
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
    "place_furnace", "make_stone_pickaxe", "collect_coal", "collect_iron",
    "make_iron_pickaxe", "collect_diamond", "make_stone_sword", "make_wood_sword",
    "make_iron_sword", "place_stone", "collect_drink", "collect_sapling",
    "place_plant", "eat_plant", "eat_cow", "wake_up",
    "defeat_zombie", "defeat_skeleton",
]

# What concrete game action pushes a subgoal forward. `do` = interact with the
# tile you face (chop tree / mine / drink / hit). Only helpful when actually
# positioned right — so following a suggestion is NOT a guaranteed win.
SUBGOAL_ACTION: dict[str, str] = {
    "place_table": "place_table",
    "make_wood_pickaxe": "make_wood_pickaxe",
    "make_wood_sword": "make_wood_sword",
    "place_stone": "place_stone",
    "place_furnace": "place_furnace",
    "make_stone_pickaxe": "make_stone_pickaxe",
    "make_stone_sword": "make_stone_sword",
    "make_iron_pickaxe": "make_iron_pickaxe",
    "make_iron_sword": "make_iron_sword",
    "place_plant": "place_plant",
    "wake_up": "sleep",
    # collect_* / eat_* / defeat_* are resolved by facing + `do`
}

# resource words we scan the text observation for
_RESOURCE_WORDS = ("tree", "stone", "coal", "iron", "diamond", "water",
                   "grass", "sand", "path", "lava", "cow", "zombie", "skeleton")
_WORD = re.compile(r"[a-z_]+")


@dataclass
class Attempt:
    action: str
    made_progress: bool


class MemoryBank:
    """Structured, updated-each-step memory of one episode."""

    def __init__(self) -> None:
        self.unlocked: set[str] = set()
        self.inventory: dict[str, int] = {}
        self.seen_resources: set[str] = set()
        self.attempts: list[Attempt] = []
        self.step: int = 0
        self.last_unlock_step: int = 0          # step of most recent new achievement
        self._prev_unlocked_count: int = 0

    # -- updates -----------------------------------------------------------
    def observe(self, obs: str, info: dict | None) -> None:
        """Refresh facts from the current observation + env info (pre-action)."""
        self.step += 1
        ach = (info or {}).get("achievements", {}) or {}
        self.unlocked = {k for k, v in ach.items() if v > 0}
        low = obs.lower() if obs else ""
        for w in _RESOURCE_WORDS:
            if w in low:
                self.seen_resources.add(w)
        self._parse_inventory(obs)

    def record_action(self, action: str, new_unlocks: int) -> None:
        """Log the action just executed and whether it advanced achievements."""
        progressed = new_unlocks > 0
        if progressed:
            self.last_unlock_step = self.step
        self.attempts.append(Attempt(action=action, made_progress=progressed))

    def _parse_inventory(self, obs: str) -> None:
        # observation line: "Inventory: wood 2, stone 1." (best-effort)
        m = re.search(r"Inventory:\s*(.+)", obs or "")
        if not m:
            return
        inv: dict[str, int] = {}
        for chunk in m.group(1).split(","):
            parts = chunk.strip().rstrip(".").split()
            if len(parts) >= 2 and parts[-1].isdigit():
                inv[" ".join(parts[:-1])] = int(parts[-1])
        if inv:
            self.inventory = inv

    # -- queries the gate uses --------------------------------------------
    def open_subgoals(self) -> list[str]:
        """Achievements not yet unlocked whose prerequisites are all met."""
        return [a for a in PRIORITY
                if a not in self.unlocked
                and all(p in self.unlocked for p in PREREQS.get(a, []))]

    def current_subgoal(self) -> str | None:
        og = self.open_subgoals()
        return og[0] if og else None

    def stall_steps(self) -> int:
        """Steps since the last new achievement (proxy for 'stuck / decayed')."""
        return self.step - self.last_unlock_step

    def repeated_failure(self, window: int = 6, threshold: int = 4) -> tuple[str, int] | None:
        """The action repeated >=threshold times in the last `window` steps with
        no progress — a behavioral loop worth interrupting."""
        recent = self.attempts[-window:]
        if len(recent) < threshold:
            return None
        counts: dict[str, int] = {}
        for a in recent:
            if not a.made_progress:
                counts[a.action] = counts.get(a.action, 0) + 1
        if not counts:
            return None
        act, n = max(counts.items(), key=lambda kv: kv[1])
        return (act, n) if n >= threshold else None

    def suggested_action(self, subgoal: str, obs: str) -> str:
        """Concrete action that pushes `subgoal` forward (for the SUGGEST cue)."""
        if subgoal in SUBGOAL_ACTION:
            return SUBGOAL_ACTION[subgoal]
        # collect_* / eat_* / defeat_*: interact if the target tile is in view,
        # else move to explore toward it.
        target = subgoal.split("_", 1)[-1]                  # wood, stone, cow...
        alias = {"wood": "tree", "drink": "water", "sapling": "grass"}.get(target, target)
        if alias in (obs or "").lower():
            return "do"
        return "move_right"                                 # neutral explore nudge

    # -- rendering ---------------------------------------------------------
    def render_full(self) -> str:
        """Passive dump of the whole bank — the `full-context` arm's payload.

        Deliberately RAW facts, no computed suggestion: this is the paper's
        "passive bank exposure" that active injection must beat.
        """
        unlocked = ", ".join(sorted(self.unlocked)) or "none"
        inv = ", ".join(f"{k} {v}" for k, v in self.inventory.items()) or "empty"
        seen = ", ".join(sorted(self.seen_resources)) or "none"
        subgoals = ", ".join(self.open_subgoals()) or "none"
        recent = "; ".join(f"{a.action}{'+' if a.made_progress else ''}"
                           for a in self.attempts[-8:]) or "none"
        return (
            "[MEMORY BANK]\n"
            f"unlocked: {unlocked}\n"
            f"inventory: {inv}\n"
            f"seen: {seen}\n"
            f"open_subgoals: {subgoals}\n"
            f"recent_actions: {recent}"
        )
