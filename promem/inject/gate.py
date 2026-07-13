"""Gate: the proactive memory agent's core decision — each step, inject a
memory-grounded reminder or stay silent.

The paper's headline is that *selective intervention beats passive exposure*.
The gate is where "selective" lives. Two triggers, both grounded in the bank:

  1. STALL   - the current open subgoal has gone `stall_n` steps with no new
               achievement (the agent has likely lost the thread → decay).
  2. LOOP    - the agent is repeating a failed action (a behavioral loop).

An anti-spam `cooldown_k` suppresses a fresh reminder for K steps after one
fires. Sweeping (stall_n, cooldown_k) traces the injection-frequency → reward
curve (Q2): aggressive gates remind constantly and should tip into context-spam.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..memory.bank import MemoryBank


@dataclass
class Gate:
    stall_n: int = 8          # fire when a subgoal stalls this many steps
    cooldown_k: int = 5       # stay silent for this many steps after firing
    loop_window: int = 6
    loop_threshold: int = 4

    _last_fire: int = -10_000  # step index of the last injected reminder

    def reset(self) -> None:
        self._last_fire = -10_000

    def maybe_reminder(self, bank: MemoryBank, obs: str, step: int) -> str | None:
        """Return the reminder text to inject, or None to stay silent."""
        if step - self._last_fire < self.cooldown_k:
            return None

        # 1) behavioral loop — interrupt it
        loop = bank.repeated_failure(self.loop_window, self.loop_threshold)
        if loop is not None:
            act, n = loop
            sg = bank.current_subgoal()
            alt = bank.suggested_action(sg, obs) if sg else "do"
            self._last_fire = step
            toward = f" toward '{sg}'" if sg else ""
            return (f"You have repeated '{act}' {n}x with no progress. "
                    f"Try something else{toward}. SUGGEST: {alt}")

        # 2) stalled subgoal — resurface it
        sg = bank.current_subgoal()
        if sg is not None and bank.stall_steps() >= self.stall_n:
            self._last_fire = step
            return (f"Open subgoal '{sg}' is still unmet after "
                    f"{bank.stall_steps()} steps with no new achievement. "
                    f"SUGGEST: {bank.suggested_action(sg, obs)}")

        return None
