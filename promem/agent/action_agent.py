"""The ACTION agent — deliberately *unmodified* by ProMem.

It receives a text context (the raw observation, possibly with a passive bank
dump or a single injected reminder appended) and returns ONE game action name.
It has NO memory tooling of its own; whatever memory help it gets arrives only
through the context ProMem chose to inject. This is the paper's "plug-and-play
with an unmodified action agent."

Two backends:
  * HeuristicActionAgent - deterministic, network-free; for CI plumbing only.
  * LLMActionAgent       - a real (small) model via 9router; where the Q1/Q2
                           findings actually come from.
"""

from __future__ import annotations

import re

import numpy as np

_SUGGEST = re.compile(r"SUGGEST:\s*([a-z_]+)", re.IGNORECASE)


def parse_suggest(ctx: str) -> str | None:
    """Extract the single suggested action name from an injected reminder."""
    m = _SUGGEST.search(ctx or "")
    return m.group(1).lower() if m else None


def with_retries(backend, tries: int = 5, base_delay: float = 1.5):
    """Wrap a `(system,user)->str` backend so transient gateway hiccups
    (ConnectionReset, 429/5xx) don't kill a long multi-hundred-call run.
    Exponential backoff; re-raises the last error only after `tries` attempts.
    """
    import time

    def wrapped(system: str, user: str) -> str:
        last = None
        for i in range(tries):
            try:
                return backend(system, user)
            except Exception as e:                       # noqa: BLE001 (harness resilience)
                last = e
                if i < tries - 1:
                    time.sleep(base_delay * (2 ** i))
        raise last

    return wrapped


class HeuristicActionAgent:
    """A forgetful scripted agent, for offline CI ONLY.

    Behaviour by design (NOT a finding — it merely lets the 3-arm harness run
    and produce differing, valid trajectories without a network):
      * If the context carries an explicit `SUGGEST: <act>` (active-injection
        arm), follow it with probability `obey`. This models "a timely, computed
        nudge is easy to act on."
      * It does NOT mine a raw bank dump (full-context arm) for the same cue —
        modelling lost-in-the-middle, the exact decay the paper targets. Whether
        that modelling assumption actually holds for real models is precisely
        what the LLM arm tests; do not read the heuristic numbers as evidence.
    """

    def __init__(self, task_actions: list[str], seed: int = 0, obey: float = 0.85):
        self.task_actions = list(task_actions)
        self.rng = np.random.default_rng(seed)
        self.obey = obey

    def act(self, ctx: str) -> str:
        sug = parse_suggest(ctx)
        if sug and sug in self.task_actions and self.rng.random() < self.obey:
            return sug
        # otherwise explore/interact heuristically off the raw observation
        low = (ctx or "").lower()
        if "tree" in low and self.rng.random() < 0.5:
            return "do" if "do" in self.task_actions else self._rand_move()
        return self._rand_move()

    def _rand_move(self) -> str:
        choices = [a for a in ("move_left", "move_right", "move_up", "move_down", "do", "do")
                   if a in self.task_actions] or self.task_actions
        return choices[int(self.rng.integers(len(choices)))]


class LLMActionAgent:
    """Real action agent via any OpenAI-compatible backend (9router).

    `backend(system, user) -> str` — reuse automem.meta.driver.client_from_env.
    The system prompt gives it NO memory ops: it just plays, so any memory
    benefit is attributable to ProMem's injection, not to the agent.
    """

    SYSTEM = (
        "You are playing Crafter, a long-horizon survival game. Each turn you see "
        "a text observation; sometimes a [REMINDER] or [MEMORY BANK] is appended. "
        "Reply with EXACTLY ONE action name from the allowed list and nothing else."
    )

    def __init__(self, backend, task_actions: list[str]):
        self.backend = backend
        self.task_actions = list(task_actions)

    def act(self, ctx: str) -> str:
        user = (f"{ctx}\n\nAllowed actions: {', '.join(self.task_actions)}.\n"
                "Output ONE action name only.")
        text = self.backend(self.SYSTEM, user) or ""
        return self._coerce(text)

    def _coerce(self, text: str) -> str:
        low = text.lower()
        # exact token match first, then substring, else noop/first action
        for a in self.task_actions:
            if re.search(rf"\b{re.escape(a)}\b", low):
                return a
        return "noop" if "noop" in self.task_actions else self.task_actions[0]
