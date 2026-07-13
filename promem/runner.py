"""The 3-arm episode runner.

The ONLY thing that differs across arms is the context the (unmodified) action
agent sees each step:

  no-mem           - raw observation only.
  full-context     - observation + a passive dump of the whole memory bank every
                     step (the paper's "passive bank exposure"; the strong
                     baseline — this is the "just stuff it into the context"
                     objection made concrete).
  active-injection - observation + at most one gate-chosen [REMINDER], only when
                     the gate decides to intervene.

Q1 = does active-injection beat full-context (not just no-mem)?
Q2 = as the gate gets more aggressive, when do injections tip into context-spam?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .memory.bank import MemoryBank
from .inject.gate import Gate

ARMS = ("no-mem", "full-context", "active-injection")
N_CRAFTER_ACHIEVEMENTS = 22
_WORD = re.compile(r"[a-z0-9_]+")


def _toks(text: str) -> int:
    return len(_WORD.findall((text or "").lower()))


@dataclass
class EpisodeResult:
    arm: str
    score: float                       # unlocked / 22
    steps: int
    unlocked: list[str]
    reward: float
    injects: int                       # reminders actually injected (active arm)
    inject_rate: float                 # injects / steps
    ctx_tokens: int                    # total extra context tokens spent (proxy)
    trace: list = field(default_factory=list)  # (step, injected, action, new_unlocks)

    def digest(self) -> dict:
        return {"arm": self.arm, "score": round(self.score, 4), "steps": self.steps,
                "reward": round(self.reward, 3), "injects": self.injects,
                "inject_rate": round(self.inject_rate, 4), "ctx_tokens": self.ctx_tokens,
                "unlocked": self.unlocked}


def _build_context(arm: str, obs: str, bank: MemoryBank, gate: Gate, step: int):
    """Return (context_text, injected_bool, extra_tokens)."""
    if arm == "no-mem":
        return obs, False, 0
    if arm == "full-context":
        dump = bank.render_full()
        return f"{obs}\n\n{dump}", False, _toks(dump)
    if arm == "active-injection":
        reminder = gate.maybe_reminder(bank, obs, step)
        if reminder:
            return f"{obs}\n\n[REMINDER]\n{reminder}", True, _toks(reminder)
        return obs, False, 0
    raise ValueError(f"unknown arm {arm!r}")


def run_episode(env, agent, bank: MemoryBank, gate: Gate, arm: str,
                max_steps: int = 200, describe_fn=None) -> EpisodeResult:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if describe_fn is None:
        from .envs import default_describe
        describe_fn = default_describe()

    action_names = list(env.action_names)
    gate.reset()
    env.reset()
    info: dict = {}
    prev_unlocked = 0
    injects = ctx_tokens = 0
    reward_total = 0.0
    trace: list = []

    step = 0
    while step < max_steps:
        obs = describe_fn(env, info)
        bank.observe(obs, info)

        ctx, injected, extra = _build_context(arm, obs, bank, gate, step)
        ctx_tokens += extra
        injects += int(injected)

        name = agent.act(ctx)
        if name not in action_names:
            name = "noop" if "noop" in action_names else action_names[0]

        _, r, done, info = env.step(action_names.index(name))
        reward_total += float(r)

        ach = info.get("achievements", {}) if info else {}
        now_unlocked = sum(1 for v in ach.values() if v > 0)
        new_unlocks = max(0, now_unlocked - prev_unlocked)
        prev_unlocked = now_unlocked
        bank.record_action(name, new_unlocks)

        trace.append((step, injected, name, new_unlocks))
        step += 1
        if done:
            break

    ach = info.get("achievements", {}) if info else {}
    unlocked = [k for k, v in ach.items() if v > 0]
    score = len(unlocked) / N_CRAFTER_ACHIEVEMENTS
    return EpisodeResult(
        arm=arm, score=score, steps=step, unlocked=unlocked, reward=reward_total,
        injects=injects, inject_rate=injects / max(1, step), ctx_tokens=ctx_tokens,
        trace=trace,
    )
