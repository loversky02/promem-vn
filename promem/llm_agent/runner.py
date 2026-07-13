"""Episode runner for the faithful LLM memory agent (two-phase, §3.3).

Arms differ only in what the *unmodified* action agent sees each step:
  no-mem    - raw observation.
  full-bank - obs + the LLM-curated bank (status stays private), refreshed at memory
              steps (Phase 1 only). The paper's 'expose full bank' ablation.
  active    - obs + at most one LLM-decided [REMINDER] at memory steps (Phase 1 + 2).
"""

from __future__ import annotations

import re

from ..runner import EpisodeResult, N_CRAFTER_ACHIEVEMENTS
from .memory_agent import LLMMemoryAgent

LLM_ARMS = ("no-mem", "full-bank", "active")
_WORD = re.compile(r"[a-z0-9_]+")
DEFAULT_TASK = "Play Crafter: survive and unlock as many distinct achievements as possible."


def _toks(t: str) -> int:
    return len(_WORD.findall((t or "").lower()))


def _short(obs: str, n: int = 160) -> str:
    return (obs or "").replace("\n", " ")[:n]


def run_episode_llm(env, action_agent, memory_agent: LLMMemoryAgent, arm: str,
                    task: str = DEFAULT_TASK, max_steps: int = 100, describe_fn=None) -> EpisodeResult:
    if arm not in LLM_ARMS:
        raise ValueError(f"arm must be one of {LLM_ARMS}, got {arm!r}")
    if describe_fn is None:
        from ..envs import default_describe
        describe_fn = default_describe()

    names = list(env.action_names)
    env.reset()
    info: dict = {}
    prev_unlocked = 0
    injects = ctx_tokens = 0
    reward = 0.0
    trajectory: list[str] = []
    trace: list = []

    step = 0
    while step < max_steps:
        obs = describe_fn(env, info)
        reminder = None
        if arm == "full-bank":
            if memory_agent.is_trigger(step):
                memory_agent.manage(task, trajectory)
            bank = memory_agent.bank.render_for_action_agent()
            ctx = f"{obs}\n\n[MEMORY BANK]\n{bank}"
            ctx_tokens += _toks(bank)
        elif arm == "active":
            if memory_agent.is_trigger(step):
                reminder = memory_agent.step(task, trajectory, obs)
            if reminder:
                ctx = f"{obs}\n\n[REMINDER]\n{reminder}"
                injects += 1
                ctx_tokens += _toks(reminder)
            else:
                ctx = obs
        else:  # no-mem
            ctx = obs

        name = action_agent.act(ctx)
        if name not in names:
            name = "noop" if "noop" in names else names[0]
        _, r, done, info = env.step(names.index(name))
        reward += float(r)

        ach = info.get("achievements", {}) if info else {}
        now = sum(1 for v in ach.values() if v > 0)
        new = max(0, now - prev_unlocked)
        prev_unlocked = now

        trajectory.append(f"t{step}: obs[{_short(obs)}] action={name} new_unlock={new}")
        trace.append((step, bool(reminder), name, new))
        step += 1
        if done:
            break

    ach = info.get("achievements", {}) if info else {}
    unlocked = [k for k, v in ach.items() if v > 0]
    return EpisodeResult(
        arm=arm, score=len(unlocked) / N_CRAFTER_ACHIEVEMENTS, steps=step, unlocked=unlocked,
        reward=reward, injects=injects, inject_rate=injects / max(1, step),
        ctx_tokens=ctx_tokens, trace=trace,
    )
