"""3-arm runner: hermetic (stub env + stub describe_fn, no crafter/automem).

Verifies PLUMBING only — that the three arms differ in the context the action
agent receives and that injections are counted. Findings need the LLM arm.
"""

from promem import MemoryBank, Gate, run_episode, ARMS
from promem.agent import HeuristicActionAgent


class StubEnv:
    """A minimal Crafter-shaped env: `do` chops the tree (unlocks collect_wood)."""

    def __init__(self):
        self.action_names = ["noop", "move_right", "do", "place_table", "make_wood_pickaxe"]
        self._ach: dict[str, int] = {}

    def reset(self):
        self._ach = {}

    def step(self, a_idx):
        name = self.action_names[a_idx]
        reward = 0.0
        if name == "do":                              # interacting with the tree
            self._ach["collect_wood"] = 1
            reward = 1.0
        return None, reward, False, {"achievements": dict(self._ach)}


def _describe(env, info):
    return "You are @ facing down. A tree is 1 step down. Inventory: (empty)."


class ConstantAgent:
    def __init__(self, name): self.name = name
    def act(self, ctx): return self.name


def test_all_arms_run_and_return_valid_result():
    for arm in ARMS:
        env = StubEnv()
        agent = HeuristicActionAgent(env.action_names, seed=1)
        res = run_episode(env, agent, MemoryBank(), Gate(4, 2), arm,
                          max_steps=30, describe_fn=_describe)
        assert res.arm == arm
        assert res.steps == 30
        assert 0.0 <= res.score <= 1.0


def test_arm_context_costs_differ():
    # no-mem: no extra context, no injects
    env = StubEnv()
    r_none = run_episode(env, ConstantAgent("noop"), MemoryBank(), Gate(4, 2),
                         "no-mem", max_steps=20, describe_fn=_describe)
    assert r_none.ctx_tokens == 0 and r_none.injects == 0

    # full-context: pays context tokens every step, but injects nothing
    env = StubEnv()
    r_full = run_episode(env, ConstantAgent("noop"), MemoryBank(), Gate(4, 2),
                         "full-context", max_steps=20, describe_fn=_describe)
    assert r_full.ctx_tokens > 0 and r_full.injects == 0

    # active-injection: a stalled noop agent triggers reminders
    env = StubEnv()
    r_act = run_episode(env, ConstantAgent("noop"), MemoryBank(), Gate(3, 2),
                        "active-injection", max_steps=20, describe_fn=_describe)
    assert r_act.injects > 0
    assert r_act.inject_rate > 0


def test_invalid_action_name_is_coerced():
    env = StubEnv()
    res = run_episode(env, ConstantAgent("fly_to_moon"), MemoryBank(), Gate(4, 2),
                      "no-mem", max_steps=5, describe_fn=_describe)
    # coerced to noop -> nothing unlocks, but the episode completes cleanly
    assert res.steps == 5 and res.score == 0.0


def test_following_suggestions_can_make_progress():
    # end-to-end plumbing: an obeying agent that gets a 'do' suggestion unlocks
    # collect_wood in the stub env (NOT a finding — just proves the wire works).
    env = StubEnv()
    agent = HeuristicActionAgent(env.action_names, seed=0, obey=1.0)
    res = run_episode(env, agent, MemoryBank(), Gate(3, 1), "active-injection",
                      max_steps=40, describe_fn=_describe)
    assert res.score > 0.0
    assert "collect_wood" in res.unlocked
