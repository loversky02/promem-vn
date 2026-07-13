"""Faithful two-phase LLM memory agent — hermetic (stub backend + stub env, $0)."""

from promem.llm_agent import StructuredBank, LLMMemoryAgent
from promem.llm_agent.runner import run_episode_llm, LLM_ARMS


class StubBackend:
    """Canned two-phase LLM: Phase 1 saves a fact + status, Phase 2 reminds (or is silent)."""

    def __init__(self, remind=True):
        self.remind = remind
        self.phase1 = self.phase2 = 0

    def __call__(self, system, user):
        if "bank-edit" in system:                       # Phase 1
            self.phase1 += 1
            return "SAVE_KNOWLEDGE a tree is 1 step down\nUPDATE_STATUS goal: collect_wood"
        self.phase2 += 1                                 # Phase 2
        return "REMIND face the tree and do" if self.remind else "SILENT"


class StubEnv:
    def __init__(self):
        self.action_names = ["noop", "move_right", "do"]
        self._ach = {}

    def reset(self):
        self._ach = {}

    def step(self, a_idx):
        if self.action_names[a_idx] == "do":
            self._ach["collect_wood"] = 1
        return None, 0.0, False, {"achievements": dict(self._ach)}


def _describe(env, info):
    return "You are @ facing down. A tree is 1 step down. Inventory: (empty)."


class ConstantAgent:
    def __init__(self, name): self.name = name
    def act(self, ctx): return self.name


# -- bank --------------------------------------------------------------
def test_bank_edits_and_private_status():
    b = StructuredBank()
    kid = b.save_knowledge("tree at 1 down")
    b.save_procedural("did move_right -> nothing")
    b.update_status("goal: collect_wood")
    assert "goal: collect_wood" in b.render_for_memory_agent()
    assert "goal: collect_wood" not in b.render_for_action_agent()   # status is private
    assert "tree at 1 down" in b.render_for_action_agent()
    assert b.delete(kid) and "tree at 1 down" not in b.render_for_action_agent()


def test_apply_parses_edit_commands():
    a = LLMMemoryAgent(StubBackend())
    a._apply("SAVE_KNOWLEDGE wood is west\nUPDATE_STATUS need a table\nSAVE_PROCEDURAL tried do -> ok\nNONE")
    assert any("wood is west" in e.content for e in a.bank.knowledge)
    assert any("tried do -> ok" in e.content for e in a.bank.procedural)
    assert a.bank.status == "need a table"


def test_parse_intervention():
    assert LLMMemoryAgent._parse_intervention("REMIND go chop the tree") == "go chop the tree"
    assert LLMMemoryAgent._parse_intervention("SILENT") is None
    assert LLMMemoryAgent._parse_intervention("um i am not sure") is None


def test_step_runs_both_phases():
    be = StubBackend(remind=True)
    a = LLMMemoryAgent(be, interval=5)
    rem = a.step("task", ["t0: ..."], "obs")
    assert be.phase1 == 1 and be.phase2 == 1
    assert rem == "face the tree and do"
    assert a.bank.knowledge and a.bank.status == "goal: collect_wood"


# -- runner ------------------------------------------------------------
def test_llm_arms_run_and_count_correctly():
    # no-mem: no memory calls, no context cost
    be = StubBackend()
    r = run_episode_llm(StubEnv(), ConstantAgent("noop"), LLMMemoryAgent(be, interval=5),
                        "no-mem", max_steps=20, describe_fn=_describe)
    assert r.injects == 0 and r.ctx_tokens == 0 and be.phase1 == 0 and be.phase2 == 0

    # full-bank: Phase 1 runs at triggers, bank exposed every step, no injects
    be = StubBackend()
    r = run_episode_llm(StubEnv(), ConstantAgent("noop"), LLMMemoryAgent(be, interval=5),
                        "full-bank", max_steps=20, describe_fn=_describe)
    assert r.injects == 0 and r.ctx_tokens > 0 and be.phase1 == 4 and be.phase2 == 0

    # active: both phases at triggers -> reminders injected
    be = StubBackend(remind=True)
    r = run_episode_llm(StubEnv(), ConstantAgent("noop"), LLMMemoryAgent(be, interval=5),
                        "active", max_steps=20, describe_fn=_describe)
    assert r.injects == 4 and be.phase1 == 4 and be.phase2 == 4 and r.ctx_tokens > 0


def test_silent_memory_agent_never_injects():
    be = StubBackend(remind=False)
    r = run_episode_llm(StubEnv(), ConstantAgent("do"), LLMMemoryAgent(be, interval=5),
                        "active", max_steps=20, describe_fn=_describe)
    assert r.injects == 0 and be.phase2 == 4          # asked 4 times, always silent
    assert "collect_wood" in r.unlocked               # ConstantAgent('do') still progresses
