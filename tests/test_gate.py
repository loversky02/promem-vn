"""Gate: fires on stall / loop, respects cooldown, resets cleanly."""

from promem import MemoryBank, Gate
from promem.memory.bank import Attempt


def _stalled_bank(step: int, unlocked=()):
    b = MemoryBank()
    b.observe("a tree nearby", {"achievements": {a: 1 for a in unlocked}})
    b.step = step
    b.last_unlock_step = 0
    return b


def test_fires_on_stalled_subgoal():
    b = _stalled_bank(step=10)                      # stall_steps == 10
    g = Gate(stall_n=8, cooldown_k=5)
    msg = g.maybe_reminder(b, "a tree nearby", step=0)
    assert msg is not None
    assert "collect_wood" in msg and "SUGGEST" in msg


def test_silent_before_stall_threshold():
    b = _stalled_bank(step=3)                       # stall_steps == 3 < 8
    g = Gate(stall_n=8, cooldown_k=5)
    assert g.maybe_reminder(b, "tree", step=0) is None


def test_cooldown_suppresses_second_reminder():
    b = _stalled_bank(step=10)
    g = Gate(stall_n=8, cooldown_k=5)
    assert g.maybe_reminder(b, "tree", step=0) is not None      # fires, last_fire=0
    assert g.maybe_reminder(b, "tree", step=2) is None          # 2-0 < 5 -> silent
    assert g.maybe_reminder(b, "tree", step=6) is not None      # 6-0 >= 5 -> fires


def test_fires_on_behavioral_loop_even_without_stall():
    b = MemoryBank()
    b.observe("tree", {"achievements": {}})
    b.step = 1                                       # low stall
    b.attempts = [Attempt("move_right", False) for _ in range(4)]
    g = Gate(stall_n=99, cooldown_k=5)               # stall trigger disabled
    msg = g.maybe_reminder(b, "tree", step=0)
    assert msg is not None and "repeated" in msg.lower()


def test_reset_clears_last_fire():
    b = _stalled_bank(step=10)
    g = Gate(stall_n=8, cooldown_k=5)
    g.maybe_reminder(b, "tree", step=0)
    g.reset()
    assert g.maybe_reminder(b, "tree", step=1) is not None      # not suppressed after reset
