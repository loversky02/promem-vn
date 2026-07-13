"""inject_audit: necessary vs wasted labeling from a trace (spam proxy)."""

from promem.analysis import audit_trace, aggregate, InjectAudit


def test_no_injects_is_zero_rate():
    trace = [(0, False, "move_right", 0), (1, False, "do", 1)]
    a = audit_trace(trace)
    assert a.injects == 0 and a.wasted_rate == 0.0


def test_inject_followed_by_progress_is_necessary():
    # inject at step 1; achievement unlocks at step 2 (within window)
    trace = [(0, False, "move_right", 0),
             (1, True, "do", 0),
             (2, False, "do", 1)]
    a = audit_trace(trace, window=5)
    assert a.injects == 1 and a.necessary == 1 and a.wasted == 0


def test_inject_with_no_following_progress_is_wasted():
    trace = [(0, True, "noop", 0)] + [(i, False, "noop", 0) for i in range(1, 10)]
    a = audit_trace(trace, window=5)
    assert a.injects == 1 and a.wasted == 1 and a.wasted_rate == 1.0


def test_progress_outside_window_still_wasted():
    # inject at 0, progress only at step 8 (window=5 -> not counted)
    trace = [(0, True, "do", 0)] + [(i, False, "noop", 0) for i in range(1, 8)] + [(8, False, "do", 1)]
    a = audit_trace(trace, window=5)
    assert a.injects == 1 and a.wasted == 1


def test_immediate_progress_at_inject_step_counts():
    trace = [(0, True, "do", 1)]
    a = audit_trace(trace, window=5)
    assert a.necessary == 1 and a.wasted == 0


def test_aggregate_folds_many_episodes():
    good = [(0, True, "do", 1)]
    bad = [(0, True, "noop", 0), (1, False, "noop", 0)]
    agg = aggregate([good, bad, bad], window=2)
    assert agg.injects == 3 and agg.necessary == 1 and agg.wasted == 2
    assert round(agg.wasted_rate, 3) == round(2 / 3, 3)
