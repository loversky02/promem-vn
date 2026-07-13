"""MemoryBank: fact updates, subgoal derivation, stall + loop detection."""

from promem import MemoryBank


def _info(*unlocked):
    return {"achievements": {a: 1 for a in unlocked}}


def test_observe_reads_unlocked_and_resources():
    b = MemoryBank()
    b.observe("You see a tree and stone. Inventory: wood 2.", _info("collect_wood"))
    assert b.unlocked == {"collect_wood"}
    assert "tree" in b.seen_resources and "stone" in b.seen_resources
    assert b.inventory.get("wood") == 2


def test_open_subgoals_respect_prereqs():
    b = MemoryBank()
    b.observe("start", _info())                 # nothing unlocked
    # collect_wood/collect_drink/... have no prereqs; wood is top priority
    assert b.current_subgoal() == "collect_wood"
    assert "place_table" not in b.open_subgoals()   # needs collect_wood first

    b.observe("start", _info("collect_wood"))
    assert "place_table" in b.open_subgoals()
    assert b.current_subgoal() == "place_table"


def test_stall_steps_grow_then_reset_on_progress():
    b = MemoryBank()
    for _ in range(5):
        b.observe("x", _info())
        b.record_action("move_right", new_unlocks=0)
    assert b.stall_steps() >= 5
    b.observe("x", _info())
    b.record_action("do", new_unlocks=1)            # progress!
    assert b.stall_steps() == 0


def test_repeated_failure_detects_loop():
    b = MemoryBank()
    for _ in range(5):
        b.observe("x", _info())
        b.record_action("move_up", new_unlocks=0)
    loop = b.repeated_failure(window=6, threshold=4)
    assert loop is not None and loop[0] == "move_up" and loop[1] >= 4


def test_suggested_action_maps_craft_and_collect():
    b = MemoryBank()
    b.observe("start", _info("collect_wood"))
    assert b.suggested_action("place_table", "no resources") == "place_table"
    # collect_wood -> 'do' only when a tree is in view
    assert b.suggested_action("collect_wood", "a tree is here") == "do"
    assert b.suggested_action("collect_wood", "nothing") != "do"


def test_render_full_is_raw_facts_without_suggestion():
    b = MemoryBank()
    b.observe("tree. Inventory: wood 1.", _info("collect_wood"))
    dump = b.render_full()
    assert "unlocked" in dump and "open_subgoals" in dump
    assert "SUGGEST" not in dump                    # passive exposure, no nudge
