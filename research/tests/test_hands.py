"""The parts of the harness that must be right before any number it prints
means anything.

Nothing here touches a real screen. The observation backends are exercised
against constructed elements, because a test that depends on what happens to be
in the foreground is a test that fails differently on every machine, and this
file has to be able to say the guard works on a build server too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hands import trace as trace_module
from hands.actions import Actions, escape_text, to_send_keys
from hands.elements import Element, Observation, Rect
from hands.guard import Guard
from hands.observe import fuse


def element(eid=0, name="Save", control_type="Button", box=(0, 0, 10, 10),
            source="uia", **kw):
    return Element(eid=eid, name=name, control_type=control_type,
                   rect=Rect(*box), source=source, **kw)


# --- geometry ---------------------------------------------------------------


def test_a_box_knows_where_its_middle_is():
    assert Rect(10, 20, 30, 60).center == (20, 40)


def test_boxes_that_miss_each_other_score_zero():
    assert Rect(0, 0, 10, 10).iou(Rect(50, 50, 60, 60)) == 0.0


def test_identical_boxes_score_one():
    assert Rect(0, 0, 10, 10).iou(Rect(0, 0, 10, 10)) == 1.0


# --- the guard --------------------------------------------------------------


def test_a_delete_button_is_refused_when_nobody_can_be_asked():
    """The unattended default. Fails closed, because the harness runs with
    nobody watching and a silent yes is the one outcome that cannot be undone."""
    guard = Guard()
    decision = guard.check("click", {}, element(name="Delete"))
    assert not decision
    assert "delete" in decision.reason
    assert guard.blocked


def test_an_ordinary_button_goes_straight_through():
    assert Guard().check("click", {}, element(name="Save"))


def test_undelete_is_not_delete():
    """Substring matching would stop this, and stopping the wrong button costs
    a task that should have succeeded."""
    assert Guard().check("click", {}, element(name="Undelete"))
    assert Guard().check("click", {}, element(name="Resend later"))


def test_windows_accelerators_and_ellipses_do_not_hide_the_word():
    """Real Windows labels are "&Delete" and "Delete...", and a guard that only
    matches the bare word misses the exact buttons it exists to catch."""
    assert not Guard().check("click", {}, element(name="&Delete"))
    assert not Guard().check("click", {}, element(name="Delete..."))


def test_a_delete_keypress_is_caught_even_though_it_has_no_label():
    """Typing into a focused list and pressing delete destroys something
    without ever pressing a button the guard could read."""
    assert not Guard().check("hotkey", {"keys": "shift+delete"})


def test_a_yes_from_the_confirmer_lets_it_through():
    guard = Guard(confirm=lambda question: True)
    assert guard.check("click", {}, element(name="Send"))
    assert not guard.blocked


def test_a_confirmer_that_raises_is_not_a_yes():
    def explode(question):
        raise RuntimeError("no console attached")

    guard = Guard(confirm=explode)
    assert not guard.check("click", {}, element(name="Send"))


# --- the trace --------------------------------------------------------------


def test_every_action_lands_on_its_own_line(tmp_path):
    path = tmp_path / "run.jsonl"
    with trace_module.Trace(path, run_id="t1") as trace:
        trace.task_start("t-01", "uia", "open notepad")
        trace.action("t-01", 1, "click", {"eid": 0}, element=element())
        trace.task_end("t-01", "uia", True, 1, 1.5)

    records = trace_module.read(path)
    assert [r["kind"] for r in records] == ["task_start", "action", "task_end"]
    assert records[1]["element"]["name"] == "Save"
    assert records[2]["ok"] is True


def test_a_blocked_action_is_recorded_as_loudly_as_a_performed_one(tmp_path):
    """A run that scored well because the guard stopped it is not the same
    result as one that was never asked to do anything dangerous."""
    path = tmp_path / "run.jsonl"
    with trace_module.Trace(path) as trace:
        trace.blocked("t-01", 2, "click", {"eid": 3}, "reads 'Delete'")
    records = trace_module.read(path)
    assert records[0]["kind"] == "blocked"


def test_a_truncated_last_line_is_reported_not_swallowed(tmp_path):
    """What a killed run leaves behind. Dropping it quietly would hide that the
    run never finished."""
    path = tmp_path / "run.jsonl"
    path.write_text('{"kind": "task_start"}\n{"kind": "act', encoding="utf-8")
    records = trace_module.read(path)
    assert records[-1]["kind"] == "truncated"


# --- keyboard translation ---------------------------------------------------


def test_literal_text_cannot_turn_into_a_menu_chord():
    """send_keys reads % as alt. Typing "50% off" unescaped opens a menu."""
    assert escape_text("50% off") == "50{%} off"
    assert escape_text("a^b+c") == "a{^}b{+}c"


def test_hotkeys_become_send_keys_syntax():
    assert to_send_keys("ctrl+s") == "^s"
    assert to_send_keys("ctrl+shift+n") == "^+n"
    assert to_send_keys("alt+f4") == "%{F4}"
    assert to_send_keys("enter") == "{ENTER}"


def test_an_unknown_key_raises_rather_than_guessing():
    """A mistyped hotkey in an unattended run lands somewhere nobody sees."""
    with pytest.raises(ValueError):
        to_send_keys("ctrl+frobnicate")
    with pytest.raises(ValueError):
        to_send_keys("hyper+s")


# --- actions ----------------------------------------------------------------


def test_a_dry_run_records_everything_and_touches_nothing(tmp_path):
    path = tmp_path / "run.jsonl"
    with trace_module.Trace(path) as trace:
        hands = Actions(guard=Guard(), trace=trace, task_id="t-01", dry_run=True)
        assert hands.click(element(), step=1).ok
        assert hands.type_text("hello", step=2).ok
    kinds = [r["kind"] for r in trace_module.read(path)]
    assert kinds == ["action", "action"]


def test_the_guard_stops_a_click_before_the_mouse_moves(tmp_path):
    """dry_run is off here, so if the guard let this through it would really
    click. It passes because the click never happens, not because it is fake."""
    path = tmp_path / "run.jsonl"
    with trace_module.Trace(path) as trace:
        hands = Actions(guard=Guard(), trace=trace, task_id="t-01")
        result = hands.click(element(name="Delete"), step=1)
    assert not result.ok and result.blocked
    assert trace_module.read(path)[0]["kind"] == "blocked"


# --- fusion -----------------------------------------------------------------


def test_uia_wins_when_both_backends_see_the_same_button():
    """A box labelled "icon" must never displace a control that knows its own
    name and whether it is enabled."""
    uia = Observation(elements=[element(0, "Save", box=(0, 0, 100, 40))], mode="uia")
    vision = Observation(
        elements=[element(0, "icon", "Detected", (2, 2, 98, 38), "vision")],
        mode="vision",
    )
    fused = fuse(uia, vision)
    assert len(fused.elements) == 1
    assert fused.elements[0].name == "Save"
    assert fused.elements[0].source == "uia"


def test_vision_fills_in_what_uia_never_reported():
    """The custom-drawn surface case, which is the entire reason mode (c)
    exists."""
    uia = Observation(elements=[element(0, "Save", box=(0, 0, 100, 40))], mode="uia")
    vision = Observation(
        elements=[element(0, "icon", "Detected", (500, 500, 560, 560), "vision")],
        mode="vision",
    )
    fused = fuse(uia, vision)
    assert len(fused.elements) == 2
    assert {e.source for e in fused.elements} == {"uia", "vision"}
    # Ids stay unique after the merge, or a planner can address two elements.
    assert len({e.eid for e in fused.elements}) == 2


def test_the_hybrid_is_billed_for_both_passes():
    """Reporting the larger of the two would hide what the hybrid costs, which
    is one of the three numbers Experiment 1 exists to produce."""
    uia = Observation(mode="uia", latency_s=0.2)
    vision = Observation(mode="vision", latency_s=1.8)
    assert fuse(uia, vision).latency_s == pytest.approx(2.0)


# --- the reset, which is the only part of the harness that deletes ----------


def test_a_reset_clears_the_scratch_file_so_a_repeat_pass_means_something(tmp_path):
    """Without this, pass two of Experiment 2 passes its check on pass one's
    leftovers whether or not the agent does anything."""
    from hands.loop import reset

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    victim = scratch / "np-01.txt"
    victim.write_text("stale", encoding="utf-8")

    detail = reset({"kind": "delete_path", "path": str(victim)}, scratch)
    assert not victim.exists()
    assert "removed" in detail


def test_a_reset_outside_the_scratch_root_is_refused(tmp_path):
    """A mistyped path in a task file must not be able to take something real.
    Refused rather than redirected: a silent redirect is a worse surprise."""
    from hands.loop import reset

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    precious = tmp_path / "precious.txt"
    precious.write_text("keep me", encoding="utf-8")

    detail = reset({"kind": "delete_path", "path": str(precious)}, scratch)
    assert precious.exists()
    assert detail.startswith("REFUSED")


def test_a_reset_cannot_escape_upwards(tmp_path):
    from hands.loop import reset

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    precious = tmp_path / "precious.txt"
    precious.write_text("keep me", encoding="utf-8")

    detail = reset({"kind": "delete_path",
                    "path": str(scratch / ".." / "precious.txt")}, scratch)
    assert precious.exists()
    assert detail.startswith("REFUSED")


def test_a_task_with_no_reset_is_left_alone(tmp_path):
    from hands.loop import reset

    assert reset(None, tmp_path) == "no reset"
