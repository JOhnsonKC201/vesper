"""The ambient loop.

Almost every test here asserts that Vesper stayed quiet. That ratio is the
point: an assistant that speaks up unprompted is one bad gate away from being
unbearable, so the gates get more coverage than the speaking does.
"""

import time
from datetime import datetime
from unittest import mock

import pytest

from vesper.brain.protocol import TextDelta, TurnComplete
from vesper.proactive import ProactiveConfig, ProactiveLoop
from vesper.sensors.snapshot import Snapshot
from vesper.sensors.system import Vitals
from vesper.sensors.window import ActiveWindow


class ScriptedBrain:
    """Answers proactive consultations with whatever it is told to."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.prompts = []
        self.busy = False

    def ask(self, text):
        self.prompts.append(text)
        answer = self.answers.pop(0) if self.answers else "SILENT"
        yield TextDelta(answer)
        yield TurnComplete(text=answer, turns=1)


def snapshot(*, cpu=10.0, idle=5.0, battery=80, ac=True, window="Code.exe", at=1000.0):
    return Snapshot(
        at=at,
        window=ActiveWindow(title="main.py", process=window),
        vitals=Vitals(
            cpu_percent=cpu,
            ram_percent=50.0,
            disk_free_gb=500.0,
            battery_percent=battery,
            on_ac_power=ac,
            top_processes=(("python.exe", cpu),),
        ),
        idle_s=idle,
    )


def cfg(**kwargs):
    """Test config with quiet hours off by default.

    Without this the suite passes or fails depending on the wall clock, which is
    exactly the kind of test that erodes trust in a suite. Quiet hours get their
    own test with a frozen clock.
    """
    kwargs.setdefault("quiet_hours", "")
    return ProactiveConfig(**kwargs)


def build(answers=None, config=None, now=2000.0):
    spoken = []
    brain = ScriptedBrain(answers)
    loop = ProactiveLoop(
        brain=brain,
        speak=spoken.append,
        config=config or cfg(min_interval_s=900),
        now=lambda: now,
    )
    return loop, brain, spoken


def drive(loop, previous, current):
    """Run one tick with controlled before/after snapshots."""
    loop._previous = previous
    with mock.patch("vesper.sensors.snapshot.take", return_value=current):
        return loop.tick()


# --- it speaks when it should -----------------------------------------------


def test_speaks_when_claude_says_something_is_worth_saying():
    loop, brain, spoken = build(answers=["Your battery just dropped to nine percent."])
    said = drive(loop, snapshot(battery=30, ac=False), snapshot(battery=9, ac=False))
    assert said
    assert spoken == ["Your battery just dropped to nine percent."]
    assert loop.remarks == 1


def test_the_change_is_described_to_claude():
    loop, brain, _ = build(answers=["Battery is low."])
    drive(loop, snapshot(battery=30, ac=False), snapshot(battery=9, ac=False))
    prompt = brain.prompts[0]
    assert "what changed" in prompt
    assert "battery" in prompt.lower()
    assert "SILENT" in prompt, "the model must be told how to decline"


# --- it stays quiet ---------------------------------------------------------


def test_silent_answer_produces_no_speech():
    loop, _, spoken = build(answers=["SILENT"])
    assert drive(loop, snapshot(cpu=10), snapshot(cpu=95)) == ""
    assert spoken == []


def test_a_rambling_answer_is_treated_as_a_refusal():
    """A model that ignores the format must not get to monologue at you."""
    loop, _, spoken = build(answers=["Well, " + "there is a lot to say here. " * 20])
    assert drive(loop, snapshot(cpu=10), snapshot(cpu=95)) == ""
    assert spoken == []


def test_nothing_changed_means_claude_is_never_asked():
    loop, brain, spoken = build(answers=["Something!"])
    identical = snapshot()
    assert drive(loop, identical, identical) == ""
    assert brain.prompts == [], "must not spend a turn when there is no news"


def test_muted_never_speaks():
    loop, brain, spoken = build(answers=["Important!"])
    loop.mute()
    assert drive(loop, snapshot(battery=30, ac=False), snapshot(battery=5, ac=False)) == ""
    assert brain.prompts == []
    assert spoken == []


def test_unmute_does_not_immediately_dump_a_backlog():
    loop, _, spoken = build(answers=["Important!"], now=2000.0)
    loop.mute()
    loop.unmute()
    assert drive(loop, snapshot(battery=30, ac=False), snapshot(battery=5, ac=False)) == ""
    assert spoken == []


def test_rate_limit_blocks_a_second_remark():
    loop, _, spoken = build(
        answers=["First thing.", "Second thing."],
        config=cfg(min_interval_s=900),
    )
    drive(loop, snapshot(battery=30, ac=False), snapshot(battery=9, ac=False))
    drive(loop, snapshot(cpu=10), snapshot(cpu=99))
    assert len(spoken) == 1


def test_rate_limit_expires():
    spoken = []
    clock = {"t": 2000.0}
    brain = ScriptedBrain(["First thing.", "Second thing."])
    loop = ProactiveLoop(
        brain=brain,
        speak=spoken.append,
        config=cfg(min_interval_s=900),
        now=lambda: clock["t"],
    )
    drive(loop, snapshot(battery=30, ac=False), snapshot(battery=9, ac=False))
    clock["t"] += 1000
    drive(loop, snapshot(cpu=10), snapshot(cpu=99))
    assert len(spoken) == 2


def test_silent_while_the_user_is_away():
    loop, brain, spoken = build(
        answers=["Hello?"], config=cfg(min_interval_s=0, max_idle_s=900)
    )
    assert drive(loop, snapshot(cpu=10), snapshot(cpu=99, idle=3600)) == ""
    assert brain.prompts == []


def test_silent_during_a_conversation():
    loop, brain, spoken = build(answers=["Hi!"], config=cfg(min_interval_s=0))
    brain.busy = True
    assert drive(loop, snapshot(cpu=10), snapshot(cpu=99)) == ""
    assert brain.prompts == []


def test_silent_while_a_permission_is_standing():
    """The hands can stand for a whole session now. An unattended remark that
    ran with them could move the mouse while nobody asked, so while any grant
    is live the loop does not consult Claude at all."""
    loop, brain, spoken = build(answers=["Hi!"], config=cfg(min_interval_s=0))
    brain.grants = ("Bash(vasper click:*)",)
    assert drive(loop, snapshot(cpu=10), snapshot(cpu=99)) == ""
    assert brain.prompts == []


def test_silent_during_quiet_hours():
    loop, brain, _ = build(
        answers=["Wake up!"],
        config=ProactiveConfig(min_interval_s=0, quiet_hours="23:30-08:00"),
    )
    three_am = datetime(2026, 8, 28, 3, 0)
    with mock.patch("vesper.proactive.datetime") as fake:
        fake.now.return_value = three_am
        assert drive(loop, snapshot(cpu=10), snapshot(cpu=99)) == ""
    assert brain.prompts == []


def test_it_will_not_repeat_itself():
    loop, brain, spoken = build(
        answers=["Battery is low.", "Battery is low."],
        config=cfg(min_interval_s=0),
    )
    drive(loop, snapshot(battery=30, ac=False), snapshot(battery=9, ac=False))
    drive(loop, snapshot(cpu=10), snapshot(cpu=99))
    assert "do not repeat them" in brain.prompts[1]
    assert "Battery is low." in brain.prompts[1]


def test_a_failing_check_does_not_crash_the_loop():
    class ExplodingBrain(ScriptedBrain):
        def ask(self, text):
            raise RuntimeError("brain is gone")
            yield  # pragma: no cover

    loop = ProactiveLoop(
        brain=ExplodingBrain(),
        speak=lambda text: None,
        config=cfg(min_interval_s=0),
    )
    with pytest.raises(RuntimeError):
        drive(loop, snapshot(cpu=10), snapshot(cpu=99))
    # _run swallows it; tick() is allowed to raise so tests can see the cause.


def test_disabled_config_starts_muted():
    loop = ProactiveLoop(
        brain=ScriptedBrain(), speak=lambda t: None, config=ProactiveConfig(enabled=False)
    )
    assert loop.muted is True


def test_start_and_stop_are_safe_to_call_repeatedly():
    loop = ProactiveLoop(
        brain=ScriptedBrain(),
        speak=lambda t: None,
        config=cfg(check_interval_s=60),
    )
    loop.start()
    loop.start()
    loop.stop()
    loop.stop()
