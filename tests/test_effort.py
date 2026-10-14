"""How hard the brain thinks, chosen per question.

Measured 2026-10-14 on CLI 2.1.299 before any of this was written: the
`apply_flag_settings` control request changes effort inside a running session
and answers `success`. The same digit-sum question cost 122 output tokens at
low and 596 at max in one process, and the next turn after switching back to
low took about a second. So the switch is real, and it needs no respawn.
"""

import json

from vesper.brain.claude import BrainConfig, ClaudeBrain
from vesper.brain.effort import choose_effort, effort_request


# --- the choice ------------------------------------------------------------


def test_a_quick_question_thinks_lightly():
    assert choose_effort("what time is it", default="") == "low"
    assert choose_effort("open chrome", default="high") == "low"


def test_a_question_that_needs_reasoning_thinks_hard():
    assert choose_effort("why does my laptop get hot when I game", default="") == "high"
    assert choose_effort("help me plan my week around the two exams", default="") == "high"
    assert choose_effort("compare these two job offers", default="") == "high"


def test_asking_for_care_out_loud_gets_the_most():
    assert choose_effort("think hard about this, is the proof right", default="") == "max"
    assert choose_effort("take your time and figure out the bug", default="") == "max"


def test_a_long_request_counts_as_hard_even_without_a_cue():
    long = " ".join(["word"] * 30)
    assert choose_effort(long, default="") == "high"


def test_anything_else_keeps_the_configured_default():
    assert choose_effort("tell me about the meeting tomorrow afternoon please", default="medium") == "medium"
    assert choose_effort("tell me about the meeting tomorrow afternoon please", default="") == ""


def test_punctuation_and_case_do_not_matter():
    assert choose_effort("WHY, though?", default="") == "high"


def test_a_cue_must_be_a_whole_word():
    """"planet" is not "plan", "whyte" is not "why"."""
    assert choose_effort("name a planet", default="") == "low"


# --- the wire format ---------------------------------------------------------


def test_the_request_is_the_measured_control_frame():
    frame = json.loads(effort_request("high", "e1"))
    assert frame == {
        "type": "control_request",
        "request_id": "e1",
        "request": {"subtype": "apply_flag_settings", "settings": {"effortLevel": "high"}},
    }


def test_blank_means_back_to_the_cli_default():
    frame = json.loads(effort_request("", "e2"))
    assert frame["request"]["settings"] == {"effortLevel": None}


# --- the brain sends it only when it changes ---------------------------------


class _Pipe:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def flush(self):
        pass


def _brain(**kwargs):
    config = BrainConfig(adaptive_effort=True, **kwargs)
    brain = ClaudeBrain(config)
    brain._effort_now = config.effort_level()
    return brain


def _sent(pipe):
    return [json.loads(line) for line in pipe.lines]


def test_the_brain_switches_effort_before_the_question():
    brain, pipe = _brain(), _Pipe()
    brain.set_effort_for("what time is it")
    brain._apply_effort(pipe)
    assert _sent(pipe)[0]["request"]["settings"] == {"effortLevel": "low"}


def test_no_frame_when_the_level_is_already_right():
    brain, pipe = _brain(), _Pipe()
    brain.set_effort_for("what time is it")
    brain._apply_effort(pipe)
    brain.set_effort_for("open notepad")
    brain._apply_effort(pipe)
    assert len(pipe.lines) == 1


def test_off_means_never_a_frame():
    brain = ClaudeBrain(BrainConfig(adaptive_effort=False))
    pipe = _Pipe()
    brain.set_effort_for("what time is it")
    brain._apply_effort(pipe)
    assert pipe.lines == []


def test_a_local_brain_is_never_sent_effort():
    brain, pipe = _brain(local=True), _Pipe()
    brain.set_effort_for("why is the sky blue")
    brain._apply_effort(pipe)
    assert pipe.lines == []


def test_a_note_does_not_inherit_the_last_questions_effort_choice():
    """`note` and the proactive remarks go through `ask` too. The level wanted
    for a question is spent on that question and not carried to the next ask."""
    brain, pipe = _brain(), _Pipe()
    brain.set_effort_for("think hard about this")
    brain._apply_effort(pipe)
    brain._apply_effort(pipe)
    assert len(pipe.lines) == 1


def test_a_respawn_starts_from_the_configured_level_again():
    brain = _brain(effort="medium")
    brain._effort_now = "max"
    brain._reset_effort()
    assert brain._effort_now == "medium"
