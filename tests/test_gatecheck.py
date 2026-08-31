"""The check that the permission gate is still a gate.

A self test that cannot fail is worse than no self test, because it reports
"ok" either way. So most of what is asserted here is that a broken gate is
actually caught, in each of the ways it could break.

The real end-to-end version of this was run against claude 2.1.250: with
`--permission-mode manual` the write is refused and reported, and with the flag
set to `auto` the same write goes through and `verify` returns not-ok. These
tests keep that logic honest without spending a turn.
"""

import tempfile
from pathlib import Path

import pytest

from vesper import gatecheck
from vesper.brain.protocol import BrainError, PermissionNeeded, TurnComplete
from vesper.config import Config

CANARY = Path(tempfile.gettempdir()) / "vesper-gate-check.txt"


class ScriptedBrain:
    """Stands in for the CLI. `writes` means the gate let the write through."""

    def __init__(self, *, events, writes=False):
        self._events = events
        self._writes = writes
        self.stopped = False

    def ask(self, text):
        if self._writes:
            CANARY.write_text("hello", encoding="utf-8")
        yield from self._events

    def stop(self, *args, **kwargs):
        self.stopped = True


@pytest.fixture
def scripted(monkeypatch):
    made = {}

    def install(*, events, writes=False):
        brain = ScriptedBrain(events=events, writes=writes)
        made["brain"] = brain
        monkeypatch.setattr(gatecheck, "ClaudeBrain", lambda *a, **k: brain)
        return brain

    yield install
    if CANARY.exists():
        CANARY.unlink()


DENIED = PermissionNeeded("Write", "canary", tool_input={"file_path": str(CANARY)})
DONE = TurnComplete(text="I could not write that.", cost_usd=0.008)


def test_a_working_gate_passes(scripted):
    scripted(events=[DENIED, DONE])
    result = gatecheck.verify(Config())
    assert result.ok
    assert result.cost_usd == 0.008


def test_a_write_that_goes_through_fails(scripted):
    """The regression this exists for. Measured against the real CLI: with
    permission_mode `auto`, this is exactly what happens."""
    scripted(events=[DONE], writes=True)
    result = gatecheck.verify(Config())
    assert not result.ok
    assert result.wrote_anyway
    assert "went through" in result.detail


def test_a_refusal_nobody_reports_fails(scripted):
    """Nothing was written, so the machine is safe, but Vesper never saw a
    question to ask. That is a Vesper that can never act on anything, and it
    should be loud rather than mysterious."""
    scripted(events=[DONE])
    result = gatecheck.verify(Config())
    assert not result.ok
    assert result.refused and not result.reported
    assert "silently" in result.detail


def test_a_broken_brain_fails_rather_than_passing(scripted):
    scripted(events=[BrainError("the cli would not start")])
    result = gatecheck.verify(Config())
    assert not result.ok
    assert "would not start" in result.detail


def test_the_canary_is_cleaned_up_even_when_the_gate_leaks(scripted):
    scripted(events=[DONE], writes=True)
    gatecheck.verify(Config())
    assert not CANARY.exists(), "left a file behind on the user's machine"


def test_the_child_is_always_stopped(scripted):
    brain = scripted(events=[DENIED, DONE])
    gatecheck.verify(Config())
    assert brain.stopped, "a self check must not leak a claude process"


def test_the_probe_does_not_use_the_real_persona():
    """The persona tells Claude how to speak and what to do about refusals. The
    check is about the CLI underneath it, so it must not be testing the prompt
    by accident."""
    assert "Vesper" not in gatecheck.PROBE_PROMPT
    assert "self test" in gatecheck.PROBE_PROMPT
