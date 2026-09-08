"""Shared fakes.

The design goal is that the entire conversation loop can be exercised on a
machine with no microphone, no speakers, and no network. Everything that touches
hardware or Anthropic has a scripted stand-in here, so the loop's logic (turn
taking, barge-in, echo rejection, channel routing) is tested for real while the
slow and physical parts are not.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vesper.brain.protocol import (
    SessionReady,
    TextDelta,
    ToolStarted,
    TurnComplete,
)
from vesper.stt.whisper import Transcript


class FakeBrain:
    """Replays scripted replies instead of talking to Claude.

    `script` maps nothing: replies are consumed in order. Each reply is a plain
    string, streamed out as small deltas the way the real CLI does, so the
    sentence assembler and channel router are genuinely exercised.
    """

    def __init__(self, replies: list[str] | None = None, *, chunk: int = 4) -> None:
        self.replies = list(replies or [])
        self.chunk = chunk
        self.asked: list[str] = []
        self.notes: list[str] = []
        self.interrupted = threading.Event()
        self.started = False
        self.stopped = False
        self.session_id = "fake-session"
        self.total_cost_usd = 0.0
        self.turn_count = 0
        self.tools_to_report: list[tuple[str, str]] = []
        # Two slots, like the real brain: a one-shot grant handed back when
        # the turn ends, and a standing one that outlives it.
        self._one_shot: tuple[str, ...] = ()
        self.standing: tuple[str, ...] = ()
        self.grant_history: list[tuple[str, ...]] = []
        self.revoked = False
        self.standing_revoked = False
        # What `claude auth status` would say. None means "could not tell".
        self.logged_in = True
        self.restarts = 0

    def start(self, **kwargs):
        self.started = True

    def stop(self, *args, **kwargs):
        self.stopped = True

    def restart(self):
        self.restarts += 1

    def auth_status(self):
        return self.logged_in

    def interrupt(self):
        self.interrupted.set()

    def note(self, text):
        self.notes.append(text)

    # Permission grants. Recorded rather than simulated: what matters to the
    # loop is that a yes widens the allowlist and that the widening is handed
    # back afterwards, both of which are observable here.
    @property
    def grants(self):
        return tuple(dict.fromkeys(self.standing + self._one_shot))

    @grants.setter
    def grants(self, specs):
        self._one_shot = tuple(specs)

    def grant(self, specs, *, standing=False):
        specs = tuple(specs)
        self.grant_history.append(specs)
        if standing:
            self.standing = specs
        else:
            self._one_shot = specs

    def revoke(self):
        self._one_shot = ()

    def revoke_soon(self):
        self.revoked = True
        self.revoke()
        return None

    def revoke_standing(self):
        self.standing_revoked = True
        self.standing = ()

    def ask(self, text):
        self.asked.append(text)
        self.interrupted.clear()
        reply = self.replies.pop(0) if self.replies else "Nothing to report."
        yield SessionReady(session_id=self.session_id, tools=("Bash",))
        for name, detail in self.tools_to_report:
            yield ToolStarted(name, detail)
        for start in range(0, len(reply), self.chunk):
            if self.interrupted.is_set():
                break
            yield TextDelta(reply[start : start + self.chunk])
        self.turn_count += 1
        self.total_cost_usd += 0.008
        yield TurnComplete(
            text=reply,
            session_id=self.session_id,
            cost_usd=0.008,
            duration_ms=1200,
            ttft_ms=400,
            turns=1,
        )


class FakeSTT:
    """Returns queued transcripts, ignoring the audio it is handed."""

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.queue = list(transcripts or [])
        self.calls = 0
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio, sample_rate: int = 16000) -> Transcript:
        self.calls += 1
        if not self.queue:
            return Transcript(text="", rejected_reason="no-text")
        text = self.queue.pop(0)
        if not text:
            return Transcript(text="", rejected_reason="too-quiet")
        return Transcript(
            text=text,
            duration_s=len(audio) / sample_rate if len(audio) else 1.0,
            latency_s=0.01,
            avg_logprob=-0.3,
        )


class FakeVoice:
    """Records what was said instead of making noise."""

    name = "fake"

    def __init__(self, duration: float = 0.0) -> None:
        self.duration = duration
        self.lines: list[str] = []
        self.closed = False

    def speak(self, text: str, stop: threading.Event) -> None:
        self.lines.append(text)
        if self.duration:
            deadline = __import__("time").monotonic() + self.duration
            while __import__("time").monotonic() < deadline:
                if stop.is_set():
                    return
                __import__("time").sleep(0.002)

    def close(self) -> None:
        self.closed = True


class FakeMic:
    """Feeds a scripted sequence of audio blocks."""

    def __init__(self, blocks: list[np.ndarray] | None = None) -> None:
        self.blocks = list(blocks or [])
        self.started = False
        self.stopped = False
        self.drained = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self, timeout: float = 1.0):
        return self.blocks.pop(0) if self.blocks else None

    def preroll(self):
        return np.zeros(0, dtype=np.float32)

    def drain(self):
        self.drained += 1


class RecordingUI:
    """Captures everything the loop would have displayed."""

    def __init__(self) -> None:
        self.heard_lines: list[tuple[str, bool]] = []
        self.screens: list[str] = []
        self.tools: list[tuple[str, str]] = []
        self.permissions: list[tuple[str, str]] = []
        self.answers: list = []
        self.discards: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.interruptions: list[int] = []
        self.states: list[str] = []
        self.decisions: list[tuple[str, str]] = []
        self.notes: list[str] = []

    def heard(self, text, addressed):
        self.heard_lines.append((text, addressed))

    def thinking(self, what):
        self.states.append(what)

    def tool(self, name, detail):
        self.tools.append((name, detail))

    def screen(self, text):
        self.screens.append(text)

    def permission(self, tool, detail):
        self.permissions.append((tool, detail))

    def decision(self, decision, action):
        self.decisions.append((decision, action))

    def answered(self, turn, total_s, first_speech_s):
        self.answers.append(turn)

    def interrupted(self, dropped):
        self.interruptions.append(dropped)

    def discarded(self, reason):
        self.discards.append(reason)

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.notes.append(message)


@pytest.fixture
def fake_brain():
    return FakeBrain()


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def fake_voice():
    return FakeVoice()


@pytest.fixture
def ui():
    return RecordingUI()
