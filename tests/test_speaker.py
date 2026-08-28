"""Speaker queue and barge-in.

These are concurrency tests, so they use a fake voice with controllable timing
rather than real audio. The behaviour under test is ordering and interruption,
not sound.
"""

import threading
import time

import pytest

from vesper.audio.speaker import Speaker
from vesper.tts.base import NullVoice


class FakeVoice:
    """A voice that takes a known amount of time and honours the stop event."""

    name = "fake"

    def __init__(self, duration: float = 0.05) -> None:
        self.duration = duration
        self.started: list[str] = []
        self.finished: list[str] = []
        self.interrupted: list[str] = []
        self.closed = False
        self._entered = threading.Event()

    def speak(self, text: str, stop: threading.Event) -> None:
        self.started.append(text)
        self._entered.set()
        deadline = time.monotonic() + self.duration
        while time.monotonic() < deadline:
            if stop.is_set():
                self.interrupted.append(text)
                return
            time.sleep(0.002)
        self.finished.append(text)

    def wait_until_speaking(self, timeout: float = 2.0) -> bool:
        return self._entered.wait(timeout)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def voice():
    return FakeVoice()


def test_speaks_queued_lines_in_order(voice):
    speaker = Speaker(voice)
    for line in ["one", "two", "three"]:
        speaker.say(line)
    assert speaker.wait_until_idle(timeout=5)
    speaker.close()
    assert voice.finished == ["one", "two", "three"]


def test_say_returns_immediately(voice):
    voice.duration = 0.5
    speaker = Speaker(voice)
    start = time.monotonic()
    speaker.say("a long line")
    elapsed = time.monotonic() - start
    speaker.close()
    assert elapsed < 0.1, "say() must not block the caller"


def test_blank_lines_are_ignored(voice):
    speaker = Speaker(voice)
    speaker.say("")
    speaker.say("   ")
    speaker.say(None)
    assert speaker.wait_until_idle(timeout=2)
    speaker.close()
    assert voice.started == []


def test_barge_in_cuts_off_the_current_line(voice):
    voice.duration = 2.0
    speaker = Speaker(voice)
    speaker.say("a very long sentence")
    assert voice.wait_until_speaking()

    start = time.monotonic()
    speaker.barge_in()
    speaker.wait_until_idle(timeout=2)
    elapsed = time.monotonic() - start

    speaker.close()
    assert voice.interrupted == ["a very long sentence"]
    assert voice.finished == []
    assert elapsed < 0.5, "interruption must be near-immediate"


def test_barge_in_drops_the_whole_queue(voice):
    voice.duration = 1.0
    speaker = Speaker(voice)
    for line in ["first", "second", "third", "fourth"]:
        speaker.say(line)
    assert voice.wait_until_speaking()

    dropped = speaker.barge_in()
    speaker.wait_until_idle(timeout=3)
    speaker.close()

    assert dropped == 3, "three lines were still queued behind the one playing"
    assert voice.finished == []
    assert "second" not in voice.started


def test_barge_in_reports_zero_when_nothing_is_queued(voice):
    speaker = Speaker(voice)
    assert speaker.barge_in() == 0
    speaker.close()


def test_speaker_recovers_and_accepts_new_lines_after_barge_in(voice):
    voice.duration = 1.0
    speaker = Speaker(voice)
    speaker.say("interrupt me")
    assert voice.wait_until_speaking()
    speaker.barge_in()

    voice.duration = 0.02
    speaker.say("the reply")
    assert speaker.wait_until_idle(timeout=5)
    speaker.close()
    assert voice.finished == ["the reply"]


def test_line_queued_before_barge_in_never_speaks(voice):
    """The race the generation counter exists to close.

    A line dequeued by the worker but not yet started must still be dropped if a
    barge-in lands in that gap.
    """
    voice.duration = 0.3
    speaker = Speaker(voice)
    speaker.say("playing")
    assert voice.wait_until_speaking()
    speaker.say("queued behind it")
    speaker.barge_in()
    speaker.wait_until_idle(timeout=3)
    speaker.close()
    assert "queued behind it" not in voice.started


def test_callbacks_fire_for_completed_lines(voice):
    started, done = [], []
    speaker = Speaker(voice, on_start=started.append, on_done=done.append)
    speaker.say("hello")
    assert speaker.wait_until_idle(timeout=5)
    speaker.close()
    assert started == ["hello"]
    assert done == ["hello"]


def test_done_callback_does_not_fire_for_an_interrupted_line(voice):
    voice.duration = 2.0
    done = []
    speaker = Speaker(voice, on_done=done.append)
    speaker.say("cut me off")
    assert voice.wait_until_speaking()
    speaker.barge_in()
    speaker.wait_until_idle(timeout=3)
    speaker.close()
    assert done == []


def test_a_failing_voice_does_not_kill_the_worker():
    class BrokenVoice:
        name = "broken"

        def __init__(self):
            self.calls = 0

        def speak(self, text, stop):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("audio device vanished")

        def close(self):
            pass

    errors = []
    broken = BrokenVoice()
    speaker = Speaker(broken, on_error=errors.append)
    speaker.say("this one explodes")
    speaker.say("this one must still be attempted")
    assert speaker.wait_until_idle(timeout=5)
    speaker.close()

    assert len(errors) == 1
    assert broken.calls == 2, "the worker thread must survive a backend failure"


def test_speaking_flag_tracks_playback(voice):
    voice.duration = 0.4
    speaker = Speaker(voice)
    assert speaker.speaking is False
    speaker.say("something")
    assert voice.wait_until_speaking()
    assert speaker.speaking is True
    speaker.wait_until_idle(timeout=3)
    assert speaker.speaking is False
    speaker.close()


def test_close_shuts_down_the_backend():
    voice = FakeVoice()
    speaker = Speaker(voice)
    speaker.say("bye")
    speaker.wait_until_idle(timeout=3)
    speaker.close()
    assert voice.closed is True


def test_null_voice_satisfies_the_interface():
    voice = NullVoice()
    speaker = Speaker(voice)
    speaker.say("silent")
    assert speaker.wait_until_idle(timeout=2)
    speaker.close()
    assert voice.spoken == ["silent"]
