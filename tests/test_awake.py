"""Waking on the name, staying awake briefly, then going back to sleep.

Vesper used to need his name on every single utterance. The follow-up window
that would have fixed that existed in `wake.py` from the start and was switched
off in the config, and it was also wired to the wrong clock: it counted from
the moment *he* finished thinking, so with half duplex muting the microphone
for the whole answer, a long reply spent most of the window before you could
say anything into it.

These cover the three parts of the fix: the clock runs from your speech, the
state is visible, and approving a change to the machine still costs you his
name even while he is awake.

The gate takes `now` as a parameter and owns no clock, so the timeline is
written out as plain floats. Nothing here sleeps.
"""

from __future__ import annotations

import time

import numpy as np

from vesper.audio.speaker import Speaker
from vesper.conversation import Conversation, ConversationConfig
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI


def _audio(seconds: float = 0.5) -> np.ndarray:
    return np.zeros(int(16_000 * seconds), dtype=np.float32)


def _conversation(transcripts=None, *, window_s=25.0, replies=None):
    speaker = Speaker(FakeVoice(duration=0.0))
    ui = RecordingUI()
    conversation = Conversation(
        brain=FakeBrain(replies or ["Right."]),
        stt=FakeSTT(transcripts or []),
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig(follow_up_window_s=window_s)),
        config=ConversationConfig(greet_on_start=False),
        ui=ui,
    )
    return conversation, speaker, ui


# --- the clock runs from your speech ----------------------------------------


def test_the_window_is_reset_by_every_thing_you_say():
    """"Goes to sleep if I don't speak" is the whole request. A window that
    only ever started when Vesper replied would sleep on you mid-conversation.

    Mutation that fails this: delete the `engage` from `_on_utterance`.
    """
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.engage(now=100.0)

    # Twenty seconds later, still awake, and saying something pushes it out.
    assert gate.check("and what about memory", now=120.0).triggered
    gate.engage(now=120.0)

    assert gate.check("and the disk", now=140.0).triggered, "it slept mid sentence"


def test_silence_puts_it_back_to_sleep():
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.engage(now=100.0)
    assert not gate.check("and what about memory", now=126.0).triggered


def test_speaking_to_him_wakes_him(monkeypatch):
    conversation, speaker, _ = _conversation(["Vesper what time is it"])
    monkeypatch.setattr(conversation, "respond", lambda text: None)
    try:
        assert conversation.status()["awake"] is False
        conversation._on_utterance(_audio())
        assert conversation.status()["awake"] is True
    finally:
        speaker.close()


def test_a_voice_that_is_not_yours_cannot_hold_the_window_open(monkeypatch):
    """The window is opened after the voiceprint check, not before it, so
    someone else saying his name across the room cannot keep the microphone
    live for plain speech.

    Mutation that fails this: move the `engage` above `_is_the_right_voice`.
    """
    conversation, speaker, _ = _conversation(["Vesper what time is it"])
    monkeypatch.setattr(conversation, "_is_the_right_voice", lambda audio: False)
    try:
        conversation._on_utterance(_audio())
        assert conversation.status()["awake"] is False
    finally:
        speaker.close()


# --- and again when he stops talking ----------------------------------------


class Talking:
    """A speaker whose `speaking` flag the test drives by hand.

    The real one would need the test to sleep for the length of an utterance,
    and a suite that sleeps is a suite that goes flaky on a loaded machine.
    """

    def __init__(self):
        import types

        self.speaking = False
        self.voice = types.SimpleNamespace(name="fake")
        self.lines = []

    def say(self, line):
        self.lines.append(line)

    def barge_in(self):
        return 0

    def close(self):
        pass


def _finishes_speaking(conversation):
    """Drive one block while he talks, and one after he has stopped."""
    block = np.zeros(480, dtype=np.float32)
    conversation.speaker.speaking = True
    conversation._handle_block(block)
    conversation.speaker.speaking = False
    # The tail is what stops the microphone hearing his last syllable. Rewind
    # past it the way the rest of the suite does rather than waiting for it.
    conversation._spoke_until -= 1.0
    conversation._handle_block(block)


def test_the_window_starts_again_when_he_finishes_answering():
    """With half duplex the microphone is deaf for the whole answer, so a
    window that started when the turn completed spent fifteen of its
    twenty-five seconds on a fifteen second reply.

    Mutation that fails this: delete the `_was_speaking` branch in
    `_handle_block`.
    """
    conversation, speaker, _ = _conversation()
    speaker.close()
    conversation.speaker = Talking()
    conversation._in_exchange = True

    _finishes_speaking(conversation)

    assert conversation.status()["awake"] is True


def test_saying_hello_to_an_empty_room_does_not_open_the_window():
    """The startup greeting goes through the same speaker. An assistant that
    left the microphone live for plain speech because it greeted a room nobody
    was in is the accident the wake word exists to prevent.

    Mutation that fails this: drop the `_in_exchange` guard.
    """
    conversation, speaker, _ = _conversation()
    speaker.close()
    conversation.speaker = Talking()

    _finishes_speaking(conversation)

    assert conversation.status()["awake"] is False


def test_a_remark_he_started_can_be_answered_without_his_name():
    """He volunteers "your battery is at nine percent"; you should be able to
    say "plug it in" rather than "Vesper, plug it in"."""
    conversation, speaker, _ = _conversation()
    speaker.close()
    conversation.speaker = Talking()

    conversation.volunteer("Your battery is at nine percent.")
    _finishes_speaking(conversation)

    assert conversation.speaker.lines == ["Your battery is at nine percent."]
    assert conversation.status()["awake"] is True


# --- approving still costs you his name -------------------------------------


class Request:
    """Enough of an ActionRequest for the answer paths to be exercised."""

    approvable = True

    def written(self):
        return "Write: C:/notes.txt"

    def spoken(self):
        return "create notes dot txt"

    def grants(self):
        return ("Write",)


def test_a_bare_yes_inside_the_window_approves_nothing():
    """The window makes plain speech count as talking to him, which is how
    people actually reply. Approving a change to the machine is the one thing
    it deliberately does not cover: a "yes" from the television, or from
    someone in the room agreeing with something else entirely, must never be
    able to grant anything.

    Mutation that fails this: drop the `named` check from `_settle_consent`.
    """
    conversation, speaker, _ = _conversation()
    conversation._pending = Request()

    handled = conversation._settle_consent("yes", echo=False, named=False)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    assert handled is True, "it fell through and was sent to Claude as a question"
    assert conversation._pending is not None, "the request was thrown away"
    assert conversation.approvals == 0
    assert conversation.speaker.voice.lines[-1] == "I'll need my name on that one."


def test_a_bare_no_inside_the_window_still_refuses():
    """Deliberately not symmetric. Making someone say a name before they are
    allowed to stop something is the wrong way round, and refusing cannot
    cause harm.

    Mutation that fails this: gate the whole answer on `named` rather than
    only the yes.
    """
    conversation, speaker, _ = _conversation()
    conversation._pending = Request()

    handled = conversation._settle_consent("no", echo=False, named=False)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    assert handled is True
    assert conversation.refusals == 1
    assert conversation._pending is None


def test_typed_input_never_has_to_say_the_name():
    """`hear()` is the door for typed input, and someone at the keyboard is a
    stronger identity check than any voiceprint."""
    import inspect

    from vesper.conversation import Conversation

    source = inspect.getsource(Conversation.hear)
    assert "named=" not in source, "typed input was made to say his name"
    assert "_settle_consent(text, echo=False)" in source


# --- and the question goes to sleep with him --------------------------------


def test_the_pending_question_lapses_when_he_goes_back_to_sleep():
    """A question nobody answered for twenty-five seconds is not one you still
    want answered by the next thing said in the room.

    Mutation that fails this: drop the `_lapse_consent` from `_wake_tick`.
    """
    conversation, speaker, ui = _conversation()
    speaker.close()
    conversation._pending = Request()
    conversation.wake.engage(time.monotonic())
    conversation._wake_tick()
    assert conversation._pending is not None, "it lapsed while he was still awake"

    conversation.wake.disengage()
    conversation._wake_tick()

    assert conversation._pending is None
    assert ("ignored", "Write: C:/notes.txt") in ui.decisions


def test_going_to_sleep_is_noticed_once_rather_than_every_block():
    """`_wake_tick` runs thirty times a second. Anything it does on a change
    has to happen on the change, not on every call after it."""
    conversation, speaker, ui = _conversation()
    speaker.close()
    conversation.wake.engage(time.monotonic())
    conversation._wake_tick()
    conversation.wake.disengage()

    for _ in range(5):
        conversation._wake_tick()

    assert conversation._in_exchange is False


# --- and you can see which he is --------------------------------------------


class RecordingTray:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def _updates(tray, count, timeout=5.0):
    """Wait for the pushes, which happen on their own short lived threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(tray.updates) < count:
        time.sleep(0.01)
    return tray.updates


def test_the_tray_is_told_the_moment_it_changes():
    """Expiry is not an event anywhere: `engaged()` simply starts returning
    False. The audio loop is what looks, because it is already running."""
    conversation, speaker, _ = _conversation()
    speaker.close()
    conversation.tray = RecordingTray()

    conversation.wake.engage(time.monotonic())
    conversation._wake_tick()
    conversation.wake.disengage()
    conversation._wake_tick()

    assert _updates(conversation.tray, 2) == [{"awake": True}, {"awake": False}]


def test_telling_the_tray_never_blocks_the_audio_loop():
    """`Shell_NotifyIcon` round trips to Explorer and blocks for seconds when
    Explorer is hung. A stalled microphone read is dropped blocks, and dropped
    blocks are a wake word nobody heard.

    Mutation that fails this: call `self.tray.update(...)` inline.
    """
    class Hung:
        def update(self, **kwargs):
            time.sleep(30)

    conversation, speaker, _ = _conversation()
    speaker.close()
    conversation.tray = Hung()
    conversation.wake.engage(time.monotonic())

    started = time.monotonic()
    conversation._wake_tick()

    assert time.monotonic() - started < 1.0, "the audio loop waited for the shell"


def test_the_awake_state_is_a_scalar_like_everything_else_in_the_snapshot():
    """The dashboard reads this across a thread boundary once a second."""
    conversation, speaker, _ = _conversation()
    speaker.close()
    assert isinstance(conversation.status()["awake"], bool)


class Field:
    def __init__(self):
        self.text = ""
        self.fg = ""

    def config(self, **kwargs):
        self.text = kwargs.get("text", self.text)
        self.fg = kwargs.get("fg", self.fg)


def _dashboard(data):
    from vesper.ui import dashboard as dashboard_module

    board = dashboard_module.Dashboard(lambda: data)
    for name in ("state", "pause", "uptime", "detail"):
        board._fields[name] = Field()
    board._apply(data)
    return board._fields["state"], dashboard_module


def test_the_dashboard_shows_three_states_not_two():
    asleep, module = _dashboard({"paused": False, "awake": False})
    awake, _ = _dashboard({"paused": False, "awake": True})
    paused, _ = _dashboard({"paused": True, "awake": True})

    assert (asleep.text, awake.text, paused.text) == ("asleep", "awake", "paused")
    # Asleep is the resting state and keeps the warm star. Awake gets the cool
    # one, the same colour the tray uses, so the two surfaces agree.
    assert asleep.fg == module.STAR
    assert awake.fg == module.COOL
    assert paused.fg == module.MUTED, "paused must win over awake"


def test_the_tray_star_changes_colour_for_awake():
    """The third icon has been generated into var/icons since the icon set was
    written and loaded by nothing. This is what finally uses it."""
    from vesper.ui.tray import TrayIcon

    tray = TrayIcon(name="Vesper", icons={})
    assert tray.icon_state() == "listening"

    tray.state.awake = True
    assert tray.icon_state() == "working"

    tray.state.listening = False
    assert tray.icon_state() == "paused", "paused must win over awake"


def test_the_answer_to_the_nag_is_not_swallowed_as_an_echo():
    """The sentence asking for his name cannot contain the words the answer
    will be made of.

    `_is_own_voice` discards anything overlapping 60% or more with what Vesper
    just said. "Vesper, yes" is two unique words, so a nag containing "vesper"
    and "yes" scores 100% and throws away the very reply it asked for. The
    first version of this said "Say 'Vesper, yes' and I will" and did exactly
    that.

    Mutation that fails this: put either word back in the sentence.
    """
    conversation, speaker, _ = _conversation()
    conversation._pending = Request()
    conversation._settle_consent("yes", echo=False, named=False)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    # Every way of agreeing that `consent.py` accepts, since any of them can
    # be what the user says next.
    answers = (
        "Vesper yes", "Vesper, yes please", "Vesper do it", "Vesper sure",
        "Vesper okay", "Vesper go ahead", "Vesper yes go ahead",
    )
    for answer in answers:
        assert not conversation._is_own_voice(answer), (
            f"the nag would swallow {answer!r}: "
            f"{conversation.speaker.voice.lines[-1]!r}"
        )


# --- the two things that still cost you his name ----------------------------


def test_an_overheard_quit_cannot_end_him():
    """"Exit" and "quit" are ordinary English, and with the window open they
    arrive here from anything said in the room within twenty five seconds of
    talking to him. Ending the assistant is the second thing, after approving
    a change, that is worth making you say his name for.

    Mutation that fails this: drop the `named` guard from the shutdown branch.
    """
    conversation, speaker, _ = _conversation()
    conversation._running.set()

    handled = conversation._handle_local("quit", named=False)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    assert handled is True, "it fell through and was sent to Claude"
    assert conversation.running, "an unnamed word in the room ended him"
    assert conversation.speaker.voice.lines[-1] == "I'll need my name on that one."


def test_naming_him_still_ends_him():
    conversation, speaker, _ = _conversation()
    conversation._running.set()

    conversation._handle_local("shut down", named=True)
    speaker.close()

    assert not conversation.running


def test_pausing_puts_him_to_sleep():
    """Otherwise a window opened before you paused is still counting behind an
    icon that says paused."""
    conversation, speaker, _ = _conversation()
    conversation.wake.engage(time.monotonic())
    conversation.pause(True)
    speaker.close()

    assert conversation.status()["awake"] is False


def test_open_mic_reports_itself_as_awake():
    """With the wake word turned off entirely there is no window, and
    everything said is acted on. A panel reading `engaged` would call that
    asleep, which is the opposite of the truth."""
    from vesper.wake import WakeConfig, WakeGate

    gate = WakeGate(WakeConfig(require_wake_word=False))
    assert gate.engaged(now=0.0) is False
    assert gate.awake(now=0.0) is True
