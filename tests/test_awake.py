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
from vesper.brain.consent import ActionRequest
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


# --- but a question he is asking holds the window open ----------------------


def _write_request() -> ActionRequest:
    return ActionRequest(tool="Write", tool_input={"file_path": "C:/notes.txt"})


def test_a_question_asked_late_in_the_window_is_still_answerable():
    """On 2026-09-07 at 10:45:45 "extend my screen" opened the window. The turn
    took nineteen seconds, the question was asked at 10:46:04, and at 10:46:10,
    exactly twenty-five seconds after the utterance, the window closed and the
    question was logged as ignored while it was still being spoken. The answer
    at 10:46:20 landed on nothing.

    Mutation that fails this: drop the `hold_open` from `_ask_consent`.
    """
    conversation, speaker, ui = _conversation()
    now = time.monotonic()
    conversation.wake.engage(now - 19.0)
    conversation._awake = True
    conversation._in_exchange = True

    conversation._ask_consent(_write_request())
    speaker.wait_until_idle(timeout=5.0)

    assert conversation.wake.engaged(now + 7.0), "the old window would have closed at six"
    assert conversation._in_exchange is True

    conversation._wake_tick()
    assert conversation._pending is not None, "the question lapsed while being asked"
    assert not [d for d in ui.decisions if d[0] == "ignored"]

    conversation._settle_consent("yes", echo=False, named=True)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()
    assert conversation.approvals == 1


def test_the_held_window_still_closes_after_the_consent_window():
    """The hold is sized to the consent window, not forever."""
    conversation, speaker, ui = _conversation()
    conversation.wake.engage(time.monotonic())
    conversation._ask_consent(_write_request())
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()
    conversation._wake_tick()
    assert conversation._awake is True

    conversation.wake._engaged_until = time.monotonic() - 0.1
    conversation._wake_tick()

    assert conversation._pending is None
    assert ("ignored", "Write: C:/notes.txt") in ui.decisions


def test_a_question_from_claude_keeps_the_window_open_longer():
    """When he asks which one you meant, you need longer than a follow-up to
    answer, and the answer must not need his name.

    Mutation that fails this: drop the question hold from `_run_turn`.
    """
    conversation, speaker, _ = _conversation(
        ["Vesper delete the file"],
        replies=["Two Delete buttons here. The toolbar one or the menu one?"],
    )
    conversation._on_utterance(_audio())
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()
    assert conversation.wake.engaged(time.monotonic() + 30.0)
    assert conversation._in_exchange is True


def test_a_plain_answer_from_claude_keeps_the_ordinary_window():
    conversation, speaker, _ = _conversation(
        ["Vesper delete the file"], replies=["Right."]
    )
    conversation._on_utterance(_audio())
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()
    assert not conversation.wake.engaged(time.monotonic() + 30.0)


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


class Frame:
    """Enough of a tk.Frame for the pending banner, which packs and unpacks."""

    def __init__(self):
        self.mapped = False
        self.before = None

    def winfo_ismapped(self):
        return self.mapped

    def pack(self, **kwargs):
        self.mapped = True
        self.before = kwargs.get("before")

    def pack_forget(self):
        self.mapped = False


class Log:
    """Enough of a tk.Text to record what would have been written."""

    def __init__(self):
        self.lines: list[tuple[str, str]] = []
        self.state = "disabled"

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)

    def insert(self, _where, text, tag=""):
        assert self.state == "normal", "a read only widget was written to"
        self.lines.append((tag, text))

    def delete(self, *_args):
        self.lines.clear()

    def see(self, _where):
        pass


def _dashboard(data, transcript=None):
    from vesper.ui import dashboard as dashboard_module

    board = dashboard_module.Dashboard(lambda: data, transcript=transcript)
    for name in ("state", "dot", "pause", "uptime", "hint", "ask",
                 "summary", "care", "after_ask"):
        board._fields[name] = Field()
    board._fields["ask_frame"] = Frame()
    board._fields["log"] = Log()
    board._apply(data)
    return board, dashboard_module


def test_the_dashboard_names_four_states_not_three():
    """"waiting on you" was folded into "awake", which is the one state that
    needs something from you and so the one worth telling apart."""
    asleep, module = _dashboard({"paused": False, "awake": False})
    listening, _ = _dashboard({"paused": False, "awake": True})
    waiting, _ = _dashboard({"paused": False, "awake": True, "pending": "edit notes"})
    paused, _ = _dashboard({"paused": True, "awake": True, "pending": "edit notes"})

    assert asleep._fields["state"].text == "asleep"
    assert listening._fields["state"].text == "listening"
    assert waiting._fields["state"].text == "waiting on you"
    assert paused._fields["state"].text == "paused", "paused wins over everything"

    assert asleep._fields["state"].fg == module.STAR
    assert listening._fields["state"].fg == module.COOL
    assert waiting._fields["state"].fg == module.WARN
    assert paused._fields["state"].fg == module.MUTED
    # The dot and the word are the same colour, so it reads from a distance.
    assert asleep._fields["dot"].fg == asleep._fields["state"].fg


def test_every_state_says_what_to_do_about_it():
    """A state nobody knows how to act on is decoration."""
    for data in ({"awake": False}, {"awake": True},
                 {"awake": True, "pending": "x"}, {"paused": True}):
        board, _ = _dashboard(data)
        assert board._fields["hint"].text.strip(), data


def test_the_question_appears_only_while_one_is_pending():
    quiet, _ = _dashboard({"awake": True})
    assert quiet._fields["ask_frame"].mapped is False

    asking, _ = _dashboard({"awake": True, "pending": "edit notes dot txt"})
    assert asking._fields["ask_frame"].mapped is True
    assert "edit notes dot txt" in asking._fields["ask"].text
    # Packed before the conversation, not appended. Without an anchor it lands
    # underneath the Quit button, which is the last place you would look for
    # the question you are being asked.
    assert asking._fields["ask_frame"].before is asking._fields["after_ask"]


def test_the_conversation_is_shown_and_only_appended_to():
    """Redrawing it every tick would fight anyone scrolled back through it."""
    history = [("you", "what is my battery"), ("vasper", "Full, and on mains.")]
    board, _ = _dashboard({"awake": True}, transcript=lambda: tuple(history))
    written = "".join(text for _tag, text in board._fields["log"].lines)
    assert "what is my battery" in written
    assert "Full, and on mains." in written

    history.append(("you", "open chrome"))
    board._fields["log"].lines.clear()
    board._apply({"awake": True})
    written = "".join(text for _tag, text in board._fields["log"].lines)
    assert written.strip().endswith("open chrome"), "only the new line redrawn"


def test_a_restart_clears_the_stale_conversation():
    history = [("you", "one"), ("vasper", "two"), ("you", "three")]
    board, _ = _dashboard({"awake": True}, transcript=lambda: tuple(history))
    history.clear()
    history.append(("you", "fresh"))
    board._apply({"awake": True})
    written = "".join(text for _tag, text in board._fields["log"].lines)
    assert "fresh" in written and "three" not in written


def test_an_unasked_change_is_the_one_number_that_shouts():
    """It should always be zero, so it is the only counter that changes colour."""
    calm, module = _dashboard({"approvals": 2, "refusals": 1})
    assert calm._fields["care"].fg == module.GOOD
    assert "NOT ASKED" not in calm._fields["care"].text

    loud, _ = _dashboard({"approvals": 1, "unasked": 2})
    assert loud._fields["care"].fg == module.WARN
    assert "NOT ASKED ABOUT" in loud._fields["care"].text


def test_a_clean_session_says_so_in_words():
    board, _ = _dashboard({"turns": 3})
    assert board._fields["care"].text == "nothing on your machine has been changed"


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


# --- taking the hands back needs no name ------------------------------------


def test_hands_off_needs_no_name():
    """Like a no: making someone say a name before they may stop something is
    the wrong way round, and stopping cannot cause harm."""
    conversation, speaker, ui = _conversation()
    conversation._standing = {"hands": 0.0}

    handled = conversation._handle_local("hands off", named=False)
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    assert handled is True
    assert conversation.hands_standing is False
    assert conversation.brain.standing_revoked
    assert speaker.voice.lines[-1] == "Okay, hands off. I'll ask next time."
    assert ("hands-off", "hands") in ui.decisions


def test_bare_sleep_and_quiet_are_local():
    """Tonight "Vasper sleep." and "Vasper, quiet." both went to Claude, cost
    a turn each, and got an answer instead of silence."""
    conversation, speaker, _ = _conversation()
    conversation.wake.engage(time.monotonic())

    assert conversation._handle_local("sleep") is True
    # Mute barges in on whatever is being said, so let "Sleeping." land first.
    speaker.wait_until_idle(timeout=5.0)
    assert conversation._handle_local("quiet") is True
    speaker.wait_until_idle(timeout=5.0)
    speaker.close()

    assert not conversation.wake.engaged(time.monotonic())
    assert speaker.voice.lines[-2:] == ["Sleeping.", "Quiet from now on."]
    assert conversation.brain.asked == []


# --- the holding phrases do not repeat ----------------------------------------


def test_a_filler_is_not_repeated_within_three():
    """Avoiding only the previous one meant "Hang on." and "One moment." could
    alternate for a whole evening, which is the robotic thing the fillers exist
    to avoid."""
    from vesper.brain.persona import THINKING_FILLERS

    conversation, speaker, _ = _conversation()
    speaker.close()
    recent: list[str] = []
    for _ in range(60):
        chosen = conversation._filler(THINKING_FILLERS)
        assert chosen not in recent[-3:], recent[-4:]
        recent.append(chosen)


def test_a_small_pool_still_yields_a_filler():
    conversation, speaker, _ = _conversation()
    speaker.close()
    for _ in range(6):
        assert conversation._filler(("A.", "B.")) in ("A.", "B.")
