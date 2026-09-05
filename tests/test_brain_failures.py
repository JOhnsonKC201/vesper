"""A turn the CLI reports as failed is a failure, not an answer.

On 2026-09-04 the CLI's saved login expired while the laptop slept. Every turn
came back as an error result whose text was

    Failed to authenticate: OAuth session expired and could not be refreshed

and Vesper spoke that sentence, in its own voice, three times, then counted each
turn as a success. Nothing in the log said anything had gone wrong. These tests
pin the behaviour that should have happened instead: the CLI's words are never
spoken, one honest sentence is, the failure is logged and counted, the brain is
respawned once so a login fixed elsewhere can reach it, and after that Vesper
waits quietly for `claude auth status` to say the login is back.

The error text below is verbatim from the brain's transcript of that night.
"""

import threading

import numpy as np

from vesper.audio.speaker import Speaker
from vesper.brain.protocol import TextDelta, TurnComplete
from vesper.conversation import Conversation, ConversationConfig
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI

VERBATIM = "Failed to authenticate: OAuth session expired and could not be refreshed"


class FailingBrain(FakeBrain):
    """Answers the way the CLI does when its login is dead.

    The first `fail_turns` asks yield one error result and no deltas, exactly the
    frame shape recorded on 2026-09-04. After that it answers like FakeBrain.
    """

    def __init__(
        self,
        *,
        fail_turns=10**9,
        replies=None,
        logged_in=True,
        error_text=VERBATIM,
        error_fields=None,
    ):
        super().__init__(replies)
        self.fail_turns = fail_turns
        self.logged_in = logged_in
        self.error_text = error_text
        self.error_fields = dict(error_fields or {})
        self.restarts = 0
        self.status_checks = 0

    def restart(self):
        self.restarts += 1

    def auth_status(self):
        self.status_checks += 1
        return self.logged_in

    def ask(self, text):
        self.asked.append(text)
        self.interrupted.clear()
        if len(self.asked) <= self.fail_turns:
            yield TurnComplete(
                text=self.error_text,
                session_id=self.session_id,
                is_error=True,
                **self.error_fields,
            )
            return
        reply = self.replies.pop(0) if self.replies else "Nothing to report."
        for start in range(0, len(reply), self.chunk):
            yield TextDelta(reply[start : start + self.chunk])
        self.turn_count += 1
        yield TurnComplete(text=reply, session_id=self.session_id, turns=1)


class LockingBrain(FailingBrain):
    """Holds a lock for as long as its `ask` generator is open, like the real
    ClaudeBrain does, and refuses a restart while it is held.

    FakeBrain has no lock, so a respawn issued from inside the turn's loop
    passes the other tests and deadlocks the real assistant. This one fails
    instead of hanging.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._busy = threading.Lock()

    def ask(self, text):
        with self._busy:
            yield from super().ask(text)

    def restart(self):
        if not self._busy.acquire(timeout=1.0):
            raise RuntimeError("restart called while a turn still held the busy lock")
        try:
            super().restart()
        finally:
            self._busy.release()


def build(brain, transcripts):
    stt = FakeSTT(transcripts)
    voice = FakeVoice()
    speaker = Speaker(voice)
    conversation = Conversation(
        brain=brain,
        stt=stt,
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(greet_on_start=False),
        ui=RecordingUI(),
    )
    return conversation, voice, speaker, conversation.ui


def audio(seconds=1.0):
    return np.full(int(16000 * seconds), 0.2, dtype=np.float32)


def say(conversation, speaker, times=1):
    for _ in range(times):
        conversation._on_utterance(audio())
        assert speaker.wait_until_idle(5.0), "speaker never drained"


def login_lines(voice):
    return [line for line in voice.lines if "login" in line.lower()]


# --- naming the failure -----------------------------------------------------


def test_the_recorded_error_is_classified_as_a_login_failure():
    from vesper.brain import failures

    done = TurnComplete(text=VERBATIM, is_error=True)
    assert failures.classify(done) == failures.AUTH


def test_the_api_error_code_alone_is_enough():
    from vesper.brain import failures

    done = TurnComplete(text="", is_error=True, api_error="authentication_failed")
    assert failures.classify(done) == failures.AUTH


def test_other_error_results_are_other():
    from vesper.brain import failures

    done = TurnComplete(text="Rate limit reached", is_error=True)
    assert failures.classify(done) == failures.OTHER


def test_a_normal_answer_that_mentions_login_is_not_an_error():
    from vesper.brain import failures

    done = TurnComplete(text="Your login expired last week, by the way.", is_error=False)
    assert failures.classify(done) is None


def test_the_spoken_lines_carry_no_url_and_no_dash():
    from vesper.brain import failures

    for kind in (failures.AUTH, failures.OTHER):
        line = failures.spoken_line(kind)
        assert line.strip()
        assert "http" not in line.lower()
        assert "—" not in line and "–" not in line


# --- the CLI's words are never Vesper's ------------------------------------


def test_the_cli_error_text_is_never_spoken():
    brain = FailingBrain()
    conv, voice, speaker, ui = build(brain, ["Vesper, extend my screen"])
    say(conv, speaker)
    speaker.close()

    assert VERBATIM not in voice.lines
    assert not any("authenticate" in line.lower() for line in voice.lines)
    assert len(login_lines(voice)) == 1, voice.lines
    # Logged, with the CLI's own words where a person reading the log needs
    # them, once per attempt.
    assert len(ui.errors) == 2 and all(VERBATIM in e for e in ui.errors)
    # Counted as failures, never reset, and no cost line for an empty turn.
    assert conv.register.failures_in_a_row == 2
    assert ui.answers == []
    # Respawned once so a login fixed elsewhere can reach the child, and the
    # same payload asked again exactly once.
    assert brain.restarts == 1
    assert len(brain.asked) == 2 and brain.asked[0] == brain.asked[1]
    assert conv.locked_out


def test_one_retry_recovers_without_a_word_about_it():
    brain = FailingBrain(fail_turns=1, replies=["It is ten past eleven."])
    conv, voice, speaker, ui = build(brain, ["Vesper, what time is it"])
    say(conv, speaker)
    speaker.close()

    assert voice.lines == ["It is ten past eleven."]
    assert conv.register.failures_in_a_row == 0
    assert brain.restarts == 1
    assert not conv.locked_out
    assert len(ui.answers) == 1


def test_the_respawn_happens_outside_the_turn_that_failed():
    brain = LockingBrain(fail_turns=1, replies=["Fine."])
    conv, voice, speaker, ui = build(brain, ["Vesper, hello"])
    say(conv, speaker)
    speaker.close()

    assert voice.lines == ["Fine."], (voice.lines, ui.errors)
    assert brain.restarts == 1


def test_other_errors_are_spoken_plainly_and_do_not_respawn():
    brain = FailingBrain(
        error_text="boom", error_fields={"error_subtype": "error_during_execution"}
    )
    conv, voice, speaker, ui = build(brain, ["Vesper, how is the disk"])
    say(conv, speaker)
    speaker.close()

    assert "boom" not in voice.lines
    assert len(voice.lines) == 1 and "Claude's side" in voice.lines[0]
    assert brain.restarts == 0
    assert len(brain.asked) == 1
    assert conv.register.failures_in_a_row == 1
    assert not conv.locked_out
    assert len(ui.errors) == 1 and "boom" in ui.errors[0]


# --- locked out, and back -----------------------------------------------------


def test_locked_out_it_complains_once_and_logs_the_rest():
    brain = FailingBrain(logged_in=False)
    conv, voice, speaker, ui = build(
        brain, ["Vesper, extend my screen", "Vesper, are you there", "Vesper, hello"]
    )
    say(conv, speaker, times=3)
    speaker.close()

    assert len(login_lines(voice)) == 1, voice.lines
    # Two attempts on the first utterance, then two "still not logged in".
    assert len(ui.errors) == 4
    assert len(brain.asked) == 2, "a known dead brain must not be asked again"
    assert brain.status_checks == 2
    assert conv.register.failures_in_a_row == 4
    assert conv.locked_out


def test_locked_out_it_repeats_itself_after_the_nag_interval():
    from vesper import conversation as module

    brain = FailingBrain(logged_in=False)
    conv, voice, speaker, ui = build(brain, ["Vesper, hello", "Vesper, hello again"])
    say(conv, speaker)
    conv._locked_out_told_at -= module.AUTH_NAG_S + 1
    say(conv, speaker)
    speaker.close()

    assert len(login_lines(voice)) == 2
    # Repeated because the quiet ran out, not because the turn was tried again.
    assert len(brain.asked) == 2 and conv.locked_out


def test_the_lockout_lifts_when_the_login_is_back():
    brain = FailingBrain(logged_in=False)
    conv, voice, speaker, ui = build(brain, ["Vesper, hello", "Vesper, are you back"])
    say(conv, speaker)
    assert conv.locked_out

    brain.logged_in = True
    brain.fail_turns = 0
    brain.replies = ["Back, and listening."]
    say(conv, speaker)
    speaker.close()

    assert voice.lines[-1] == "Back, and listening."
    assert brain.restarts == 2, "a fresh child is what re-reads the credentials"
    assert not conv.locked_out
    assert conv.register.failures_in_a_row == 0


# --- at startup ---------------------------------------------------------------


def test_start_does_not_spawn_a_brain_that_cannot_log_in():
    brain = FailingBrain(logged_in=False)
    conv, voice, speaker, ui = build(brain, [])
    conv.start()
    assert speaker.wait_until_idle(5.0)
    conv.stop()

    assert brain.started is False
    assert len(login_lines(voice)) == 1
    assert len(ui.errors) == 1 and "not logged in" in ui.errors[0]
    assert conv.locked_out
    assert conv.mic.started, "it must still listen, or it can never notice the login is back"


def test_start_proceeds_when_the_check_cannot_tell():
    brain = FailingBrain(logged_in=None)
    conv, voice, speaker, ui = build(brain, [])
    conv.start()
    conv.stop()

    assert brain.started is True
    assert login_lines(voice) == []
    assert not conv.locked_out
