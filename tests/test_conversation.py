"""The conversation loop, end to end, with no hardware and no network.

This is the test that actually says whether Vesper works: whether being spoken
to produces speech, whether interrupting it shuts it up, and whether it can tell
its own voice from yours.
"""

import threading
import time

import numpy as np
import pytest

from vesper.audio.speaker import Speaker
from vesper.audio.vad import EndpointConfig
from vesper import conversation as conversation_module
from vesper.conversation import Conversation, ConversationConfig
from vesper.stt.whisper import Transcript
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI


def build(
    *,
    replies=None,
    transcripts=None,
    voice_duration=0.0,
    config=None,
    wake_config=None,
):
    brain = FakeBrain(replies)
    stt = FakeSTT(transcripts)
    voice = FakeVoice(duration=voice_duration)
    speaker = Speaker(voice)
    mic = FakeMic()
    ui = RecordingUI()
    conversation = Conversation(
        brain=brain,
        stt=stt,
        speaker=speaker,
        mic=mic,
        wake=WakeGate(wake_config or WakeConfig()),
        config=config or ConversationConfig(greet_on_start=False),
        ui=ui,
    )
    return conversation, brain, stt, voice, speaker, ui


def audio(seconds=1.0, level=0.2):
    return np.full(int(16000 * seconds), level, dtype=np.float32)


def settle(speaker, timeout=5.0):
    assert speaker.wait_until_idle(timeout), "speaker never drained"


# --- a basic exchange -------------------------------------------------------


def test_being_addressed_produces_speech():
    conv, brain, _, voice, speaker, ui = build(
        replies=["Just after two in the morning."],
        transcripts=["Vesper, what time is it"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()

    assert voice.lines == ["Just after two in the morning."]
    assert conv.turns == 1
    assert len(ui.answers) == 1


def test_the_wake_word_is_not_sent_to_claude():
    conv, brain, _, _, speaker, _ = build(
        replies=["Fine."], transcripts=["Vesper, how is the disk"]
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    # Only the spoken part is checked. The machine context above it is read
    # live from this machine, and asserting on the whole payload made the test
    # fail whenever the foreground window happened to be titled "Vesper".
    utterance = brain.asked[0].split("[end context]")[-1]
    assert "how is the disk" in utterance
    assert "Vesper" not in utterance


def test_machine_context_is_attached_to_every_turn():
    conv, brain, _, _, speaker, _ = build(
        replies=["Fine."], transcripts=["Vesper, what am I doing"]
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    prompt = brain.asked[0]
    assert "machine context" in prompt
    assert "time:" in prompt


def test_speech_not_addressed_to_vesper_is_ignored():
    conv, brain, _, voice, speaker, ui = build(transcripts=["so anyway I told him no"])
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert brain.asked == []
    assert voice.lines == []
    assert ui.heard_lines == [("so anyway I told him no", False)]


def test_a_follow_up_needs_no_wake_word():
    conv, brain, _, _, speaker, _ = build(
        replies=["Sixteen.", "Sixty percent."],
        transcripts=["Vesper how many cores", "and the memory"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert len(brain.asked) == 2
    assert "and the memory" in brain.asked[1]


def test_unclear_audio_is_discarded_without_asking_claude():
    conv, brain, _, _, speaker, ui = build(transcripts=[""])
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert brain.asked == []
    assert ui.discards == ["too-quiet"]


def test_the_name_alone_gets_an_acknowledgement():
    conv, brain, _, voice, speaker, _ = build(transcripts=["Vesper"])
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert voice.lines == ["Yes?"]
    assert brain.asked == []


# --- speaking as it thinks --------------------------------------------------


def test_a_multi_sentence_reply_is_spoken_sentence_by_sentence():
    """Each sentence must be queued as it completes, not held to the end."""
    conv, _, _, voice, speaker, _ = build(
        replies=["The build passed. Nothing else broke. You are clear to push."],
        transcripts=["Vesper how is the build"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert voice.lines == [
        "The build passed.",
        "Nothing else broke.",
        "You are clear to push.",
    ]


def test_screen_content_is_displayed_and_never_spoken():
    conv, _, _, voice, speaker, ui = build(
        replies=["It fails on line forty. <screen>main.py:40 raise ValueError</screen>"],
        transcripts=["Vesper where does it fail"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()

    spoken = " ".join(voice.lines)
    assert "main.py" not in spoken
    assert "It fails on line forty." in spoken
    assert any("main.py:40" in s for s in ui.screens)


def test_markdown_never_reaches_the_speech_engine():
    conv, _, _, voice, speaker, _ = build(
        replies=["**Three** things broke. See `main.py` for the first one."],
        transcripts=["Vesper what broke"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    spoken = " ".join(voice.lines)
    assert "**" not in spoken and "`" not in spoken
    assert "Three things broke." in spoken


def test_tool_calls_are_surfaced_to_the_ui():
    conv, brain, _, _, speaker, ui = build(
        replies=["Sixteen cores."], transcripts=["Vesper how many cores"]
    )
    brain.tools_to_report = [("Bash", "python -c print(os.cpu_count())")]
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert ui.tools == [("Bash", "python -c print(os.cpu_count())")]


# --- hearing itself ---------------------------------------------------------


def test_vesper_does_not_answer_its_own_voice():
    """On laptop speakers the mic hears the reply. Without this it loops forever."""
    conv, brain, stt, voice, speaker, ui = build(
        replies=["The build passed and nothing else broke."],
        transcripts=["Vesper how is the build", "The build passed and nothing else broke"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    conv._on_utterance(audio())  # the mic picking up the reply
    settle(speaker)
    speaker.close()

    assert len(brain.asked) == 1, "it answered its own echo"
    assert conv.echo_rejections == 1
    assert "heard myself" in ui.discards


def test_a_genuine_reply_that_shares_a_word_still_gets_through():
    conv, brain, _, _, speaker, _ = build(
        replies=["The build passed and nothing else broke.", "Right."],
        transcripts=["Vesper how is the build", "okay push it then"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert len(brain.asked) == 2


def test_a_very_short_utterance_is_never_treated_as_echo():
    conv, brain, _, _, speaker, _ = build(
        replies=["Yes it did.", "Sure."],
        transcripts=["Vesper did it pass", "yes"],
    )
    conv._on_utterance(audio())
    settle(speaker)
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert len(brain.asked) == 2


# --- barge-in ---------------------------------------------------------------


def test_talking_over_vesper_stops_it_mid_sentence():
    conv, brain, _, voice, speaker, ui = build(
        replies=["This is a long answer. With several sentences. And another one."],
        transcripts=["Vesper explain everything"],
        voice_duration=1.5,
    )
    conv._on_utterance(audio())
    # Wait for playback to actually be under way.
    for _ in range(200):
        if speaker.speaking:
            break
        time.sleep(0.01)
    assert speaker.speaking

    conv._interrupt()
    settle(speaker)
    speaker.close()

    assert conv.interruptions == 1
    assert brain.interrupted.is_set()
    assert ui.interruptions and ui.interruptions[0] >= 1
    assert len(voice.lines) < 3, "queued sentences should have been dropped"


def test_self_mute_window_stops_vesper_interrupting_itself():
    """Right after playback starts, the mic hears the attack of our own voice."""
    conv, *_ = build(config=ConversationConfig(self_mute_ms=250, greet_on_start=False))
    conv._speaking_since = time.monotonic()
    loud = np.full(480, 0.5, dtype=np.float32)
    for _ in range(10):
        assert conv._detect_barge_in(loud) is False


def test_barge_in_needs_sustained_speech_not_one_noisy_block():
    conv, *_ = build(
        config=ConversationConfig(
            barge_in_blocks=3, self_mute_ms=0, greet_on_start=False
        )
    )
    conv._speaking_since = time.monotonic() - 10
    conv._barge_vad = _AlwaysSpeech()
    loud = np.full(480, 0.5, dtype=np.float32)
    assert conv._detect_barge_in(loud) is False  # 1
    assert conv._detect_barge_in(loud) is False  # 2
    assert conv._detect_barge_in(loud) is True  # 3


def test_a_single_clack_does_not_interrupt():
    conv, *_ = build(
        config=ConversationConfig(barge_in_blocks=3, self_mute_ms=0, greet_on_start=False)
    )
    conv._speaking_since = time.monotonic() - 10
    conv._barge_vad = _AlternatingSpeech()
    loud = np.full(480, 0.5, dtype=np.float32)
    for _ in range(12):
        assert conv._detect_barge_in(loud) is False


def test_half_duplex_ignores_the_microphone_while_speaking():
    conv, brain, stt, _, speaker, _ = build(
        config=ConversationConfig(half_duplex=True, greet_on_start=False),
        voice_duration=1.0,
        replies=["Talking now."],
        transcripts=["Vesper say something"],
    )
    conv._on_utterance(audio())
    for _ in range(200):
        if speaker.speaking:
            break
        time.sleep(0.01)

    before = stt.calls
    conv._handle_block(np.full(480, 0.9, dtype=np.float32))
    assert stt.calls == before, "half duplex must not transcribe while speaking"
    speaker.barge_in()
    speaker.close()


# --- errors -----------------------------------------------------------------


def test_a_brain_error_is_spoken_rather_than_swallowed():
    """Said plainly, logged in full.

    The spoken line and the logged line used to be the same string, so an OS
    error carrying a file path was read out loud in Vesper's voice. That is the
    failure brain/failures.py already prevents for the CLI's own error text,
    applied now to ours.
    """
    from vesper.brain.protocol import BrainError

    raw = "could not reach the brain: [WinError 232] C:/Vesper/vesper/brain"

    class FailingBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield BrainError(raw)

    conv, _, _, voice, speaker, ui = build(transcripts=["Vesper hello"])
    conv.brain = FailingBrain()
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()

    assert voice.lines, "a brain error must still be spoken, not swallowed"
    spoken = voice.lines[0]
    assert "WinError" not in spoken and "C:/" not in spoken
    assert spoken == "I lost my connection to Claude. Give me a moment and ask me again."
    assert ui.errors == [raw], "the detail still belongs in the log"


def test_a_timeout_already_reads_as_a_sentence_and_is_left_alone():
    from vesper.brain.protocol import BrainError

    class SlowBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield BrainError("that took too long, so I stopped waiting")

    conv, _, _, voice, speaker, _ui = build(transcripts=["Vesper hello"])
    conv.brain = SlowBrain()
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert voice.lines == ["that took too long, so I stopped waiting"]


def test_permission_requests_reach_the_ui_and_are_asked_out_loud():
    from vesper.brain.protocol import PermissionNeeded, TurnComplete

    class AskingBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield PermissionNeeded(
                "Write", "C:/notes.txt", tool_input={"file_path": "C:/notes.txt"}
            )
            yield TurnComplete(text="I want to save that note.", turns=1)

    conv, _, _, voice, speaker, ui = build(transcripts=["Vesper save a note"])
    conv.brain = AskingBrain()
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert ui.permissions == [("Write", "Write: C:/notes.txt")]
    # The answer is spoken first, then the question, so Vesper never talks over
    # its own explanation of what it was trying to do.
    assert voice.lines == [
        "I want to save that note.",
        "I want to create notes dot txt, which lets me write to other files "
        "too. Do I do this for you?",
    ]


# --- helpers ----------------------------------------------------------------


class _AlwaysSpeech:
    def is_speech(self, block):
        return True

    def reset(self):
        pass


class _AlternatingSpeech:
    def __init__(self):
        self.n = 0

    def is_speech(self, block):
        self.n += 1
        return self.n % 2 == 1

    def reset(self):
        pass


def test_one_bad_audio_block_does_not_end_the_assistant():
    """`run()` caught only KeyboardInterrupt.

    Anything else out of transcription, the voiceprint or the turn itself went
    straight through the loop and out of `run()`. Started from a login shortcut
    there is no console for that traceback, so the symptom is Vesper vanishing
    the moment you speak to it and nothing anywhere saying why.
    """
    conv, brain, stt, voice, speaker, ui = build(transcripts=["Vesper hello"])

    blocks = [np.zeros(480, dtype=np.float32) for _ in range(3)]
    handled: list[int] = []

    def explode_once(block):
        handled.append(len(handled))
        if len(handled) == 1:
            raise RuntimeError("the gpu fell over")

    conv._handle_block = explode_once
    conv.mic.blocks = list(blocks)

    def read(timeout=0.5):
        if not conv.mic.blocks:
            conv._running.clear()
            return None
        return conv.mic.blocks.pop(0)

    conv.mic.read = read
    conv.run()

    assert len(handled) == 3, "the loop kept going after the block that raised"
    assert conv.errors == 1
    assert any("handling audio failed" in message for message in ui.errors)
    assert conv.status()["errors"] == 1


# --- dropped audio ----------------------------------------------------------


def test_dropped_audio_is_reported_once_rather_than_per_block():
    """The count is kept by the microphone and the sentence is said here.

    Nine overflows inside a second is one problem, not nine, and the reporting
    is what used to cause the next one.
    """
    conv, _brain, _stt, _voice, _speaker, ui = build()
    conv.mic.overflows = 9

    conv._report_overflows()  # arms the timer, says nothing yet
    conv._overflows_said_at = 0.0
    conv._report_overflows()

    said = [line for line in ui.warnings if "dropped audio" in line]
    assert len(said) == 1, f"expected one summary, got {ui.warnings}"
    assert "9 times" in said[0]


def test_a_quiet_microphone_says_nothing_at_all():
    conv, _brain, _stt, _voice, _speaker, ui = build()
    conv._overflows_said_at = 0.0
    conv._report_overflows()
    assert not [line for line in ui.warnings if "dropped audio" in line]


def test_overflows_are_not_reported_more_than_once_a_minute():
    conv, _brain, _stt, _voice, _speaker, ui = build()
    conv._overflows_said_at = 0.0
    conv.mic.overflows = 3
    conv._report_overflows()
    conv.mic.overflows = 4
    conv._report_overflows()

    said = [line for line in ui.warnings if "dropped audio" in line]
    assert len(said) == 1
    assert conv.mic.overflows == 4, "the second burst is still being counted up"


# --- two threads, one deque -------------------------------------------------


def test_the_echo_filter_survives_being_spoken_to_from_another_thread():
    """`_is_own_voice` iterates `_spoken_recently`, which `_say` pops from.

    The proactive loop speaks from its own thread, so the two really do run at
    once, and `_spoken_recently` has a maxlen, so an append also pops.

    A smoke test rather than a reproduction, and worth saying so. The bare
    pattern does raise "deque mutated during iteration" in about three seconds
    with three writers, but it could not be provoked through this function even
    with sys.setswitchinterval at a microsecond, because nearly all of the time
    here goes on regex and set work rather than on the loop. What this catches
    is somebody removing the lock and something much more exposed taking its
    place, which is the regression worth catching.
    """
    conv, _brain, _stt, _voice, speaker, _ui = build()
    stop = threading.Event()
    raced = []

    def keep_saying():
        count = 0
        while not stop.is_set():
            count += 1
            conv._remember_said(f"the battery is at eighty percent, reading {count}")

    def keep_checking():
        while not stop.is_set():
            try:
                conv._is_own_voice("the battery is at eighty percent and charging")
            except RuntimeError as exc:  # deque mutated during iteration
                raced.append(str(exc))
                return

    writers = [threading.Thread(target=keep_saying, daemon=True) for _ in range(3)]
    checker = threading.Thread(target=keep_checking, daemon=True)
    for thread in writers:
        thread.start()
    checker.start()
    time.sleep(3.0)
    stop.set()
    for thread in writers + [checker]:
        thread.join(timeout=5)
    speaker.close()

    assert not raced, f"the echo filter raced the writer: {raced}"


def test_the_transcript_can_be_read_while_it_is_being_written():
    """The dashboard reads this on its own thread once a second.

    `tuple(deque)` turns out to be safe on its own: CPython builds it in one C
    call that never yields, and three writers for three seconds could not
    disturb it. The lock is here anyway, because that is an implementation
    detail rather than a promise, and it is exactly the promise a free threaded
    build takes away. It costs a handful of microseconds once a second.
    """
    conv, _brain, _stt, _voice, speaker, _ui = build()
    stop = threading.Event()
    raced = []

    def keep_saying():
        count = 0
        while not stop.is_set():
            count += 1
            conv._remember_said(f"line number {count}")
            conv._remember_heard(f"and a reply to {count}")

    def keep_reading():
        while not stop.is_set():
            try:
                assert isinstance(conv.transcript(), tuple)
            except RuntimeError as exc:
                raced.append(str(exc))
                return

    writers = [threading.Thread(target=keep_saying, daemon=True) for _ in range(3)]
    reader = threading.Thread(target=keep_reading, daemon=True)
    for thread in writers:
        thread.start()
    reader.start()
    time.sleep(1.5)
    stop.set()
    for thread in writers + [reader]:
        thread.join(timeout=5)
    speaker.close()

    assert not raced, f"transcript() raced the writer: {raced}"


# --- starting up ------------------------------------------------------------


def test_the_microphone_opens_before_any_model_is_built():
    """Vesper used to be deaf for the whole of startup.

    Measured one piece at a time on this machine: 4.5s of model building warm,
    and the log has a cold start where whisper alone took 7.4s. Startup is also
    when you are most likely to say something, having just started it.
    """
    order = []

    conv, _brain, stt, _voice, speaker, _ui = build()
    conv.mic.start = lambda: order.append("mic")
    stt.load = lambda: order.append("whisper")
    conv.brain.auth_status = lambda: order.append("brain") or True

    conv.start()
    for thread in conv._warming:
        thread.join(timeout=10)
    conv._running.clear()
    settle(speaker)
    speaker.close()

    assert order[0] == "mic", f"the microphone opened at position {order.index('mic')}"
    assert "whisper" in order, "the model was never warmed"


def test_a_model_that_will_not_warm_does_not_stop_him_starting():
    """The person who asks for it should be told, not the log at startup."""
    conv, _brain, stt, _voice, speaker, ui = build()

    def refuse():
        raise RuntimeError("out of memory")

    stt.load = refuse
    conv.start()
    for thread in conv._warming:
        thread.join(timeout=10)
    conv._running.clear()
    settle(speaker)
    speaker.close()

    assert conv.mic.started, "the microphone should be open regardless"
    assert any("did not warm up" in w for w in ui.warnings), ui.warnings


def test_the_first_utterance_never_builds_a_second_model():
    """Two copies of small.en is how a card with room for one runs out.

    The warming thread and the first utterance can both find the model unset,
    so `load` takes a lock and checks again inside it.
    """
    from vesper.stt.whisper import Listener, WhisperConfig

    stt = Listener(WhisperConfig(model="tiny.en", device="cpu"), log=lambda m: None)
    builds = []
    real = stt._load_unlocked

    def counted():
        builds.append(1)
        time.sleep(0.05)  # widen the window the lock has to cover
        real()

    stt._load_unlocked = counted
    threads = [threading.Thread(target=stt.load) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(builds) == 1, f"{len(builds)} models were built at once"


# --- the voice profile, watched rather than trusted --------------------------
#
# Added after a real profile rejected its owner 266 times across eleven days and
# nothing anywhere said so. These pin the two halves of the fix: a profile that
# never matches is called out, and a profile that does match keeps up with the
# room it is used in.


class StubProfile:
    def __init__(self, *, calibrated=True, match_threshold=0.5):
        self.calibrated = calibrated
        self.match_threshold = match_threshold
        self.reject_threshold = match_threshold - 0.2
        self.samples = 4
        self.created = "2026-09-11T00:00:00"
        self.spread = 0.9


class StubVoicePrint:
    """A voiceprint whose verdict the test chooses."""

    def __init__(self, verdict, score, *, calibrated=True, match_threshold=0.5):
        self.enrolled = True
        self.profile = StubProfile(
            calibrated=calibrated, match_threshold=match_threshold
        )
        self._verdict, self._score = verdict, score
        self.adapted = 0

    def compare(self, audio, sample_rate=16000):
        return self._verdict, self._score

    def adapt(self, audio, sample_rate=16000):
        self.adapted += 1
        return True


def _score_many(conv, count):
    for _ in range(count):
        conv._is_the_right_voice(audio())


def test_a_profile_that_never_matches_is_called_out_once():
    conv, _, _, _, speaker, ui = build()
    conv.voiceprint = StubVoicePrint("unsure", 0.32)
    _score_many(conv, conversation_module.VOICE_DOUBT_AFTER + 6)
    speaker.close()

    complaints = [w for w in ui.warnings if "not matched you" in w]
    assert len(complaints) == 1, ui.warnings
    assert "--voicecheck" in complaints[0]


def test_nothing_is_said_before_there_is_enough_evidence():
    conv, _, _, _, speaker, ui = build()
    conv.voiceprint = StubVoicePrint("unsure", 0.32)
    _score_many(conv, conversation_module.VOICE_DOUBT_AFTER - 1)
    speaker.close()

    assert not [w for w in ui.warnings if "not matched you" in w]


def test_a_working_profile_is_never_called_out():
    conv, _, _, _, speaker, ui = build()
    conv.voiceprint = StubVoicePrint("match", 0.82)
    _score_many(conv, conversation_module.VOICE_DOUBT_AFTER + 6)
    speaker.close()

    assert not [w for w in ui.warnings if "not matched you" in w]


def test_rejections_alone_still_raise_the_alarm():
    """Ignoring you 266 times is the failure mode, so it has to count too."""
    conv, _, _, _, speaker, ui = build()
    conv.voiceprint = StubVoicePrint("different", 0.10)
    _score_many(conv, conversation_module.VOICE_DOUBT_AFTER + 2)
    speaker.close()

    assert [w for w in ui.warnings if "not matched you" in w]


def test_status_reports_both_halves_of_the_voice_story():
    conv, _, _, _, speaker, _ = build()
    conv.voiceprint = StubVoicePrint("match", 0.82)
    _score_many(conv, 3)
    speaker.close()

    status = conv.status()
    assert status["voice_scored"] == 3
    assert status["voice_matches"] == 3


def test_a_marginal_match_is_folded_into_the_profile():
    """A clip that only just cleared the bar is the one worth learning from."""
    conv, _, _, _, speaker, _ = build()
    print_ = StubVoicePrint("match", 0.52, match_threshold=0.5)
    conv.voiceprint = print_
    conv._is_the_right_voice(audio())
    speaker.close()

    assert print_.adapted == 1


def test_a_comfortable_match_teaches_nothing_and_is_not_stored():
    conv, _, _, _, speaker, _ = build()
    print_ = StubVoicePrint("match", 0.95, match_threshold=0.5)
    conv.voiceprint = print_
    conv._is_the_right_voice(audio())
    speaker.close()

    assert print_.adapted == 0


def test_learning_is_rate_limited_so_the_file_is_not_rewritten_constantly():
    conv, _, _, _, speaker, _ = build()
    print_ = StubVoicePrint("match", 0.52, match_threshold=0.5)
    conv.voiceprint = print_
    _score_many(conv, 8)
    speaker.close()

    assert print_.adapted == 1


def test_an_uncalibrated_profile_is_never_learned_into():
    """Adding clips to a profile whose bar was never measured compounds a guess."""
    conv, _, _, _, speaker, _ = build()
    print_ = StubVoicePrint("match", 0.52, calibrated=False, match_threshold=0.5)
    conv.voiceprint = print_
    conv._is_the_right_voice(audio())
    speaker.close()

    assert print_.adapted == 0


# --- reacting fast ----------------------------------------------------------
#
# Two separate savings. A prefix is transcribed while the endpointer is still
# waiting out the silence, which takes the transcriber off the critical path
# entirely; and a holding phrase now fires on a deadline rather than only on the
# first tool call, so a plain question no longer buys the brain's whole time to
# first token as dead air.


class SlowBrain(FakeBrain):
    """Says nothing for a while, then answers. The shape of a real first token."""

    def __init__(self, replies, delay=0.45):
        super().__init__(replies)
        self.delay = delay

    def ask(self, payload):
        time.sleep(self.delay)
        yield from super().ask(payload)


def _slow_build(replies, delay=0.45, **kwargs):
    conv, brain, stt, voice, speaker, ui = build(replies=replies, **kwargs)
    slow = SlowBrain(replies, delay=delay)
    conv.brain = slow
    return conv, slow, voice, speaker, ui


def test_a_silent_turn_gets_a_holding_phrase():
    """Without this a plain question is 1.5s of silence, which reads as a crash."""
    conv, _, voice, speaker, _ = _slow_build(["It is half past two."], delay=0.5)
    conv.config.quick_filler_after_s = 0.1
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    assert len(voice.lines) == 2, voice.lines
    assert voice.lines[-1] == "It is half past two."


def test_a_fast_turn_says_no_holding_phrase_at_all():
    """The filler is for dead air. An answer that arrives first leaves none."""
    conv, _, voice, speaker, _ = _slow_build(["Half past two."], delay=0.0)
    conv.config.quick_filler_after_s = 5.0
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    assert voice.lines == ["Half past two."]


def test_zero_disables_the_holding_phrase():
    conv, _, voice, speaker, _ = _slow_build(["Half past two."], delay=0.3)
    conv.config.quick_filler_after_s = 0.0
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    assert voice.lines == ["Half past two."]


def test_only_one_holding_phrase_however_many_threads_want_one():
    """The timer and the first tool call both want to speak. One of them wins."""
    conv, brain, voice, speaker, _ = _slow_build(["Done."], delay=0.3)
    conv.config.quick_filler_after_s = 0.05
    brain.tools_to_report = [("Bash", "ls")]
    conv.respond("list my files")
    settle(speaker)
    speaker.close()

    assert len(voice.lines) == 2, voice.lines


def test_the_timer_does_not_speak_after_the_turn_is_over():
    """A cancelled timer is the whole reason the turn wraps in try/finally.

    Left running, a turn that answered instantly would be followed half a second
    later by a cheerful "let me look" into the silence.
    """
    conv, _, voice, speaker, _ = _slow_build(["Half past two."], delay=0.0)
    conv.config.quick_filler_after_s = 0.15
    conv.respond("what time is it")
    settle(speaker)
    time.sleep(0.4)  # past the deadline the timer would have fired at
    settle(speaker)
    speaker.close()

    assert voice.lines == ["Half past two."]


def test_a_prefix_transcribed_during_the_pause_is_the_one_used():
    conv, _, stt, _, speaker, _ = build(
        replies=["Two."], transcripts=["Vesper, what is one plus one"]
    )
    conv._start_early_transcription(audio())
    conv.endpointer.peek_was_final = True
    transcript = conv._transcript_for(audio())
    speaker.close()

    assert transcript.text == "Vesper, what is one plus one"
    # One decode, not two: the saving is that the full audio is never re-read.
    assert stt.calls == 1, stt.calls


def test_a_prefix_is_thrown_away_when_you_kept_talking():
    """Answering half a sentence because somebody paused is the failure here."""
    conv, _, stt, _, speaker, _ = build(
        replies=["Two."], transcripts=["Vesper, what is one", "Vesper, what is one plus one"]
    )
    conv._start_early_transcription(audio())
    conv.endpointer.peek_was_final = False
    transcript = conv._transcript_for(audio())
    speaker.close()

    assert transcript.text == "Vesper, what is one plus one"


def test_no_prefix_means_the_ordinary_path():
    conv, _, stt, _, speaker, _ = build(
        replies=["Two."], transcripts=["Vesper, what time is it"]
    )
    transcript = conv._transcript_for(audio())
    speaker.close()

    assert transcript.text == "Vesper, what time is it"
    assert stt.calls == 1


# --- two bugs the speed work shipped with -----------------------------------


def test_a_slow_prefix_is_waited_for_and_then_actually_used():
    """Both bugs in one sentence: it waited, then threw the answer away.

    The collector snapshotted the result before waiting for it, so a worker that
    had not finished yet was waited on and then ignored, and the full audio was
    decoded again. That is two decodes on exactly the slow machines the early
    look was built to help.
    """
    conv, _, stt, _, speaker, _ = build(
        replies=["Two."], transcripts=["Vesper, what is one plus one"]
    )
    real = stt.transcribe

    def slow(audio, sample_rate=16000):
        time.sleep(0.25)
        return real(audio, sample_rate)

    stt.transcribe = slow
    conv._start_early_transcription(audio())
    conv.endpointer.peek_was_final = True
    transcript = conv._transcript_for(audio())  # collects before the worker ends
    speaker.close()

    assert transcript.text == "Vesper, what is one plus one"
    assert stt.calls == 1, f"decoded {stt.calls} times, so the wait bought nothing"


def test_an_answer_with_no_streaming_deltas_silences_the_holding_phrase(monkeypatch):
    """The fallback reply path speaks without going through say_sentence, so it
    left `spoken_anything` false. A holding phrase whose deadline landed there
    then offered to look into a question that had just been answered.

    Timer.cancel cannot close this, because it cannot stop a callback that has
    already begun, so the guard the callback reads has to be accurate instead.
    The timer is replaced here rather than raced: the callback is captured, the
    turn is run to completion, and only then is it fired, which is the exact
    ordering the bug needs and the one a sleep cannot pin down.
    """
    fired = []

    class CapturedTimer:
        def __init__(self, interval, function):
            self.function = function
            fired.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(conversation_module.threading, "Timer", CapturedTimer)

    conv, _, _, voice, speaker, _ = build()

    class NoDeltaBrain(FakeBrain):
        def ask(self, payload):
            from vesper.brain.protocol import TurnComplete

            self.asked.append(payload)
            yield TurnComplete(
                text="It is half past two.", is_error=False, api_error="",
                cost_usd=0.0, duration_ms=1, session_id="fake-session",
            )

    conv.brain = NoDeltaBrain()
    conv.config.quick_filler_after_s = 0.1
    conv.respond("what time is it")
    settle(speaker)
    assert voice.lines == ["It is half past two."], voice.lines

    # The deadline arrives late, after the answer is already out.
    assert fired, "no holding-phrase timer was ever created"
    fired[0].function()
    settle(speaker)
    speaker.close()

    assert voice.lines == ["It is half past two."], (
        "a holding phrase spoke after the answer it was meant to cover"
    )


def test_a_superseded_prefix_does_not_stall_the_audio_loop():
    """You paused, it started decoding, you carried on talking.

    The speculative answer is now a prefix of a sentence nobody asked about, and
    the full decode has to queue behind it. Waiting for that unbounded was the
    bug: this runs on the only thread reading the microphone, whose queue holds
    about 7.7 seconds before the callback starts dropping the oldest audio. A
    slow decode stopped being latency and became lost speech.
    """
    conv, _, stt, _, speaker, ui = build(
        replies=["Two."], transcripts=["Vesper, what is one plus one"]
    )
    conv.config.early_transcribe_wait_s = 0.2

    started = threading.Event()
    release = threading.Event()

    def wedged(audio, sample_rate=16000):
        started.set()
        release.wait(5.0)
        return Transcript(text="prefix")

    stt.transcribe = wedged
    conv._start_early_transcription(audio())
    assert started.wait(2.0), "the speculative worker never began"
    conv.endpointer.peek_was_final = False  # you kept talking

    began = time.monotonic()
    transcript = conv._transcript_for(audio())
    waited = time.monotonic() - began
    release.set()
    speaker.close()

    assert waited < 2.0, f"the audio loop was blocked for {waited:.1f}s"
    assert not transcript.ok
    assert "busy" in (transcript.rejected_reason or "")
    assert any("dropped one utterance" in w for w in ui.warnings), ui.warnings
