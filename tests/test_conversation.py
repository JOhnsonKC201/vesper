"""The conversation loop, end to end, with no hardware and no network.

This is the test that actually says whether Vesper works: whether being spoken
to produces speech, whether interrupting it shuts it up, and whether it can tell
its own voice from yours.
"""

import time

import numpy as np
import pytest

from vesper.audio.speaker import Speaker
from vesper.audio.vad import EndpointConfig
from vesper.conversation import Conversation, ConversationConfig
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
    assert "how is the disk" in brain.asked[0]
    assert "Vesper" not in brain.asked[0]


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
    from vesper.brain.protocol import BrainError

    class FailingBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield BrainError("I lost my connection to Claude.")

    conv, _, _, voice, speaker, ui = build(transcripts=["Vesper hello"])
    conv.brain = FailingBrain()
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert voice.lines == ["I lost my connection to Claude."]
    assert ui.errors == ["I lost my connection to Claude."]


def test_permission_requests_reach_the_ui():
    from vesper.brain.protocol import PermissionNeeded, TurnComplete

    class AskingBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield PermissionNeeded("Write", "C:/notes.txt")
            yield TurnComplete(text="I need your okay to write that file.", turns=1)

    conv, _, _, voice, speaker, ui = build(transcripts=["Vesper save a note"])
    conv.brain = AskingBrain()
    conv._on_utterance(audio())
    settle(speaker)
    speaker.close()
    assert ui.permissions == [("Write", "C:/notes.txt")]
    assert voice.lines == ["I need your okay to write that file."]


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
