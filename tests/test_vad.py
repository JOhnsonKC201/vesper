"""Voice activity detection and endpointing.

Fixtures are real speech, synthesized locally by Piper and resampled to 16kHz.
Synthetic tones are not good enough here: Silero is trained on speech and scores
a sine wave near zero, so a tone-based test would pass with the model broken.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from vesper.audio.vad import (
    SILERO_WINDOW,
    EndpointConfig,
    Endpointer,
    VoiceActivity,
)

FIXTURES = Path(__file__).parent / "fixtures"
BLOCK = 480  # 30ms at 16kHz, matching Microphone


def load_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name), "rb") as w:
        assert w.getframerate() == 16000
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def blocks(audio: np.ndarray, size: int = BLOCK):
    for start in range(0, len(audio) - size + 1, size):
        yield audio[start : start + size]


def silence(seconds: float) -> np.ndarray:
    # Real rooms are never digitally silent; a tiny noise floor is more honest
    # and stops the energy gate from looking better than it is.
    rng = np.random.default_rng(0xC0FFEE)
    return (rng.standard_normal(int(16000 * seconds)) * 0.0005).astype(np.float32)


@pytest.fixture(scope="module")
def speech():
    return load_wav("speech_short.wav")


# --- VoiceActivity ----------------------------------------------------------


def test_silero_actually_loads():
    """If this fails everything below is silently testing the energy fallback."""
    assert VoiceActivity().using_neural is True


def test_silero_really_scores_frames_rather_than_falling_back(speech):
    """The Echo Flow bug: model loaded, never ran, RMS answered every call."""
    vad = VoiceActivity()
    for block in blocks(speech):
        vad.is_speech(block)
    assert vad.neural_frames > 0, "the neural path never executed"
    expected = len(speech) // SILERO_WINDOW
    assert vad.neural_frames >= expected - 2


def test_detects_real_speech(speech):
    vad = VoiceActivity()
    hits = sum(1 for block in blocks(speech) if vad.is_speech(block))
    total = len(list(blocks(speech)))
    assert hits / total > 0.5, f"only {hits}/{total} blocks scored as speech"


def test_rejects_room_silence():
    vad = VoiceActivity()
    quiet = silence(2.0)
    hits = sum(1 for block in blocks(quiet) if vad.is_speech(block))
    assert hits == 0


def test_handles_blocks_that_are_not_window_aligned(speech):
    """480-sample blocks never align to Silero's 512. The residual buffer is
    what makes the arithmetic work, and it is easy to get subtly wrong."""
    vad = VoiceActivity()
    for size in (480, 300, 1000, 512):
        for block in blocks(speech, size):
            vad.is_speech(block)
    assert vad.neural_frames > 0


def test_empty_block_is_not_speech():
    assert VoiceActivity().is_speech(np.zeros(0, dtype=np.float32)) is False


def test_energy_fallback_works_without_a_model(monkeypatch):
    vad = VoiceActivity()
    vad._model = None  # simulate silero failing to load
    assert vad.is_speech(np.full(BLOCK, 0.5, dtype=np.float32)) is True
    assert vad.is_speech(np.zeros(BLOCK, dtype=np.float32)) is False
    assert vad.fallback_frames == 2


def test_reset_clears_partial_window(speech):
    vad = VoiceActivity()
    vad.is_speech(speech[:100])
    vad.reset()
    assert vad._residual.size == 0


# --- Endpointer -------------------------------------------------------------


def run_endpointer(audio: np.ndarray, config: EndpointConfig | None = None):
    endpointer = Endpointer(VoiceActivity(), config or EndpointConfig())
    utterances = []
    for block in blocks(audio):
        result = endpointer.feed(block)
        if result is not None:
            utterances.append(result)
    tail = endpointer.flush()
    if tail is not None:
        utterances.append(tail)
    return utterances


def test_one_utterance_from_speech_surrounded_by_silence(speech):
    audio = np.concatenate([silence(0.5), speech, silence(1.5)])
    utterances = run_endpointer(audio)
    assert len(utterances) == 1
    duration = len(utterances[0]) / 16000
    assert 2.0 < duration < 4.5, f"captured {duration:.2f}s, expected around 2.5s"


def test_two_utterances_separated_by_a_long_pause(speech):
    audio = np.concatenate([silence(0.3), speech, silence(1.6), speech, silence(1.6)])
    assert len(run_endpointer(audio)) == 2


def test_a_short_pause_does_not_split_an_utterance(speech):
    """People pause mid-sentence. 400ms must not end the turn."""
    audio = np.concatenate([silence(0.3), speech, silence(0.4), speech, silence(1.5)])
    assert len(run_endpointer(audio)) == 1


def test_pure_silence_yields_nothing():
    assert run_endpointer(silence(4.0)) == []


def test_a_brief_blip_is_discarded(speech):
    """A door closing should not become a transcription request."""
    blip = speech[: int(16000 * 0.12)]
    audio = np.concatenate([silence(0.3), blip, silence(1.5)])
    assert run_endpointer(audio) == []


def contiguous_offset(stream: np.ndarray, piece: np.ndarray) -> int | None:
    """Where `piece` sits verbatim inside `stream`, or None if it never does.

    Utterances always open on a block boundary, so only those offsets are worth
    testing. Handing a static filler in as the lead-in, which is what the old
    version of the test below did, cannot detect a repeat: filler has nothing
    in common with the audio that gets repeated.
    """
    for start in range(0, len(stream) - len(piece) + 1, BLOCK):
        if np.array_equal(stream[start : start + len(piece)], piece):
            return start
    return None


def capture_one(stream: np.ndarray, config: EndpointConfig | None = None):
    endpointer = Endpointer(VoiceActivity(), config)
    for block in blocks(stream):
        result = endpointer.feed(block)
        if result is not None:
            return result
    return None


def test_an_utterance_is_a_verbatim_run_of_what_was_fed(speech):
    """No sample may be repeated, dropped or reordered on the way out.

    The lead-in used to be read from the microphone's own ring, which is
    appended on every captured block whether it is speech or not. By the time
    an utterance was confirmed that ring's tail held the same audio as the
    candidate blocks, and feed() concatenated both. Every utterance, and every
    enrolment clip, opened with its first 120ms played twice. A duplicated
    onset is not a contiguous run, so this is the guard against its return.
    """
    stream = np.concatenate([silence(0.8), speech, silence(1.5)])
    captured = capture_one(stream)
    assert captured is not None
    assert contiguous_offset(stream, captured) is not None


def test_the_lead_in_is_kept_so_the_first_word_survives(speech):
    """Whole words live in the moments before the endpointer is sure."""
    lead_in = silence(0.8)
    stream = np.concatenate([lead_in, speech, silence(1.5)])
    captured = capture_one(stream)
    assert captured is not None
    offset = contiguous_offset(stream, captured)
    assert offset is not None
    # It opens before the speech does, keeping real audio from before it.
    assert offset < len(lead_in)
    assert len(lead_in) - offset >= 0.15 * 16000


def test_a_discarded_candidate_stays_in_the_lead_in(speech):
    """A blip that fails to become speech is still audio that came before.

    Dropping it would punch a hole in the lead-in, and the utterance would stop
    being a verbatim run of the stream.
    """
    blip = speech[: int(16000 * 0.06)]
    stream = np.concatenate([silence(0.4), blip, silence(0.3), speech, silence(1.5)])
    captured = capture_one(stream)
    assert captured is not None
    assert contiguous_offset(stream, captured) is not None


def test_max_duration_forces_a_cut(speech):
    long_audio = np.concatenate([speech] * 6)  # about 15s of continuous speech
    config = EndpointConfig(max_utterance_s=3.0, end_silence_ms=700)
    utterances = run_endpointer(long_audio, config)
    assert len(utterances) >= 2
    for utterance in utterances:
        assert len(utterance) / 16000 <= 3.6


def test_end_silence_is_configurable(speech):
    quick = EndpointConfig(end_silence_ms=300)
    slow = EndpointConfig(end_silence_ms=1200)
    audio = np.concatenate([silence(0.3), speech, silence(0.7), speech, silence(1.5)])
    assert len(run_endpointer(audio, quick)) == 2
    assert len(run_endpointer(audio, slow)) == 1


def test_reset_abandons_a_partial_utterance(speech):
    endpointer = Endpointer(VoiceActivity())
    for block in list(blocks(speech))[:20]:
        endpointer.feed(block)
    assert endpointer.collecting is True
    endpointer.reset()
    assert endpointer.collecting is False
    assert endpointer.flush() is None


# --- looking early, so transcription happens during the wait ----------------
#
# The endpointer waits 700ms to be sure you have stopped. Transcription does not
# need that wait: whatever was said before the pause is already final. So the
# caller gets a prefix at 250ms and spends the remaining 450ms transcribing it,
# which takes the whole of Whisper off the critical path when nothing more is
# said. These pin the contract that makes that safe to rely on.


def _endpointer(**kwargs):
    settings = dict(start_speech_ms=60, end_silence_ms=700, min_utterance_ms=100,
                    early_silence_ms=250)
    settings.update(kwargs)
    return Endpointer(VoiceActivity(), EndpointConfig(**settings))


def _blocks(endpointer, audio, size=512):
    """Feed an array in blocks, returning any completed utterance."""
    done = None
    for start in range(0, audio.size, size):
        out = endpointer.feed(audio[start:start + size])
        if out is not None:
            done = out
    return done


def test_nothing_to_look_at_before_anything_is_said(speech):
    endpointer = _endpointer()
    assert endpointer.peek() is None


def test_nothing_to_look_at_while_you_are_still_talking(speech):
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    assert endpointer.collecting
    assert endpointer.peek() is None


def test_a_prefix_arrives_once_the_pause_is_long_enough(speech):
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))

    early = endpointer.peek()
    assert early is not None
    assert early.size > 0


def test_the_prefix_is_offered_only_once(speech):
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))

    assert endpointer.peek() is not None
    assert endpointer.peek() is None


def test_the_prefix_is_a_verbatim_prefix_of_the_real_utterance(speech):
    """The whole saving rests on this: a transcript of the prefix is the real
    transcript when nothing more is said, so it has to be the same audio."""
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))
    early = endpointer.peek()

    final = _blocks(endpointer, np.zeros(int(16000 * 0.6), dtype=np.float32))
    assert final is not None
    assert final.size >= early.size
    assert np.array_equal(final[:early.size], early)


def test_silence_after_the_prefix_means_the_prefix_was_the_whole_thing(speech):
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))
    assert endpointer.peek() is not None

    assert _blocks(endpointer, np.zeros(int(16000 * 0.6), dtype=np.float32)) is not None
    assert endpointer.peek_was_final is True


def test_talking_again_after_the_prefix_invalidates_it(speech):
    """Pausing mid thought must not answer half a sentence."""
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))
    assert endpointer.peek() is not None

    _blocks(endpointer, speech[:16000])
    assert _blocks(endpointer, np.zeros(int(16000 * 0.9), dtype=np.float32)) is not None
    assert endpointer.peek_was_final is False


def test_an_utterance_with_no_early_look_is_never_reported_as_final(speech):
    """peek_was_final must mean "the prefix held", not "nobody looked"."""
    endpointer = _endpointer(early_silence_ms=0)
    _blocks(endpointer, speech[:16000])
    assert _blocks(endpointer, np.zeros(int(16000 * 0.9), dtype=np.float32)) is not None
    assert endpointer.peek_was_final is False


def test_zero_disables_the_early_look_entirely(speech):
    endpointer = _endpointer(early_silence_ms=0)
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.5), dtype=np.float32))
    assert endpointer.peek() is None


def test_the_early_look_does_not_change_what_an_utterance_is(speech):
    """Same audio in, same utterance out, whether or not anybody looked early."""
    with_peek = _endpointer()
    _blocks(with_peek, speech[:16000])
    _blocks(with_peek, np.zeros(int(16000 * 0.4), dtype=np.float32))
    with_peek.peek()
    first = _blocks(with_peek, np.zeros(int(16000 * 0.6), dtype=np.float32))

    without = _endpointer(early_silence_ms=0)
    _blocks(without, speech[:16000])
    _blocks(without, np.zeros(int(16000 * 0.4), dtype=np.float32))
    second = _blocks(without, np.zeros(int(16000 * 0.6), dtype=np.float32))

    assert first is not None and second is not None
    assert np.array_equal(first, second)


def test_a_new_utterance_starts_with_a_clean_slate(speech):
    endpointer = _endpointer()
    _blocks(endpointer, speech[:16000])
    _blocks(endpointer, np.zeros(int(16000 * 0.4), dtype=np.float32))
    endpointer.peek()
    _blocks(endpointer, np.zeros(int(16000 * 0.6), dtype=np.float32))

    _blocks(endpointer, speech[:16000])
    assert endpointer.peek() is None, "the new utterance inherited the old peek"
