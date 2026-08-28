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


def test_preroll_is_prepended_so_the_first_word_survives(speech):
    endpointer = Endpointer(VoiceActivity())
    preroll = np.full(6400, 0.02, dtype=np.float32)  # 400ms marker
    captured = None
    for block in blocks(np.concatenate([speech, silence(1.5)])):
        result = endpointer.feed(block, preroll=preroll)
        if result is not None:
            captured = result
            break
    assert captured is not None
    assert len(captured) > len(speech) * 0.9
    # The marker value must appear at the very front of the utterance.
    assert np.allclose(captured[:100], 0.02)


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
