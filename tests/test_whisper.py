"""Speech to text.

The model is loaded once per module: it is the slow part, and reloading it per
test would make the suite unpleasant enough that people stop running it.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from vesper.stt.whisper import HALLUCINATIONS, Listener, Transcript, WhisperConfig

FIXTURES = Path(__file__).parent / "fixtures"


def load_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def listener():
    # tiny.en keeps the suite fast; accuracy on these fixtures is identical.
    stt = Listener(WhisperConfig(model="tiny.en"))
    stt.load()
    return stt


# --- gates (no model needed) ------------------------------------------------


def test_empty_audio_is_rejected_without_loading_a_model():
    stt = Listener()
    result = stt.transcribe(np.zeros(0, dtype=np.float32))
    assert result.rejected_reason == "empty"
    assert result.ok is False
    assert stt._model is None, "the gate must run before the model loads"


def test_very_short_audio_is_rejected():
    stt = Listener()
    audio = np.random.default_rng(1).standard_normal(1600).astype(np.float32) * 0.5
    assert stt.transcribe(audio).rejected_reason == "too-short"


def test_near_silence_is_rejected_before_transcription():
    """Whisper invents sentences from silence. Never let it try."""
    stt = Listener()
    quiet = (np.random.default_rng(2).standard_normal(16000) * 0.0005).astype(np.float32)
    assert stt.transcribe(quiet).rejected_reason == "too-quiet"


def test_hallucination_list_is_lowercase_for_matching():
    assert all(phrase == phrase.lower() for phrase in HALLUCINATIONS)
    assert "thank you." in HALLUCINATIONS


# --- transcript helpers -----------------------------------------------------


def test_transcript_ok_requires_text_and_no_rejection():
    assert Transcript(text="hello").ok is True
    assert Transcript(text="").ok is False
    assert Transcript(text="hi", rejected_reason="hallucination").ok is False


def test_uncertain_flags_low_confidence():
    assert Transcript(text="x", avg_logprob=-1.8).uncertain is True
    assert Transcript(text="x", no_speech_prob=0.9).uncertain is True
    assert Transcript(text="x", avg_logprob=-0.3, no_speech_prob=0.05).uncertain is False


# --- real transcription -----------------------------------------------------


def test_transcribes_a_short_question(listener):
    result = listener.transcribe(load_wav("speech_short.wav"))
    assert result.ok
    words = result.text.lower()
    for expected in ("running", "computer", "right now"):
        assert expected in words, f"missing {expected!r} in {result.text!r}"


def test_transcribes_a_longer_sentence(listener):
    result = listener.transcribe(load_wav("speech_long.wav"))
    assert result.ok
    words = result.text.lower()
    for expected in ("build", "minutes", "test", "push"):
        assert expected in words, f"missing {expected!r} in {result.text!r}"


def test_transcription_is_fast_enough_for_conversation(listener):
    """Faster than realtime, or the assistant feels laggy no matter what else."""
    result = listener.transcribe(load_wav("speech_short.wav"))
    assert result.latency_s < result.duration_s, (
        f"asr took {result.latency_s:.2f}s for {result.duration_s:.2f}s of audio"
    )


def test_confidence_is_high_on_clean_speech(listener):
    result = listener.transcribe(load_wav("speech_short.wav"))
    assert result.uncertain is False
    assert result.avg_logprob > -1.0


def test_model_and_device_are_reported(listener):
    assert listener.resolved_model == "tiny.en"
    assert listener.resolved_device in ("cpu", "cuda")


def test_repeated_calls_do_not_reload_the_model(listener):
    model = listener._model
    listener.transcribe(load_wav("speech_short.wav"))
    assert listener._model is model
