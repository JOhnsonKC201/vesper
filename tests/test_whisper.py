"""Speech to text.

The model is loaded once per module: it is the slow part, and reloading it per
test would make the suite unpleasant enough that people stop running it.
"""

import os
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


# --- which runtime gets asked about the gpu ---------------------------------
#
# faster-whisper does not use torch. It runs on CTranslate2, which is its own
# runtime with its own CUDA build. `_cuda_available` asked torch, and torch
# here is deliberately the cpu build (it is installed for Silero VAD and the
# voiceprint model), so the answer was "no gpu" on every machine forever,
# whatever card was in it. On this machine CTranslate2 reports one device and
# transcribes 11x faster than the cpu path it was being pinned to.


def test_the_gpu_question_is_put_to_ctranslate2_and_never_to_torch():
    import inspect

    from vesper.stt import accel, whisper as whisper_module

    source = inspect.getsource(whisper_module)
    assert "torch" not in source, (
        "faster-whisper runs on CTranslate2. Asking torch whether the gpu is "
        "usable is asking a library that does not do the work, and the cpu "
        "build of torch always answers no."
    )
    assert "ctranslate2" in inspect.getsource(accel)


def test_a_device_being_present_is_not_enough_to_be_trusted(monkeypatch):
    """The worst shape this bug can take.

    CTranslate2 reports a CUDA device on a machine with no CUDA runtime. The
    model then builds without complaint and every transcription afterwards
    raises. Load time success with inference time failure means Vesper starts
    fine at login and dies the first time you speak to it, with no console for
    the traceback to land in. So the probe is a real inference.
    """
    from vesper.stt import accel

    monkeypatch.setattr(accel, "device_count", lambda: 1)
    monkeypatch.setattr(
        accel, "probe", lambda model: "RuntimeError: cublas64_12.dll not found"
    )
    logged: list[str] = []
    stt = Listener(WhisperConfig(model="tiny.en", device="auto"), log=logged.append)
    stt.load()
    assert stt.resolved_device == "cpu"
    assert any("falling back to cpu" in line for line in logged)
    # And it still works, which is the entire point of falling back.
    assert stt.transcribe(load_wav("speech_short.wav")).ok


def test_a_cuda_that_cannot_even_build_falls_back_the_same_way(monkeypatch):
    """Refusing to build and failing the probe are one event to the listener.

    They are two different code paths and they used to log two different
    sentences, which meant a machine with no card at all (a CI runner, say)
    took the branch nobody had written the message for. Whoever is talking to
    Vesper does not care which half failed, only that it is on the cpu now.
    """
    from vesper.stt import accel, whisper as whisper_module

    real_build = whisper_module.Listener._build

    def refuse_cuda(self, device, compute):
        if device == "cuda":
            raise RuntimeError("no CUDA driver")
        return real_build(self, device, compute)

    monkeypatch.setattr(whisper_module.Listener, "_build", refuse_cuda)
    monkeypatch.setattr(accel, "device_count", lambda: 1)
    logged: list[str] = []
    stt = Listener(WhisperConfig(model="tiny.en", device="cuda"), log=logged.append)
    stt.load()
    assert stt.resolved_device == "cpu"
    assert any("cuda unusable" in line and "falling back to cpu" in line
               for line in logged), logged
    # And the fallback never carries the failed device's precision with it.
    assert stt.resolved_compute == "int8"


def test_a_cpu_model_that_will_not_build_is_a_real_failure(monkeypatch):
    """There is nowhere to fall back to, so it must not be swallowed."""
    from vesper.stt import whisper as whisper_module

    def always_fail(self, device, compute):
        raise RuntimeError("no model files")

    monkeypatch.setattr(whisper_module.Listener, "_build", always_fail)
    stt = Listener(WhisperConfig(model="tiny.en", device="cpu"), log=lambda m: None)
    with pytest.raises(RuntimeError, match="no model files"):
        stt.load()


def test_one_failed_transcription_does_not_end_the_assistant(monkeypatch):
    """An always on assistant started from a login shortcut has no console.

    An exception out of `transcribe` used to propagate through `_handle_block`
    and out of `run()`, which had only `except KeyboardInterrupt`. The failure
    that produces is Vesper going silent, permanently, at the moment you first
    speak to it.
    """
    stt = Listener(WhisperConfig(model="tiny.en", device="cpu"), log=lambda m: None)
    stt.load()

    class Exploding:
        def transcribe(self, *a, **k):
            raise RuntimeError("the gpu fell over")

    stt._model = Exploding()
    stt.resolved_device = "cpu"  # nothing left to fall back to
    result = stt.transcribe(load_wav("speech_short.wav"))
    assert result.ok is False
    assert result.rejected_reason == "transcription-failed"
    assert stt.failures == 1


def test_a_gpu_that_fails_mid_session_moves_to_the_cpu_and_keeps_going():
    """A driver reset should cost one utterance, not the rest of the day."""
    stt = Listener(WhisperConfig(model="tiny.en", device="cpu"), log=lambda m: None)
    stt.load()
    working = stt._model

    class ExplodesOnce:
        def transcribe(self, *a, **k):
            raise RuntimeError("device lost")

    stt._model = ExplodesOnce()
    stt.resolved_device = "cuda"  # pretend we were on the gpu
    stt._build = lambda device, compute: setattr(stt, "_model", working) or (
        setattr(stt, "resolved_device", device)
    )
    result = stt.transcribe(load_wav("speech_short.wav"))
    assert stt.resolved_device == "cpu"
    assert result.ok, "the retry on the cpu should have produced a real transcript"


def test_the_cuda_dll_search_is_idempotent():
    """`enable_dll_search` is called from the model load and the self check."""
    from vesper.stt import accel

    first = accel.enable_dll_search()
    before = os.environ["PATH"]
    second = accel.enable_dll_search()
    assert first == second
    assert os.environ["PATH"] == before, "PATH must not grow on every call"


def test_a_gpu_that_never_answers_is_treated_as_unusable():
    """A broken CUDA install does not reliably raise.

    A half uninstalled one was seen sitting inside `encode` rather than failing,
    and a startup that hangs is worse than the failure the probe exists to
    catch: an assistant launched from a login shortcut that never comes up, with
    nothing on screen to say why.
    """
    import time

    from vesper.stt import accel

    class NeverReturns:
        def transcribe(self, *args, **kwargs):
            time.sleep(30)
            return iter(()), None

    started = time.monotonic()
    problem = accel.probe(NeverReturns(), timeout_s=0.5)
    assert "did not answer" in problem
    assert time.monotonic() - started < 5, "the probe must not wait it out"


def test_a_working_model_probes_clean_and_a_broken_one_reports_why():
    from vesper.stt import accel

    class Works:
        def transcribe(self, *args, **kwargs):
            return iter(()), None

    class Broken:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found")

    assert accel.probe(Works()) == ""
    assert "cublas64_12" in accel.probe(Broken())
