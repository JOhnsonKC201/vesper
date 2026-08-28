"""Speech to text with faster-whisper, running locally.

Model weights are already on this machine in the HuggingFace cache (Echo Flow
put them there), so nothing downloads on first run.

Model choice is a latency decision, not a quality one. In conversation the reply
cannot start until transcription finishes, so a 200ms model that gets "what's my
battery" right beats a 3s model that also gets proper nouns right. base.en on
CPU int8 is the default; large-v3-turbo is available for dictation-grade work.

The hallucination filter is not optional. Whisper reliably invents "Thank you.",
"Thanks for watching!" and similar from near-silence, and a voice assistant that
answers phantom sentences is worse than one that mishears.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Whisper's greatest hits when handed silence or noise. Only applied to short
# clips, where a genuine utterance of this text is implausible.
HALLUCINATIONS = frozenset(
    phrase.lower()
    for phrase in [
        "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
        "you", "bye.", "bye", ".", "!", "?", "...", "Thank you for watching!",
        "please subscribe", "subtitles by the amara.org community",
        "transcription by castingwords", "www.mooji.org", "okay.", "oh.",
    ]
)


@dataclass
class Transcript:
    """What was heard, plus how confident the model was about it."""

    text: str
    language: str = "en"
    duration_s: float = 0.0
    latency_s: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    low_confidence_words: tuple[str, ...] = ()
    rejected_reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.rejected_reason

    @property
    def uncertain(self) -> bool:
        """Worth asking 'did you say...' rather than acting on it."""
        return self.avg_logprob < -1.0 or self.no_speech_prob > 0.6


@dataclass
class WhisperConfig:
    model: str = "base.en"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "en"
    beam_size: int = 5
    short_clip_s: float = 3.0
    min_duration_s: float = 0.35
    min_rms: float = 0.003
    word_confidence_floor: float = 0.55
    # Biases the decoder toward words this user actually says. Measured on
    # en_GB-alan speech: without it, "Vesper" came back as "Vespa" or "best
    # but" and the assistant simply never woke up. With it, three for three.
    initial_prompt: str | None = None


def wake_word_prompt(words, name: str = "") -> str:
    """Build an initial_prompt that makes Whisper expect the assistant's name."""
    named = [w.strip().title() for w in words if w and w.strip()]
    if name and name.title() not in named:
        named.insert(0, name.title())
    if not named:
        return ""
    if len(named) == 1:
        return f"Talking to an assistant named {named[0]}."
    return (
        f"Talking to an assistant named {named[0]}, also called "
        + ", ".join(named[1:])
        + "."
    )


class Listener:
    """Transcribes float32 mono audio at 16kHz."""

    def __init__(self, config: WhisperConfig | None = None, *, log=None) -> None:
        self.config = config or WhisperConfig()
        self._log = log or (lambda message: None)
        self._model = None
        self.resolved_model = ""
        self.resolved_device = ""

    # --- model --------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device = self.config.device
        compute = self.config.compute_type
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        started = time.monotonic()
        self._model = WhisperModel(self.config.model, device=device, compute_type=compute)
        self.resolved_model = self.config.model
        self.resolved_device = device
        self._log(
            f"whisper {self.config.model} on {device}/{compute} "
            f"loaded in {time.monotonic() - started:.2f}s"
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    # --- transcription ------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> Transcript:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / sample_rate

        gate = self._pre_gate(audio, duration)
        if gate:
            return Transcript(text="", duration_s=duration, rejected_reason=gate)

        self.load()
        started = time.monotonic()

        # Short clips do not benefit from beam search, and greedy decoding saves
        # 150-300ms, which is the difference between snappy and sluggish.
        beam = 1 if duration < self.config.short_clip_s else self.config.beam_size

        segments, info = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=beam,
            vad_filter=True,
            word_timestamps=True,
            # Without this, one bad transcription poisons every later one.
            condition_on_previous_text=False,
            initial_prompt=self.config.initial_prompt,
        )

        pieces: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        weak_words: list[str] = []
        for segment in segments:
            pieces.append(segment.text)
            logprobs.append(getattr(segment, "avg_logprob", 0.0) or 0.0)
            no_speech.append(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            for word in getattr(segment, "words", None) or []:
                if (word.probability or 1.0) < self.config.word_confidence_floor:
                    weak_words.append(word.word.strip())

        text = " ".join(p.strip() for p in pieces if p.strip()).strip()
        transcript = Transcript(
            text=text,
            language=getattr(info, "language", self.config.language or "en"),
            duration_s=duration,
            latency_s=time.monotonic() - started,
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            no_speech_prob=float(np.mean(no_speech)) if no_speech else 0.0,
            low_confidence_words=tuple(weak_words),
        )

        reason = self._post_gate(transcript)
        if reason:
            return Transcript(
                text="",
                duration_s=duration,
                latency_s=transcript.latency_s,
                rejected_reason=reason,
            )
        return transcript

    # --- gates --------------------------------------------------------------

    def _pre_gate(self, audio: np.ndarray, duration: float) -> str:
        if audio.size == 0:
            return "empty"
        if duration < self.config.min_duration_s:
            return "too-short"
        if float(np.sqrt(np.mean(audio**2))) < self.config.min_rms:
            return "too-quiet"
        return ""

    def _post_gate(self, transcript: Transcript) -> str:
        if not transcript.text:
            return "no-text"
        # Only treat stock phrases as hallucinations on short clips. On a long
        # clip "thank you" is probably a real thing the user said.
        if transcript.duration_s < 2.0:
            stripped = transcript.text.strip().lower()
            if stripped in HALLUCINATIONS:
                return "hallucination"
        return ""
