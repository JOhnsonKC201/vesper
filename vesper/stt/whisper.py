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

import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import accel

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
    # Cores CTranslate2 may use while transcribing. Left unset it takes every
    # physical core, and although a transcription is only about half a second,
    # that half second is the one moment an always-on assistant can make the
    # machine feel slow. 0 restores the take-everything default.
    cpu_threads: int = 4
    language: str | None = "en"
    beam_size: int = 5
    short_clip_s: float = 3.0
    min_duration_s: float = 0.35
    min_rms: float = 0.003
    word_confidence_floor: float = 0.55
    # The model's own confidence, which was being measured and then ignored.
    # An always-on assistant hands Whisper near-silence all day, and it answers
    # with fluent invented English rather than with nothing.
    max_no_speech: float = 0.72
    min_logprob: float = -1.15
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


def _is_repetitive(text: str) -> bool:
    """Does this look like the decoder stuck in a loop rather than speech?

    Both patterns are from a real log: "Choo, choo, choo, choo, choo." and
    "I don't know if you can hear me, but I don't know if you can hear me."
    People repeat themselves, but not like this, and the cost of being wrong is
    one missed sentence against a turn spent on nothing.
    """
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w]
    if len(words) < 4:
        return False

    # The same word four times over, consecutively.
    run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        if run >= 4:
            return True

    # Or a phrase said twice with almost nothing else in between.
    if len(words) >= 8 and len(set(words)) / len(words) <= 0.42:
        return True

    half = len(words) // 2
    if len(words) >= 8 and words[:half] == words[half:half * 2]:
        return True

    # Or the same run of words appearing twice with a joining word wedged in,
    # which is the shape the decoder actually produced: "I don't know if you
    # can hear me, but I don't know if you can hear me." A whole clause said
    # twice is the model looping, not a person speaking.
    longest = _longest_repeated_run(words)
    return longest >= 4 and (2 * longest) / len(words) >= 0.6


def _longest_repeated_run(words: list[str]) -> int:
    """Length of the longest word sequence that occurs more than once."""
    seen: dict[tuple[str, ...], int] = {}
    longest = 0
    # Bounded so a long dictation cannot make this quadratic in a hot path.
    for size in range(4, min(len(words) // 2, 12) + 1):
        seen.clear()
        for start in range(len(words) - size + 1):
            gram = tuple(words[start : start + size])
            if gram in seen and start - seen[gram] >= size:
                longest = max(longest, size)
                break
            seen.setdefault(gram, start)
    return longest

class Listener:
    """Transcribes float32 mono audio at 16kHz."""

    def __init__(self, config: WhisperConfig | None = None, *, log=None) -> None:
        self.config = config or WhisperConfig()
        self._log = log or (lambda message: None)
        self._model = None
        self.resolved_model = ""
        self.resolved_device = ""
        self.resolved_compute = ""
        # Surfaced by --check and the dashboard: a silent assistant with a
        # rising failure count is a very different bug from a deaf one.
        self.failures = 0

    # --- model --------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        device = self.config.device
        if device == "auto":
            device = "cuda" if accel.device_count() else "cpu"
        if device == "cuda":
            # Before the model is built, not after: CTranslate2 resolves its
            # CUDA libraries at construction, and PATH is what it reads.
            accel.enable_dll_search()

        compute = self.config.compute_type
        if compute == "auto":
            compute = accel.best_compute_type(device)

        try:
            self._build(device, compute)
        except Exception as exc:
            # An explicit `device: cuda` in the config on a machine that cannot
            # do it is a mistake to report, not a reason to have no ears.
            if device == "cpu":
                raise
            self._log(f"whisper could not build on {device}: {type(exc).__name__}: {exc}")
            self._build("cpu", accel.best_compute_type("cpu"))
            return

        # A model that built on the GPU has proved nothing yet. If the runtime
        # is not really there, every transcription raises instead, and the
        # first one is the first time you speak to it.
        if device == "cuda":
            problem = accel.probe(self._model)
            if problem:
                self._log(f"whisper cuda unusable ({problem}), falling back to cpu")
                hint = accel.missing_runtime_hint()
                if hint:
                    self._log(f"whisper: {hint}")
                self._build(
                    "cpu",
                    accel.best_compute_type("cpu")
                    if self.config.compute_type == "auto"
                    else self.config.compute_type,
                )

    def _build(self, device: str, compute: str) -> None:
        """Construct the model on one device and record what actually happened."""
        from faster_whisper import WhisperModel

        started = time.monotonic()
        self._model = WhisperModel(
            self.config.model,
            device=device,
            compute_type=compute,
            cpu_threads=max(0, self.config.cpu_threads),
        )
        self.resolved_model = self.config.model
        self.resolved_device = device
        self.resolved_compute = compute
        self._log(
            f"whisper {self.config.model} on {device}/{compute} "
            f"loaded in {time.monotonic() - started:.2f}s"
        )

    def _fall_back_to_cpu(self, reason: str) -> bool:
        """Move to the CPU after the GPU failed mid session. True if it moved.

        A driver reset or a card taken by something else should cost one
        utterance, not the rest of the day. Only ever called when the current
        device is cuda, so it cannot loop.
        """
        if self.resolved_device != "cuda":
            return False
        self._log(f"whisper gpu failed ({reason}), moving to cpu for this session")
        try:
            self._build("cpu", accel.best_compute_type("cpu"))
        except Exception as exc:  # nothing left to fall back to
            self._log(f"whisper cpu reload failed: {type(exc).__name__}: {exc}")
            return False
        return True

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

        pieces: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        weak_words: list[str] = []
        try:
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

            # Inside the try on purpose. `transcribe` hands back a generator and
            # does no work until it is drawn from, so this loop is where a
            # missing CUDA library or a driver reset actually raises.
            for segment in segments:
                pieces.append(segment.text)
                logprobs.append(getattr(segment, "avg_logprob", 0.0) or 0.0)
                no_speech.append(getattr(segment, "no_speech_prob", 0.0) or 0.0)
                for word in getattr(segment, "words", None) or []:
                    if (word.probability or 1.0) < self.config.word_confidence_floor:
                        weak_words.append(word.word.strip())
        except Exception as exc:
            # Never let this reach the audio loop. An assistant started at login
            # has no console, so an exception here is not a traceback anyone
            # reads, it is Vesper going silent the moment you first speak to it.
            reason = f"{type(exc).__name__}: {exc}"
            if self._fall_back_to_cpu(reason):
                return self.transcribe(audio, sample_rate)
            self._log(f"transcription failed: {reason}")
            self.failures += 1
            return Transcript(
                text="",
                duration_s=duration,
                latency_s=time.monotonic() - started,
                rejected_reason="transcription-failed",
            )

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
        """Reject what the model invented rather than heard.

        Whisper does not return nothing when handed near-silence. It returns
        confident, well-formed English, and an always-on assistant feeds it
        near-silence all day. Every check below comes from a real session log.
        """
        if not transcript.text:
            return "no-text"

        stripped = transcript.text.strip().lower()

        # The worst one, because it wakes him. Handed quiet room tone, Whisper
        # echoes its own initial_prompt back as your speech: "Talking to an
        # assistant named Vesper, also called Jarvis." That contains the wake
        # word, so it woke Vesper and spent a turn on a sentence nobody said.
        if self._echoes_the_prompt(stripped):
            return "prompt-echo"

        # Loops. "Choo, choo, choo, choo, choo." and "one more, one more, one
        # more, one more" are what the decoder does when there is nothing to
        # decode. Real speech does not repeat one token five times.
        if _is_repetitive(stripped):
            return "looping"

        # The model's own opinion, which was being collected and then ignored.
        if transcript.no_speech_prob > self.config.max_no_speech:
            return "silence"
        if transcript.avg_logprob < self.config.min_logprob:
            return "low-confidence"

        # Stock filler, only on short clips. On a long one "thank you" is
        # probably a real thing somebody said.
        if transcript.duration_s < 2.0 and stripped in HALLUCINATIONS:
            return "hallucination"
        return ""

    def _echoes_the_prompt(self, stripped: str) -> bool:
        """Is this the initial_prompt coming back at us?"""
        prompt = (self.config.initial_prompt or "").strip().lower()
        if not prompt:
            return False
        cleaned = stripped.strip(" .,!?")
        target = prompt.strip(" .,!?")
        if cleaned == target:
            return True
        # It also comes back truncated, so a long prefix of the prompt counts.
        return len(cleaned) >= 18 and target.startswith(cleaned)
