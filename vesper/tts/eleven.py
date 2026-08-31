"""ElevenLabs speech, with Piper underneath it at all times.

## Why this is a router and not a fourth backend

The obvious shape is a fourth `engine` alongside piper, sapi and none, with the
dashboard swapping `Speaker.voice` when you pick a different voice. That is
wrong for two reasons.

`Speaker.voice` is read inside the worker thread at `audio/speaker.py:130`.
Swapping it while the worker is inside `voice.speak()` means coordinating a
handover and calling `close()` on a backend that is mid utterance. Here,
changing voice is assigning `self.voice_id`, a plain string read once at the
top of the next `speak()`. There is no handover to get wrong.

More importantly, a cloud voice that can fail must have a local voice behind
it, permanently, not as a startup fallback. `conversation.py` documents that
the local commands have to work with the network down: "Quiet from now on."
and "Shutting down." are handled without touching Claude precisely so they
survive an outage. If those went through a network call, an outage would take
away the ability to shut the assistant up. So Piper is a member of this class,
not an alternative to it.

## The order

    1. cache hit         -> play from disk. free, instant, works offline
    2. no key, no budget,
       or any failure    -> Piper
    3. otherwise         -> stream, play, cache, count

## One deliberate rough edge

If the stream fails *after* audio has started playing, this does not fall back.
Replaying the sentence from the beginning in a different voice is worse than
stopping short. Failures before the first sample fall back cleanly, which is
the case that actually happens: a refused key, an exhausted budget, no network.

## What leaves the machine

The text of what Vesper says, which can contain the contents of your files,
because he has read access to the whole drive. Microphone audio never does.
Setting `voice.engine` back to `piper` stops all of it.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import eleven_api
from .base import Voice
from .budget import Budget
from .cache import AudioCache
from .eleven_api import DEFAULT_MODEL, SAMPLE_RATE, ElevenClient, ElevenError

# Matches Piper. 1024 frames at 22050Hz is 46ms, which is the barge-in latency
# the rest of the system is written around.
_BLOCK_FRAMES = 1024

def align(chunks):
    """Turn arbitrarily split PCM bytes into whole int16 samples.

    Its own function rather than three lines inline, because the bug it
    prevents is invisible. HTTP chunk boundaries fall wherever the network puts
    them, and one of them landing inside a two-byte sample shifts every sample
    after it by a byte. That does not raise anything: it plays as noise, and
    only for the unlucky utterances where a boundary happened to land odd. A
    seam here is what lets a test drive the odd case on purpose.
    """
    remainder = b""
    for chunk in chunks:
        buffer = remainder + chunk
        usable = len(buffer) - (len(buffer) % 2)
        remainder = buffer[usable:]
        if usable:
            yield np.frombuffer(buffer[:usable], dtype="<i2")


# Lines Vesper says over and over. Worth having in the good voice permanently,
# and they cost about 350 characters to put there, once.
STOCK_PHRASES: tuple[str, ...] = (
    "Yes?",
    "Doing it.",
    "Leaving it.",
    "Quiet from now on.",
    "Back on.",
    "Shutting down.",
    "Vesper here. I'm listening.",
    "Let me look.",
    "One moment.",
    "Checking now.",
    "Hang on.",
    "Let me check that.",
    "Give me a second.",
    "Still looking.",
    "Bear with me.",
    "Almost there.",
    "I'm not keeping copies, so there's nothing to put back.",
)


class ElevenTTS:
    """A `Voice` that prefers ElevenLabs and falls back to Piper."""

    def __init__(
        self,
        fallback: Voice,
        *,
        client: ElevenClient,
        cache: AudioCache | None = None,
        budget: Budget | None = None,
        voice_id: str = eleven_api.DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL,
        log=None,
    ) -> None:
        self.fallback = fallback
        self.client = client
        self.cache = cache or AudioCache(None)
        self.budget = budget or Budget(None)
        # Read at the top of each speak(), so the dashboard changes voice by
        # assigning this and nothing else has to be coordinated.
        self.voice_id = voice_id or eleven_api.DEFAULT_VOICE_ID
        self.model_id = model_id or DEFAULT_MODEL
        self._log = log or (lambda message: None)

        self.name = f"eleven:{eleven_api.voice_named(self.voice_id)}"
        self.sample_rate = SAMPLE_RATE
        self._stream = None
        self._lock = threading.Lock()

        # Why the last utterance did not use the cloud, for the dashboard.
        self.last_reason = ""
        self.cloud_utterances = 0
        self.fallback_utterances = 0

    # --- voice selection ----------------------------------------------------

    def use(self, voice_id: str) -> None:
        """Switch voice. Safe to call from any thread, including mid sentence.

        Assignment of a str is atomic under the GIL, and `speak` reads it once
        at the top, so the worst case is that a sentence already in flight
        finishes in the old voice. That is the correct behaviour anyway.
        """
        if not voice_id:
            return
        self.voice_id = voice_id
        self.name = f"eleven:{eleven_api.voice_named(voice_id)}"

    # --- Voice protocol -----------------------------------------------------

    def speak(self, text: str, stop: threading.Event) -> None:
        self.say_as(text, self.voice_id, stop)

    def say_as(self, text: str, voice_id: str, stop: threading.Event) -> None:
        """Speak in a named voice, whatever the current one is.

        The current voice is just the default argument. The dashboard's preview
        button uses this to let you hear a candidate without committing to it,
        and it shares `self._lock` with normal speech, so a preview queues
        behind whatever Vesper is already saying rather than talking over it.
        """
        text = (text or "").strip()
        voice_id = voice_id or self.voice_id
        if not text or stop.is_set():
            return

        cached = self.cache.get(voice_id, text)
        if cached:
            self.last_reason = "cache"
            self._play(cached, stop)
            return

        blocked = self._why_not(text)
        if blocked:
            self._fall_back(text, stop, blocked)
            return

        try:
            self._stream_and_play(text, voice_id, stop)
        except ElevenError as exc:
            self._fall_back(text, stop, exc.reason)
        except Exception as exc:
            # Deliberately broad. `Speaker._run` would catch this and route it
            # to on_error, so the process survives either way, but the sentence
            # would be silently dropped rather than spoken. Anything at all
            # going wrong out here has to end with Piper saying the line.
            self._fall_back(text, stop, type(exc).__name__)

    def _fall_back(self, text: str, stop: threading.Event, reason: str) -> None:
        self.last_reason = reason
        self._log(f"elevenlabs: {reason}, using piper")
        self.fallback_utterances += 1
        self.fallback.speak(text, stop)

    def close(self) -> None:
        with self._lock:
            self._teardown()
        try:
            self.fallback.close()
        except Exception:
            pass

    # --- decisions ----------------------------------------------------------

    def _why_not(self, text: str) -> str:
        """Reason to skip the network entirely, or empty to go ahead."""
        if not self.client.configured:
            return "no-key"
        if not self.budget.allows(len(text)):
            return "budget"
        return ""

    # --- the network path ---------------------------------------------------

    def _stream_and_play(self, text: str, voice_id: str, stop: threading.Event) -> None:
        """Play as the audio arrives, and cache the whole thing afterwards.

        Playing while receiving is the point: waiting for the complete response
        would add the whole synthesis time to the first word, which is the one
        number an assistant is judged on.
        """
        collected = bytearray()
        started = False

        def arriving():
            for chunk in self.client.stream_pcm(
                text, voice_id, model_id=self.model_id
            ):
                if stop.is_set():
                    return
                collected.extend(chunk)
                yield chunk

        with self._lock:
            stream = self._ensure_stream()
            try:
                for samples in align(arriving()):
                    for start in range(0, len(samples), _BLOCK_FRAMES):
                        if stop.is_set():
                            break
                        stream.write(samples[start : start + _BLOCK_FRAMES])
                        started = True
            except ElevenError:
                # Nothing has played yet, so the caller can still fall back
                # cleanly. Once audio has started, a fallback would replay the
                # sentence from the top in a different voice, which is worse
                # than stopping short.
                if not started:
                    raise
                self._log("elevenlabs: stream cut short mid sentence")
            except Exception:
                # A dead output device, most likely. Drop the stream so the
                # next utterance reopens it, exactly as PiperTTS does.
                self._teardown()
                if not started:
                    raise
                self._log("elevenlabs: playback failed mid sentence")
                return

            if stop.is_set():
                # Drop what the device still holds, so the cut is immediate
                # rather than one buffer late.
                self._teardown()
                return

        # Only a complete, uninterrupted utterance is worth keeping, and only a
        # complete one should be charged for.
        if collected:
            self.budget.spend(len(text))
            self.cloud_utterances += 1
            self.cache.put(voice_id, text, bytes(collected))
            self.last_reason = "cloud"

    def _play(self, pcm: bytes, stop: threading.Event) -> None:
        """Play PCM already on hand, in the same interruptible blocks."""
        usable = len(pcm) - (len(pcm) % 2)
        if not usable:
            return
        samples = np.frombuffer(pcm[:usable], dtype="<i2")
        with self._lock:
            stream = self._ensure_stream()
            try:
                for start in range(0, len(samples), _BLOCK_FRAMES):
                    if stop.is_set():
                        break
                    stream.write(samples[start : start + _BLOCK_FRAMES])
            except Exception:
                self._teardown()
                raise
            if stop.is_set():
                self._teardown()

    # --- audio device -------------------------------------------------------

    def _ensure_stream(self):
        import sounddevice as sd

        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=_BLOCK_FRAMES,
            )
            self._stream.start()
        return self._stream

    def _teardown(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass

    # --- warming ------------------------------------------------------------

    def prewarm(self, phrases: tuple[str, ...] = STOCK_PHRASES) -> int:
        """Synthesize the stock phrases into the cache. Returns how many.

        Run on a daemon thread at startup so it never delays the first reply.
        Skips anything already cached, so a restart costs nothing.
        """
        if not self.client.configured or not self.cache.enabled:
            return 0

        done = 0
        for phrase in phrases:
            if self.cache.has(self.voice_id, phrase):
                continue
            if not self.budget.allows(len(phrase)):
                break
            collected = bytearray()
            try:
                for chunk in self.client.stream_pcm(
                    phrase, self.voice_id, model_id=self.model_id
                ):
                    collected.extend(chunk)
            except ElevenError as exc:
                self._log(f"elevenlabs prewarm stopped: {exc.reason}")
                break
            if collected:
                self.budget.spend(len(phrase))
                self.cache.put(self.voice_id, phrase, bytes(collected))
                done += 1
        if done:
            self._log(f"elevenlabs: {done} stock phrases cached in {self.name}")
        return done

    def prewarm_async(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._prewarm_quietly, daemon=True, name="eleven-prewarm"
        )
        thread.start()
        return thread

    def _prewarm_quietly(self) -> None:
        try:
            self.prewarm()
        except Exception:
            # A failed warm-up is a slower first reply, never a crash.
            pass

    # --- reporting ----------------------------------------------------------

    def status(self) -> str:
        """One scalar line for the dashboard snapshot."""
        name = eleven_api.voice_named(self.voice_id)
        if not self.client.configured:
            return f"piper (no ElevenLabs key)"
        if self.budget.remaining <= 0:
            return f"piper (ElevenLabs budget spent)"
        return f"{name} ({self.budget.remaining:,} characters left)"


def build(
    fallback: Voice,
    *,
    api_key: str,
    voice_id: str,
    model_id: str = DEFAULT_MODEL,
    cache_dir: Path | None,
    budget_path: Path | None,
    monthly_characters: int,
    timeout_s: float,
    log=None,
) -> ElevenTTS:
    """Assemble the backend from plain config values."""
    return ElevenTTS(
        fallback,
        client=ElevenClient(api_key, timeout_s=timeout_s),
        cache=AudioCache(cache_dir),
        budget=Budget(budget_path, cap=monthly_characters),
        voice_id=voice_id,
        model_id=model_id,
        log=log,
    )
