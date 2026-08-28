"""Voice activity detection and utterance endpointing.

Silero v5 and later accept exactly 512 samples per call at 16kHz (256 at 8kHz).
Anything else raises. Echo Flow shipped for months feeding it 2048-sample slices
inside a try/except, so the model loaded, never once ran, and every decision
silently came from the RMS fallback. That class of bug is invisible in testing
because the fallback works well enough in a quiet room. Hence: an explicit
window size, an explicit counter of how many frames the neural path actually
scored, and a test that asserts it is non-zero.

Unlike a dictation endpointer, this one keeps Silero's LSTM state across frames
within an utterance (that is what the model is trained for) and resets only at
utterance boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SILERO_WINDOW = 512  # samples, at 16kHz. Not negotiable.


class VoiceActivity:
    """Silero VAD with an energy fallback.

    `neural_frames` counts frames actually scored by the model. If it stays at
    zero while audio is flowing, the neural path is broken and you are running
    on RMS without knowing it.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        energy_threshold: float = 0.01,
        on_warning=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_threshold = energy_threshold
        self._warn = on_warning or (lambda message: None)

        self.neural_frames = 0
        self.fallback_frames = 0
        self._warned = False
        self._residual = np.zeros(0, dtype=np.float32)

        self._model = None
        self._torch = None
        try:
            import torch
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            self._torch = torch
        except Exception as exc:
            self._warn(f"silero unavailable, falling back to energy gate: {exc}")

    @property
    def using_neural(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        """Clear LSTM state and any partial window. Call between utterances."""
        self._residual = np.zeros(0, dtype=np.float32)
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    def is_speech(self, block: np.ndarray) -> bool:
        """Score one block of audio. Blocks need not be window-aligned."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return False
        if self._model is None:
            return self._energy(block)

        self._residual = np.concatenate([self._residual, block])
        window_count = self._residual.size // SILERO_WINDOW
        if window_count == 0:
            # Not enough audio yet for even one window. Do not guess; report the
            # energy gate so short blocks still respond.
            return self._energy(block)

        used = window_count * SILERO_WINDOW
        windows = self._residual[:used].reshape(window_count, SILERO_WINDOW)
        self._residual = self._residual[used:]

        best = 0.0
        try:
            for window in windows:
                tensor = self._torch.from_numpy(window).unsqueeze(0)
                best = max(best, float(self._model(tensor, self.sample_rate).item()))
            self.neural_frames += window_count
            return best > self.threshold
        except Exception as exc:
            if not self._warned:
                self._warn(f"silero scoring failed, using energy gate: {exc}")
                self._warned = True
            return self._energy(block)

    def _energy(self, block: np.ndarray) -> bool:
        self.fallback_frames += 1
        return float(np.sqrt(np.mean(block**2))) > self.energy_threshold


@dataclass
class EndpointConfig:
    """Timing for turn-taking.

    `end_silence_ms` is the single most felt number in the whole assistant. Too
    short and it cuts you off while you think; too long and every exchange has a
    dead beat in it. 700ms is noticeably snappier than a dictation tool's 1500ms
    and still tolerates a mid-sentence pause.
    """

    start_speech_ms: int = 120
    end_silence_ms: int = 700
    min_utterance_ms: int = 250
    max_utterance_s: float = 30.0
    preroll_ms: int = 400


class Endpointer:
    """Turns a stream of audio blocks into complete utterances.

    States: idle -> collecting -> (emit) -> idle. Feed blocks; when an utterance
    completes, feed() returns it as one float32 array.
    """

    def __init__(
        self,
        vad: VoiceActivity,
        config: EndpointConfig | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        self.vad = vad
        self.config = config or EndpointConfig()
        self.sample_rate = sample_rate

        self._collecting = False
        self._chunks: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._collected_ms = 0.0
        self._candidate: list[np.ndarray] = []

    @property
    def collecting(self) -> bool:
        return self._collecting

    def reset(self) -> None:
        self._collecting = False
        self._chunks.clear()
        self._candidate.clear()
        self._speech_ms = self._silence_ms = self._collected_ms = 0.0
        self.vad.reset()

    def feed(self, block: np.ndarray, preroll: np.ndarray | None = None) -> np.ndarray | None:
        """Add one block. Returns a complete utterance, or None."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None

        block_ms = 1000.0 * block.size / self.sample_rate
        speech = self.vad.is_speech(block)

        if not self._collecting:
            if speech:
                self._speech_ms += block_ms
                self._candidate.append(block)
                if self._speech_ms >= self.config.start_speech_ms:
                    # Confirmed. Open the utterance with the pre-roll so the
                    # first syllable is not clipped.
                    self._collecting = True
                    self._chunks = []
                    if preroll is not None and preroll.size:
                        self._chunks.append(np.asarray(preroll, dtype=np.float32))
                    self._chunks.extend(self._candidate)
                    self._collected_ms = sum(
                        1000.0 * c.size / self.sample_rate for c in self._chunks
                    )
                    self._candidate = []
                    self._silence_ms = 0.0
            else:
                # A lone noisy block is not the start of speech.
                self._speech_ms = 0.0
                self._candidate.clear()
            return None

        self._chunks.append(block)
        self._collected_ms += block_ms

        if speech:
            self._silence_ms = 0.0
        else:
            self._silence_ms += block_ms

        too_long = self._collected_ms >= self.config.max_utterance_s * 1000
        ended = self._silence_ms >= self.config.end_silence_ms
        if ended or too_long:
            return self._finish()
        return None

    def flush(self) -> np.ndarray | None:
        """Force-close whatever is being collected. Used on shutdown."""
        return self._finish() if self._collecting else None

    def _finish(self) -> np.ndarray | None:
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self.reset()
        duration_ms = 1000.0 * audio.size / self.sample_rate
        # A 100ms blip is a door closing, not a sentence. Transcribing it wastes
        # a second and usually produces a hallucinated "Thank you."
        if duration_ms < self.config.min_utterance_ms:
            return None
        return audio
