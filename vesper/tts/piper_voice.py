"""Piper neural text to speech, running entirely offline.

Piper is a small ONNX VITS model. Once the voice file is on disk it never
touches the network again, which is why it is the default here: the whole point
of Vesper is that the only thing leaving the machine is the conversation with
Claude itself.

Playback is deliberately written in small blocks rather than one blocking call.
Barge-in has to be able to cut Vesper off mid-word, and you cannot interrupt
`sounddevice.play(); sounddevice.wait()`. Writing 1024-frame blocks and checking
the stop event between them puts the worst-case stop latency around 45ms at
22.05kHz, which is below the threshold where a person notices a delay.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .shaping import NATURAL, Character, shape

_BLOCK_FRAMES = 1024


class PiperTTS:
    """A Piper voice model wired to the default output device."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        speed: float = 1.0,
        volume: float = 1.0,
        name: str = "",
        character: Character | None = None,
    ) -> None:
        from piper import PiperVoice  # imported late: heavy, and optional

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"piper voice not found: {self._model_path}")

        self._voice = PiperVoice.load(self._model_path)
        self.name = name or self._model_path.stem
        self.sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))
        self.character = character or NATURAL

        from piper.config import SynthesisConfig

        # length_scale is inverse speed: larger stretches the audio out. Piper's
        # default of 1.0 reads a touch slow for conversation.
        #
        # The two noise scales are the character's, not the speed's. They decide
        # how much the delivery wanders in pitch and timing, which is the
        # difference between a voice that sounds chatty and one that sounds
        # composed. No filter applied afterwards can produce that.
        self._syn = SynthesisConfig(
            length_scale=1.0 / max(speed, 0.1),
            volume=max(0.0, min(volume, 1.0)),
            normalize_audio=True,
            noise_scale=self.character.noise_scale,
            noise_w_scale=self.character.noise_w_scale,
        )
        self._stream = None
        self._lock = threading.Lock()

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

    # --- Voice protocol -----------------------------------------------------

    def speak(self, text: str, stop: threading.Event) -> None:
        text = text.strip()
        if not text or stop.is_set():
            return

        with self._lock:
            stream = self._ensure_stream()
            try:
                for chunk in self._voice.synthesize(text, syn_config=self._syn):
                    if stop.is_set():
                        break
                    samples = shape(
                        np.asarray(chunk.audio_int16_array, dtype=np.int16),
                        self.sample_rate,
                        self.character,
                    )
                    for start in range(0, len(samples), _BLOCK_FRAMES):
                        if stop.is_set():
                            break
                        stream.write(samples[start : start + _BLOCK_FRAMES])
            except Exception:
                # A dead output device must not take the whole assistant down.
                # Drop the stream so the next utterance reopens it.
                self._teardown()
                raise

            if stop.is_set():
                # Drop whatever is still buffered in the device so the cut is
                # immediate rather than one buffer late.
                self._teardown()

    def synthesize(self, text: str) -> np.ndarray:
        """Render to samples without playing. Used by tests and the audio harness."""
        parts = [
            shape(
                np.asarray(chunk.audio_int16_array, dtype=np.int16),
                self.sample_rate,
                self.character,
            )
            for chunk in self._voice.synthesize(text, syn_config=self._syn)
        ]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def _teardown(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._teardown()

    # --- discovery ----------------------------------------------------------

    @staticmethod
    def find_voice(voices_dir: str | Path, preferred: str = "") -> Path | None:
        """Pick a voice model from a directory, favouring `preferred`."""
        directory = Path(voices_dir)
        if not directory.is_dir():
            return None
        models = sorted(directory.glob("*.onnx"))
        if not models:
            return None
        if preferred:
            for model in models:
                if model.stem == preferred:
                    return model
        return models[0]
