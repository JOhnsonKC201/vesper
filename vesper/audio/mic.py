"""Always-on microphone capture.

Unlike push-to-talk dictation, a conversational assistant has to be listening
before it knows it is being spoken to. So the mic runs continuously into a small
ring buffer, and the endpointer upstream decides which stretches are speech.

The pre-roll buffer is the reason for the ring: by the time voice activity is
detected, the first syllable is already in the past. Keeping a few hundred
milliseconds of history means "Vesper, what's..." does not arrive as "esper,
what's".

Two lifecycle details are carried over from Echo Flow, both of which were bugs
there first:

  - If `stream.start()` raises, the stream must be closed here. sounddevice has
    no __del__, so a dropped handle orphans the audio device until the process
    exits.
  - `stop()` must use try/finally. A PortAudio error on device removal that
    escapes leaves the recording flag set, and every later start() silently
    early-returns forever.
"""

from __future__ import annotations

import collections
import queue
import threading

import numpy as np

SAMPLE_RATE = 16_000
BLOCK_MS = 30
BLOCK_FRAMES = SAMPLE_RATE * BLOCK_MS // 1000  # 480


class Microphone:
    """Continuous capture into a queue of float32 mono blocks."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        block_frames: int = BLOCK_FRAMES,
        device: int | str | None = None,
        preroll_ms: int = 400,
        on_error=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_frames = block_frames
        self.device = device
        self._on_error = on_error or (lambda exc: None)

        self._blocks: queue.Queue = queue.Queue(maxsize=256)
        preroll_blocks = max(1, (preroll_ms * sample_rate // 1000) // block_frames)
        self._preroll: collections.deque = collections.deque(maxlen=preroll_blocks)
        self._preroll_lock = threading.Lock()

        self._stream = None
        self._running = False

    # --- lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        import sounddevice as sd

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.device,
            dtype="float32",
            blocksize=self.block_frames,
            callback=self._callback,
        )
        try:
            stream.start()
        except Exception:
            # sounddevice has no __del__; a dropped handle orphans the device.
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._stream = stream
        self._running = True

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception as exc:
            self._on_error(exc)
        finally:
            # Must run even if PortAudio raised, or start() is dead forever.
            self._running = False

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # --- capture ------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._on_error(RuntimeError(f"audio input status: {status}"))
        block = indata.copy().reshape(-1).astype(np.float32)
        with self._preroll_lock:
            self._preroll.append(block)
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            # Drop the oldest rather than block the PortAudio callback thread.
            # Blocking in here causes audible glitching across the whole system.
            try:
                self._blocks.get_nowait()
                self._blocks.put_nowait(block)
            except queue.Empty:
                pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next block of audio, or None if nothing arrived within `timeout`."""
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def preroll(self) -> np.ndarray:
        """The last few hundred milliseconds, so we do not clip the first word."""
        with self._preroll_lock:
            blocks = list(self._preroll)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    def drain(self) -> None:
        """Discard buffered audio. Used after Vesper speaks, to drop its own voice."""
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                break
        with self._preroll_lock:
            self._preroll.clear()

    # --- discovery ----------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        try:
            import sounddevice as sd

            return [
                {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0
            ]
        except Exception:
            return []
