"""Playback queue with barge-in.

Sentences arrive from the brain faster than they can be spoken, so they queue.
The queue is what makes speak-while-thinking possible: the first sentence starts
playing while Claude is still generating the third.

Barge-in is the reason this is not just `for s in sentences: voice.speak(s)`.
When the user starts talking over Vesper, everything queued has to be abandoned
and the current utterance cut off mid-word. Getting that wrong is the single
most robot-like failure a voice assistant can have.

The generation counter handles the race that a bare stop flag cannot: a barge-in
landing in the gap between the worker dequeuing an item and starting to speak it.
Without it, the worker clears the stop flag and speaks the very line the user
just interrupted.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from ..tts.base import Voice


class Speaker:
    """Turns a stream of sentences into audio, interruptibly."""

    def __init__(
        self,
        voice: Voice,
        *,
        on_start: Callable[[str], None] | None = None,
        on_done: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.voice = voice
        self._on_start = on_start or (lambda text: None)
        self._on_done = on_done or (lambda text: None)
        self._on_error = on_error or (lambda exc: None)

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self._speaking = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._running = True

        self._worker = threading.Thread(target=self._run, daemon=True, name="speaker")
        self._worker.start()

    # --- state --------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    # --- control ------------------------------------------------------------

    def say(self, text: str) -> None:
        """Queue a line. Returns immediately."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._idle.clear()
            self._queue.put((self._generation, text))

    def barge_in(self) -> int:
        """Abandon everything queued and cut off the current line.

        Returns how many queued lines were dropped, which the caller can use to
        tell Claude it was interrupted rather than finished.
        """
        dropped = 0
        with self._lock:
            self._generation += 1
            while True:
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
        self._stop.set()
        return dropped

    def use_voice(self, voice) -> None:
        """Swap the speech backend on a running assistant.

        Changing voice used to mean editing `config.yaml` and restarting, which
        is why the dashboard could not offer it for the local backends. It is
        three steps and none of them are optional:

        Cut off whatever is in flight first. Piper holds the utterance's whole
        duration inside its own lock, so closing the old backend while it is
        mid-sentence would block this thread for seconds.

        Swap under the lock, so a `say()` landing at the same moment queues
        against one voice or the other rather than half of each.

        Close the old one last. Piper keeps an output stream open, and two of
        them fighting over the device is a real symptom, not a tidiness point.
        """
        if voice is None or voice is self.voice:
            return
        self.barge_in()
        with self._lock:
            previous, self.voice = self.voice, voice
        try:
            previous.close()
        except Exception:
            # A backend that will not shut down cleanly must not stop the new
            # one from being used. The worst case is a leaked stream.
            pass

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue drains. Returns False on timeout."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 2.0) -> None:
        self.barge_in()
        self._running = False
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        try:
            self.voice.close()
        except Exception:
            pass

    # --- worker -------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            item = self._queue.get()
            if item is None:
                break

            generation, text = item

            # Clear the stop flag before re-reading the generation, never after.
            # The other order loses a barge-in that lands in this exact gap.
            self._stop.clear()
            with self._lock:
                current = self._generation
            if generation != current:
                self._mark_idle_if_drained()
                continue

            self._speaking.set()
            try:
                self._on_start(text)
                self.voice.speak(text, self._stop)
                if not self._stop.is_set():
                    self._on_done(text)
            except Exception as exc:
                self._on_error(exc)
            finally:
                self._speaking.clear()
                self._mark_idle_if_drained()

    def _mark_idle_if_drained(self) -> None:
        with self._lock:
            if self._queue.empty():
                self._idle.set()
