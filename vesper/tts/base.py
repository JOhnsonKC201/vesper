"""The voice interface.

Backends own their own playback rather than returning samples. That sounds like
the wrong seam until you try to support Windows SAPI, which speaks through COM
and never hands you a buffer. Making every backend responsible for "make this
text audible, and stop immediately when told" keeps SAPI and Piper behind one
honest interface.

`stop` is a threading.Event rather than a method call because barge-in has to
interrupt playback that is already in flight on another thread. A backend that
cannot honour it mid-utterance should at minimum check it between chunks.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class Voice(Protocol):
    """Something that can say a line out loud."""

    name: str

    def speak(self, text: str, stop: threading.Event) -> None:
        """Say `text`, blocking until finished or until `stop` is set."""
        ...

    def close(self) -> None:
        """Release audio devices and any native handles."""
        ...


class NullVoice:
    """A voice that says nothing. Used in tests and when no backend loads."""

    name = "null"

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str, stop: threading.Event) -> None:
        self.spoken.append(text)

    def close(self) -> None:
        pass
