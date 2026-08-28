"""Route streaming text into the speaker or onto the screen, live.

`split_channels` in persona.py works on a finished reply. This does the same job
on a stream, which is harder: a `<screen>` tag arrives split across deltas, so at
any moment the buffer may end with `<scr` and we cannot yet know whether that is
the start of a tag or someone talking about scripts.

The rule is: emit everything that definitely is not part of a tag, and hold back
any trailing text that could still turn into one. Worst case we hold back eight
characters for one more delta, which is imperceptible.
"""

from __future__ import annotations

OPEN = "<screen>"
CLOSE = "</screen>"


def _held_back(text: str, *tags: str) -> int:
    """How many trailing characters could still become one of `tags`."""
    for tag in tags:
        # Longest suffix of `text` that is a proper prefix of `tag`.
        for length in range(min(len(tag) - 1, len(text)), 0, -1):
            if text[-length:] == tag[:length]:
                return length
    return 0


class ChannelRouter:
    """Splits a stream of deltas into spoken text and screen text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_screen = False

    @property
    def in_screen(self) -> bool:
        """True while inside an unclosed screen block."""
        return self._in_screen

    def feed(self, delta: str) -> tuple[str, str]:
        """Returns (spoken_delta, screen_delta) for this fragment."""
        if not delta:
            return "", ""
        self._buffer += delta
        spoken: list[str] = []
        screen: list[str] = []

        while True:
            if not self._in_screen:
                index = self._buffer.find(OPEN)
                if index == -1:
                    hold = _held_back(self._buffer, OPEN)
                    emit = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
                    if emit:
                        spoken.append(emit)
                    self._buffer = self._buffer[len(emit):]
                    break
                if index:
                    spoken.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(OPEN):]
                self._in_screen = True
            else:
                index = self._buffer.find(CLOSE)
                if index == -1:
                    hold = _held_back(self._buffer, CLOSE)
                    emit = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
                    if emit:
                        screen.append(emit)
                    self._buffer = self._buffer[len(emit):]
                    break
                if index:
                    screen.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(CLOSE):]
                self._in_screen = False

        return "".join(spoken), "".join(screen)

    def flush(self) -> tuple[str, str]:
        """Release anything held back. Call at end of turn."""
        remainder, self._buffer = self._buffer, ""
        if self._in_screen:
            self._in_screen = False
            return "", remainder
        return remainder, ""

    def reset(self) -> None:
        self._buffer = ""
        self._in_screen = False
