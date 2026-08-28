"""Assemble streaming text deltas into speakable sentences.

The single biggest lever on how alive Vesper feels is how fast it starts
talking. A full turn takes roughly five seconds; the first sentence is usually
ready in under two. So we flush to the voice at sentence boundaries instead of
waiting for the turn to finish.

Getting the boundary wrong is audible in both directions. Splitting on every
period turns "598.1 GB" into two utterances with a gap in the middle. Never
splitting means the pause before Vesper speaks is the full turn latency.
"""

from __future__ import annotations

import re
from typing import Iterator

# Words that legitimately end in a period mid-sentence. Kept deliberately short:
# a false negative just delays one flush, a false positive chops a sentence.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st vs etc approx dept est fig no vol
    jan feb mar apr jun jul aug sep sept oct nov dec
    i.e e.g a.m p.m u.s u.k
    """.split()
)

_TERMINATORS = ".!?"
# Closing punctuation allowed to trail a terminator: He said "done." / (done.)
_CLOSERS = '"\')]}”’'

_WORD_TAIL = re.compile(r"([A-Za-z.]+)$")


def _is_boundary(text: str, index: int) -> bool:
    """Is the terminator at `index` a real end of sentence?"""
    char = text[index]
    if char not in _TERMINATORS:
        return False

    # Look past any closing quote or bracket.
    end = index + 1
    while end < len(text) and text[end] in _CLOSERS:
        end += 1

    # A terminator at the very end of what we have so far is ambiguous: more
    # text may still arrive. Let the caller decide via flush().
    if end >= len(text):
        return False

    # Must be followed by whitespace. "3.5" and "a.b" are not boundaries.
    if not text[end].isspace():
        return False

    if char == ".":
        # Decimal numbers: digit before and after the dot.
        if index > 0 and text[index - 1].isdigit():
            after = text[end:].lstrip()
            if after and after[0].isdigit():
                return False

        match = _WORD_TAIL.search(text[:index])
        if match:
            word = match.group(1).lower().rstrip(".")
            # Single letters are initials ("J. Smith"), not sentence ends.
            if len(word) == 1 and word.isalpha():
                return False
            if word in _ABBREVIATIONS:
                return False
            # Dotted acronyms such as "U.S.A." keep their internal dots.
            if "." in match.group(1)[:-1]:
                return False
    return True


class SentenceAssembler:
    """Accumulates text deltas, releases complete sentences.

    Also releases on a blank line (paragraph break) and force-releases when the
    buffer grows past `max_chars` without any boundary, so a model that emits a
    long unpunctuated run does not leave Vesper silent.
    """

    def __init__(self, min_chars: int = 4, max_chars: int = 320) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, delta: str) -> Iterator[str]:
        """Add streamed text. Yields any sentences that became complete."""
        if not delta:
            return
        self._buffer += delta

        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buffer = self._buffer[:cut], self._buffer[cut:].lstrip()
            chunk = chunk.strip()
            if chunk:
                yield chunk

    def _find_cut(self) -> int | None:
        """Index to cut the buffer at, or None if nothing is ready."""
        # A paragraph break is always a safe boundary.
        para = self._buffer.find("\n\n")
        if para != -1:
            return para

        for index, char in enumerate(self._buffer):
            if char not in _TERMINATORS:
                continue
            if index + 1 < self.min_chars:
                continue
            if _is_boundary(self._buffer, index):
                end = index + 1
                while end < len(self._buffer) and self._buffer[end] in _CLOSERS:
                    end += 1
                return end

        if len(self._buffer) > self.max_chars:
            # Break at the last space before the cap so we do not split a word.
            space = self._buffer.rfind(" ", 0, self.max_chars)
            return space if space > self.min_chars else self.max_chars
        return None

    def flush(self) -> str:
        """Release whatever is left. Call at end of turn."""
        remainder, self._buffer = self._buffer.strip(), ""
        return remainder

    def reset(self) -> None:
        self._buffer = ""
