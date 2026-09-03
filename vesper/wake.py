"""Wake word detection over transcribed text.

Doing this on the transcript rather than on raw audio means no second model and
no extra CPU burning in the background: the utterance is already being
transcribed, so matching a word in it is free.

The cost is that it has to tolerate what Whisper actually produces. "Vesper" at
the start of a sentence, said quickly, comes back as "Vesper", "Vespa",
"Whisper", "Jasper" or "Vester" depending on the microphone and the room. An
exact match fails often enough to make the assistant feel deaf, which is a far
worse failure than very occasionally waking when it should not have.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# Mishearings Whisper actually produced for these wake words during testing.
#
# Generous, but not to the point of including ordinary English. "whisper" and
# "jasper" were both in this list and both had to come out: they are real words
# people say, and "whisper it to me quietly" waking the assistant mid-meeting is
# a far worse failure than missing one wake. Since Whisper is now biased toward
# the name via initial_prompt (see stt/whisper.py), those substitutions are rare
# anyway, and the fuzzy pass below still catches near-misses.
HOMOPHONES: dict[str, tuple[str, ...]] = {
    # The name said out loud. Kept separate from the "vesper" entry below rather
    # than merged into it, because the two behave differently under the fuzzy
    # pass: see EXACT_ONLY.
    "vasper": ("vasper", "vaspa", "vasber", "vaspr", "vaspers", "vasparr"),
    # Still answered, deliberately. Whisper was biased toward this spelling for
    # months and still lands on it, and neither spelling is ordinary English.
    "vesper": ("vesper", "vespa", "vester", "vespers", "vespr", "vesber"),
    "jarvis": ("jarvis", "javis", "jarvus", "jervis", "charvis"),
    "computer": ("computer", "computor"),
}

# Wake words matched exactly rather than approximately, because their fuzzy
# neighbourhood is full of ordinary English.
#
# Measured, not assumed. Against the "vasper" spellings, difflib scores
# "casper", "jasper" and "vaster" at 0.833, which is exactly what the genuine
# mishearings "vasfer" and "vosper" score. "vase" and "vapor" reach 0.800;
# "paper", "laser", "taser", "caper", "aspen" and "viper" all reach 0.727,
# above the 0.72 the fuzzy pass accepts. No threshold separates the two sets,
# so there is no fuzzy pass to tune here: "-asper" and "-aper" are productive
# English endings in a way "-esper" is not, and a 60 word sample produced 18
# false wakes.
#
# What is given up is catching a mishearing nobody has listed yet. What covers
# that instead: the whisper initial_prompt now names Vasper, so the model is
# biased toward producing it, and "vesper" keeps its fuzzy pass because its
# neighbourhood is genuinely empty.
EXACT_ONLY = frozenset({"vasper"})

_WORD = re.compile(r"[a-z']+")

# How far into an utterance the wake word may appear. Addressing someone by name
# happens at the start ("Vesper, what time is it") or occasionally at the end
# ("what time is it, Vesper"), never in the middle of a clause.
_HEAD_WORDS = 3
_TAIL_WORDS = 2

# Whisper sometimes splits the name across two tokens ("Vesper" heard as "best
# but"). The homophone list cannot enumerate that, so the head of the utterance
# is also compared fuzzily, including adjacent word pairs joined together.
# 0.72 was chosen by testing: it accepts vespa/vester/bestper and rejects every
# common English opener tried against it.
_FUZZY_THRESHOLD = 0.72


def _close_enough(candidate: str, targets: set[str]) -> bool:
    for target in targets:
        if abs(len(candidate) - len(target)) > 4:
            continue
        if difflib.SequenceMatcher(None, candidate, target).ratio() >= _FUZZY_THRESHOLD:
            return True
    return False


@dataclass
class WakeConfig:
    words: tuple[str, ...] = ("vesper", "jarvis")
    # Once addressed, keep listening without the name for this long, so a
    # conversation does not require saying "Vesper" before every sentence.
    follow_up_window_s: float = 25.0
    require_wake_word: bool = True


@dataclass
class WakeResult:
    triggered: bool
    text: str
    matched: str = ""
    reason: str = ""


def _variants(words: tuple[str, ...], *, fuzzy: bool = False) -> set[str]:
    """Every spelling that counts as the name. `fuzzy` drops the exact-only ones."""
    out: set[str] = set()
    for word in words:
        key = word.lower().strip()
        if fuzzy and key in EXACT_ONLY:
            continue
        out.update(HOMOPHONES.get(key, (key,)))
        out.add(key)
    return out


def strip_wake_word(text: str, words: tuple[str, ...]) -> tuple[str, str]:
    """Remove a leading or trailing wake word. Returns (remainder, matched)."""
    accepted = _variants(words)
    approximate = _variants(words, fuzzy=True)
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return text.strip(), ""

    for index in range(min(_HEAD_WORDS, len(tokens))):
        if tokens[index] in accepted:
            matched = tokens[index]
            # Cut the original string after that word, preserving its casing and
            # punctuation for everything that follows.
            pattern = re.compile(
                r"^\W*(?:\w+\W+){%d}%s\W*" % (index, re.escape(matched)), re.IGNORECASE
            )
            remainder = pattern.sub("", text, count=1)
            return remainder.strip(), matched

    # Fuzzy pass over the head, single words and adjacent pairs.
    head = tokens[:_HEAD_WORDS]
    for index, token in enumerate(head):
        joined = [(token, 1)]
        if index + 1 < len(head):
            joined.append((token + head[index + 1], 2))
        for candidate, span in joined:
            if approximate and _close_enough(candidate, approximate):
                pattern = re.compile(
                    r"^\W*(?:\w+\W+){%d}(?:\w+\W*){%d}" % (index, span),
                    re.IGNORECASE,
                )
                remainder = pattern.sub("", text, count=1)
                return remainder.strip(), candidate

    for offset in range(1, min(_TAIL_WORDS, len(tokens)) + 1):
        if tokens[-offset] in accepted:
            matched = tokens[-offset]
            pattern = re.compile(r"[\s,]*%s\W*$" % re.escape(matched), re.IGNORECASE)
            remainder = pattern.sub("", text, count=1)
            return remainder.strip(), matched

    return text.strip(), ""


class WakeGate:
    """Decides whether an utterance was addressed to Vesper."""

    def __init__(self, config: WakeConfig | None = None) -> None:
        self.config = config or WakeConfig()
        self._engaged_until = 0.0

    def engage(self, now: float) -> None:
        """Open the follow-up window. Called after Vesper replies."""
        self._engaged_until = now + self.config.follow_up_window_s

    def disengage(self) -> None:
        self._engaged_until = 0.0

    def engaged(self, now: float) -> bool:
        return now < self._engaged_until

    def awake(self, now: float) -> bool:
        """Would plain speech, with no name in it, be acted on right now?

        Not the same question as `engaged`. With `require_wake_word` off there
        is no window at all and everything is acted on, so a status panel
        reading `engaged` would say asleep about an assistant that is listening
        to the whole room.
        """
        return not self.config.require_wake_word or self.engaged(now)

    def check(self, text: str, now: float) -> WakeResult:
        text = (text or "").strip()
        if not text:
            return WakeResult(False, "", reason="empty")

        if not self.config.require_wake_word:
            return WakeResult(True, text, reason="open-mic")

        remainder, matched = strip_wake_word(text, self.config.words)
        if matched:
            return WakeResult(True, remainder, matched=matched, reason="wake-word")

        if self.engaged(now):
            return WakeResult(True, text, reason="follow-up")

        return WakeResult(False, text, reason="not-addressed")
