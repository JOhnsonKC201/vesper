"""Lessons Vesper keeps, so a correction only has to be made once.

## What this is, and what it is not

It is not machine learning. Nothing here trains a model or adjusts a weight,
and calling it that would be a lie told with a straight face. What it is: when
you tell Vesper how you want something done, that instruction is written down
and put in front of Claude at the start of every future session. The behaviour
changes, permanently, because of something you said once. That is the useful
half of "it learns", and it is achievable without pretending.

The alternative, which this replaces, is that every correction lasts until the
session ends and then evaporates. `session.json` already carries the
conversation id across restarts, but a conversation gets compacted and dropped
eventually, and "stop reading me file paths" should outlive that.

## How something becomes a lesson

Two ways, both decided locally with no extra model call.

**You said so.** "Remember that...", "from now on...", "always...", "never...".
These are unambiguous and are stored verbatim, minus the framing.

**You corrected him.** "No, I meant...", "that's wrong", "I said...". These are
weaker signals, so they are stored as a lower-confidence lesson and only reach
the prompt once they have been said more than once. A single "no" during a
noisy transcript should not become a permanent rule.

## Why repetition matters more than recency

A lesson repeated is a lesson that did not take. Reinforcement is counted, and
the prompt is ordered by it, so the instruction you have had to give three
times sits at the top where it is least likely to be lost in a long prompt.

## Getting rid of one

"Forget that" drops the most recent. "Forget everything you have learned"
clears the file. Both are local commands that never reach Claude, for the same
reason mute and undo are: the moment you want something forgotten is not the
moment to depend on a network call.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# How many reach the prompt. The system prompt rides on every turn, so this is
# a cost paid continuously, not once. Twelve short lines is a few hundred
# tokens; a hundred would be a tax on every reply forever.
MAX_IN_PROMPT = 12

# Nothing longer than this is a rule, it is a paragraph, and a paragraph in a
# system prompt is where instructions go to be ignored.
MAX_LENGTH = 180
MIN_LENGTH = 6

# A weak signal has to be repeated before it counts. A single "no" in a noisy
# transcript should not become a permanent instruction.
CORRECTION_THRESHOLD = 2

# "Remember that I prefer X" -> "I prefer X". The framing is how you addressed
# him, not part of the rule.
#
# Each entry carries the prefix to put back, because stripping a trigger can
# strip the meaning with it. "Don't read me file paths" captures as "read me
# file paths", which as a stored rule says the opposite of what was said, and
# nothing about the stored line looks wrong. The prefix is per pattern rather
# than one shared "do not" because "stop reading" has to stay "stop reading":
# "do not reading" is not English and a system prompt full of broken grammar
# is a system prompt that gets ignored.
_EXPLICIT = (
    (re.compile(r"^(?:please\s+)?remember(?:\s+that)?[,:]?\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:from\s+now\s+on|going\s+forward|in\s+future)[,:]?\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:make\s+sure|be\s+sure)(?:\s+(?:to|you))?[,:]?\s+(.+)$", re.I), ""),
    (re.compile(r"^((?:always|never)\s+.+)$", re.I), ""),
    (re.compile(r"^stop\s+(.+ing\b.*)$", re.I), "stop "),
)

# Weaker. These say something went wrong, without saying what the rule is.
#
# "I want you to" and "don't" are down here too. They open a standing rule
# sometimes and a one-off request far more often, and on 2026-09-07 the
# one-offs won: "I want you to start up" and "don't clone it, just change to
# some woman voice" were both written into every future system prompt after a
# single hearing. Said twice, they count; said once, they are a request.
_CORRECTION = (
    (re.compile(r"^(?:no[,.]?\s+)?i\s+(?:said|meant|asked\s+for)\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:no|nope)[,.]?\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:that'?s|thats)\s+(?:not|wrong|incorrect)\b[,.]?\s*(.*)$", re.I), ""),
    (re.compile(r"^actually[,.]?\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:i\s+(?:want|need)\s+you\s+to)\s+(.+)$", re.I), ""),
    (re.compile(r"^(?:don'?t|do\s+not)\s+(?:ever\s+)?(.+)$", re.I), "do not "),
)

_FORGET_LAST = re.compile(
    r"^(?:vesper[,\s]+)?forget\s+(?:that|the\s+last\s+one|what\s+i\s+just\s+said)\b",
    re.I,
)
_FORGET_ALL = re.compile(
    r"^(?:vesper[,\s]+)?forget\s+(?:everything|all)\b", re.I
)

EXPLICIT, CORRECTION = "explicit", "correction"


@dataclass
class Lesson:
    """One thing Vesper was told about how to behave."""

    text: str
    kind: str = EXPLICIT
    times: int = 1
    created: str = ""
    last: str = ""

    @property
    def active(self) -> bool:
        """Is this strong enough to put in front of Claude?

        An explicit instruction counts immediately. A correction has to have
        happened more than once, because the shape of a correction is easy to
        match by accident on ordinary speech.
        """
        if self.kind == EXPLICIT:
            return True
        return self.times >= CORRECTION_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "text": self.text, "kind": self.kind, "times": self.times,
            "created": self.created, "last": self.last,
        }

    @staticmethod
    def from_dict(data: dict) -> "Lesson | None":
        text = str(data.get("text") or "").strip()
        if not text:
            return None
        try:
            times = max(1, int(data.get("times") or 1))
        except (TypeError, ValueError):
            times = 1
        kind = str(data.get("kind") or EXPLICIT)
        return Lesson(
            text=text,
            kind=kind if kind in (EXPLICIT, CORRECTION) else EXPLICIT,
            times=times,
            created=str(data.get("created") or ""),
            last=str(data.get("last") or ""),
        )


def extract(text: str) -> tuple[str, str] | None:
    """Is this an instruction worth keeping? Returns (lesson, kind) or None.

    Runs on every utterance, so it is regex rather than a model call. Being
    occasionally wrong is fine in both directions: a missed lesson means saying
    it again, and a wrong one is one line in a file you can see and delete.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None

    # The wake word is how you address him, never part of the instruction.
    cleaned = re.sub(r"^\s*(?:hey\s+)?vesper[,.\s]+", "", cleaned, flags=re.I).strip()
    if not cleaned:
        return None

    for pattern, prefix in _EXPLICIT:
        found = pattern.match(cleaned)
        if found:
            lesson = _tidy(found.group(1), prefix)
            if lesson:
                return lesson, EXPLICIT

    for pattern, prefix in _CORRECTION:
        found = pattern.match(cleaned)
        if found and found.lastindex:
            lesson = _tidy(found.group(1), prefix)
            if lesson:
                return lesson, CORRECTION
    return None


# A rule fits in a breath. Two words ("start up") is a command; seventeen or
# more is a request, a story, or the television. Both were stored as rules on
# 2026-09-07 before these bounds existed.
MIN_WORDS = 3
MAX_WORDS = 16


def _tidy(fragment: str, prefix: str = "") -> str:
    """Trim a captured fragment, and reject the ones that are not rules."""
    lesson = " ".join((fragment or "").split()).strip(" ,.;:")
    if len(lesson) < MIN_LENGTH or len(lesson) > MAX_LENGTH:
        return ""
    if not MIN_WORDS <= len(lesson.split()) <= MAX_WORDS:
        return ""
    # A question is a request, not a standing instruction. Anywhere in it: "I
    # want you to find it. What's my girlfriend name? You have to find it" was
    # kept because only the last character was checked.
    if "?" in lesson:
        return ""
    return f"{prefix}{lesson}" if prefix else lesson


def wants_forgetting(text: str) -> str:
    """"" , "last" or "all". Checked before anything else is done with a line."""
    cleaned = " ".join((text or "").split())
    if _FORGET_ALL.match(cleaned):
        return "all"
    if _FORGET_LAST.match(cleaned):
        return "last"
    return ""


class Lessons:
    """The store. Never raises: a broken file means Vesper has learned nothing."""

    def __init__(self, path: Path | str | None, *, max_in_prompt: int = MAX_IN_PROMPT):
        self.path = Path(path) if path else None
        self.max_in_prompt = max_in_prompt
        self._lock = threading.Lock()
        self._items: list[Lesson] | None = None

    # --- state --------------------------------------------------------------

    @property
    def items(self) -> list[Lesson]:
        if self._items is None:
            self._items = self._load()
        return self._items

    def _load(self) -> list[Lesson]:
        if self.path is None or not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        found = [Lesson.from_dict(item) for item in raw if isinstance(item, dict)]
        return [lesson for lesson in found if lesson is not None]

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps([item.to_dict() for item in self.items], indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except (OSError, ValueError):
            pass

    # --- learning -----------------------------------------------------------

    def learn(self, text: str, kind: str = EXPLICIT) -> Lesson | None:
        """Record an instruction, or reinforce one already held.

        Returns the lesson if it is now strong enough to act on, and None if it
        is a correction still waiting for a second sighting. The caller uses
        that to decide whether to say "I'll remember that", because claiming to
        have learned something that is not yet in the prompt would be a lie.
        """
        lesson = " ".join((text or "").split()).strip()
        if len(lesson) < MIN_LENGTH or len(lesson) > MAX_LENGTH:
            return None

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            existing = self._same_as(lesson)
            if existing is not None:
                existing.times += 1
                existing.last = now
                # Said outright, it is no longer a guess, whatever it started as.
                if kind == EXPLICIT:
                    existing.kind = EXPLICIT
                self._save()
                return existing if existing.active else None

            fresh = Lesson(text=lesson, kind=kind, created=now, last=now)
            self.items.append(fresh)
            self._save()
            return fresh if fresh.active else None

    def _same_as(self, lesson: str) -> Lesson | None:
        """Have we been told this already? Compared loosely, on purpose.

        Speech recognition will not produce the same string twice, so exact
        matching would fill the file with near-duplicates and never reinforce
        anything.
        """
        target = _fingerprint(lesson)
        for item in self.items:
            if _fingerprint(item.text) == target:
                return item
        return None

    # --- forgetting ---------------------------------------------------------

    def forget_last(self) -> Lesson | None:
        with self._lock:
            if not self.items:
                return None
            dropped = self.items.pop()
            self._save()
            return dropped

    def forget_all(self) -> int:
        with self._lock:
            count = len(self.items)
            self._items = []
            self._save()
            return count

    # --- using --------------------------------------------------------------

    def active(self) -> list[Lesson]:
        """The lessons worth spending prompt on, most reinforced first."""
        ready = [item for item in self.items if item.active]
        ready.sort(key=lambda item: (-item.times, item.created))
        return ready[: self.max_in_prompt]

    def prompt_block(self) -> str:
        """The lines to paste into the system prompt. Empty if there are none."""
        ready = self.active()
        if not ready:
            return ""
        lines = "\n".join(f"- {item.text}" for item in ready)
        return (
            "Things this user has told you before. They said these once and "
            "should not have to say them again:\n" + lines
        )

    def summary(self) -> str:
        held = len(self.items)
        live = len(self.active())
        if not held:
            return "nothing learned yet"
        if live == held:
            return f"{held} things learned"
        return f"{live} of {held} things learned, the rest waiting to be repeated"


def _fingerprint(text: str) -> str:
    """A loose key, so near-identical transcriptions match each other."""
    words = re.findall(r"[a-z0-9']+", (text or "").casefold())
    # Filler that speech recognition adds and drops between takes.
    skip = {"the", "a", "an", "to", "that", "please", "just", "really", "very"}
    return " ".join(word for word in words if word not in skip)
