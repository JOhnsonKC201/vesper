"""Facts Vasper keeps about the person it talks to.

## How this differs from lessons

`learning.py` keeps instructions: how to behave, stored only when you say
"remember that" or "from now on", and placed in the system prompt at spawn.
This keeps facts: "my exam is Friday", "Maya is my sister". Claude notices
them mid conversation and writes each one as a `remember:` line inside a
screen block, which is printed on the dashboard and never spoken. So nothing
is stored that you could not see being stored.

## Why per turn, not in the system prompt

The system prompt is fixed for the life of the process, so a fact learned at
ten in the morning would not reach Claude until the next restart. And a list
that grows for months does not belong in front of every question. Each turn
gets the facts that share a word with what was just asked, a handful at most,
inside the context block that already carries the time and the active window.

## What is refused

A fact rides along on every turn that mentions it, which makes the store a
target. So it refuses secrets (they would sit in a plain file and then go back
to Claude), links, anything that reads like a rule about what Vasper may do
(that would be a way around the spoken gate), and anything long.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

# The most facts kept. Oldest go first: what was true a year ago is the likeliest
# to have stopped being true.
MAX_ITEMS = 300
MAX_FACT_LENGTH = 160
MIN_FACT_LENGTH = 6
# Below this many facts, every one comes back on every turn. It is cheaper than
# being clever, and a short list is exactly when a missed match hurts most.
ALL_BELOW = 8
# The most facts one turn gets.
MAX_RETURNED = 6

# Horizontal space only after the colon: `\s` would cross a line break and
# store whatever the next screen line happened to be.
_REMEMBER_LINE = re.compile(r"^[ \t]*remember[ \t]*:[ \t]*(\S.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE)

_SECRET = re.compile(
    r"\b(password|passcode|passwd|api ?key|token|secret|pin code|ssn|"
    r"social security|card number|credit card|cvv|sk-[a-z0-9-]+)\b|\d[\d -]{11,}\d",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(allow|allows|allowed|permission|permitted|without asking|no need to ask|"
    r"don'?t ask|approve|approves|approved|authori[sz]e[sd]?|bypass|grant)\b",
    re.IGNORECASE,
)
_LINK = re.compile(r"https?://|\bwww\.", re.IGNORECASE)

# Words that carry no topic. "user" is here because Claude writes most facts
# about "the user", so it would match everything.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its me "
    "my of on or our she so that the their them they this to was we were what "
    "when where which who why will with you your user users about".split()
)


def extract_facts(screen_text: str) -> list[str]:
    """The `remember:` lines in a turn's screen channel."""
    return [m.strip() for m in _REMEMBER_LINE.findall(screen_text or "") if m.strip()]


def _words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("'s", ""))
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _fingerprint(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def refusal(text: str) -> str:
    """Why this text may not be kept, or "" when it may."""
    if not MIN_FACT_LENGTH <= len(text) <= MAX_FACT_LENGTH:
        return "length"
    if _SECRET.search(text):
        return "secret"
    if _PERMISSION.search(text):
        return "permission"
    if _LINK.search(text):
        return "link"
    return ""


@dataclass
class Fact:
    text: str
    created: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "created": self.created}


class Facts:
    """The fact store. A JSON file under var/, or memory only with no path."""

    def __init__(self, path: Path | str | None, *, max_items: int = MAX_ITEMS) -> None:
        self.path = Path(path) if path else None
        self.max_items = max_items
        self.items: list[Fact] = self._load()

    def _load(self) -> list[Fact]:
        if self.path is None or not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A file that cannot be read is a lost list, not a reason to stop
            # Vasper from starting. The next save replaces it.
            return []
        rows = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []
        facts = []
        for row in rows:
            if not (isinstance(row, dict) and isinstance(row.get("text"), str)):
                continue
            text = row["text"].strip()
            # The same rules on the way in as on the way out of a turn: a file
            # edited by hand does not get to skip them.
            if refusal(text):
                continue
            try:
                created = float(row.get("created") or 0.0)
            except (TypeError, ValueError):
                created = 0.0
            facts.append(Fact(text, created))
        return facts[-self.max_items:]

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        payload = {"facts": [f.to_dict() for f in self.items]}
        temp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temp.replace(self.path)

    def add(self, text: str) -> Fact | None:
        """Keep one fact. None when it is refused or already known."""
        text = re.sub(r"\s+", " ", (text or "")).strip().rstrip(".")
        if refusal(text):
            return None
        mark = _fingerprint(text)
        if any(_fingerprint(f.text) == mark for f in self.items):
            return None
        fact = Fact(text, time.time())
        self.items = [*self.items, fact][-self.max_items:]
        self._save()
        return fact

    def relevant(self, utterance: str) -> list[Fact]:
        """The facts worth putting in front of this question."""
        if len(self.items) < ALL_BELOW:
            return list(self.items)
        asked = _words(utterance)
        scored = [
            (len(asked & _words(f.text)), f.created, f)
            for f in self.items
        ]
        hits = sorted((s for s in scored if s[0] > 0), key=lambda s: (s[0], s[1]), reverse=True)
        return [f for _, _, f in hits[:MAX_RETURNED]]

    def context_block(self, utterance: str, user: str = "the user") -> str:
        found = self.relevant(utterance)
        if not found:
            return ""
        lines = "\n".join(f"- {f.text}" for f in found)
        return f"what you know about {user}:\n{lines}"

    def forget_last(self) -> Fact | None:
        if not self.items:
            return None
        dropped = self.items[-1]
        self.items = self.items[:-1]
        self._save()
        return dropped

    def forget_all(self) -> int:
        count = len(self.items)
        self.items = []
        self._save()
        return count

    @property
    def newest(self) -> float:
        return self.items[-1].created if self.items else 0.0
