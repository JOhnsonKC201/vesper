"""A monthly character allowance, so a metered voice cannot surprise you.

ElevenLabs bills per character: $0.10 per thousand on the standard models and
$0.05 on Flash, with 10,000 credits a month on the free tier. A Vesper reply is
around 200 characters, so free is roughly a hundred spoken replies and then it
stops. An always-on assistant would find that ceiling in an afternoon.

The design rule is that hitting the ceiling is never an error and never
silence. Over the cap, `ElevenTTS` speaks through Piper instead. You notice
because the voice changes, not because Vesper stops working or because a bill
arrives.

The default cap sits deliberately under the free allowance rather than on it,
so the limit is discovered here rather than by ElevenLabs.

Cache hits are not counted, because they cost nothing. That is the whole reason
the stock phrases stay in the good voice after the budget is gone.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path


def _this_month() -> str:
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"


class Budget:
    """Characters spent this calendar month, persisted across restarts.

    Never raises. An unreadable or unwritable file means the budget behaves as
    if nothing has been spent, which risks overspending rather than muting the
    assistant. That is the right way round: the cap is a courtesy, and Vesper
    going quiet because a JSON file was locked would be a worse failure than
    speaking a few hundred characters too many.
    """

    def __init__(self, path: Path | str | None, *, cap: int = 9_000) -> None:
        self.path = Path(path) if path else None
        self.cap = max(0, int(cap))
        self._lock = threading.Lock()
        self._month = _this_month()
        self._spent = 0
        self._loaded = False

    # --- state --------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        # A stored month that is not this one means the allowance has reset.
        if str(raw.get("month") or "") == self._month:
            try:
                self._spent = max(0, int(raw.get("characters") or 0))
            except (TypeError, ValueError):
                self._spent = 0

    def _roll(self) -> None:
        """Reset if the calendar month changed while the process was running."""
        now = _this_month()
        if now != self._month:
            self._month = now
            self._spent = 0
            self._save()

    @property
    def spent(self) -> int:
        with self._lock:
            self._load()
            self._roll()
            return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.spent)

    @property
    def month(self) -> str:
        return self._month

    def fraction(self) -> float:
        """0.0 to 1.0, for a progress bar. A cap of 0 reads as full."""
        if self.cap <= 0:
            return 1.0
        return min(1.0, self.spent / self.cap)

    # --- decisions ----------------------------------------------------------

    def allows(self, characters: int) -> bool:
        """Is there room for this line?

        Whole lines only. Speaking the first two thirds of a sentence and then
        switching voice mid-word would be worse than either voice alone.
        """
        if self.cap <= 0:
            return False
        return self.spent + max(0, int(characters)) <= self.cap

    def spend(self, characters: int) -> None:
        """Record a synthesis that actually happened."""
        amount = max(0, int(characters))
        if not amount:
            return
        with self._lock:
            self._load()
            self._roll()
            self._spent += amount
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._loaded = True
            self._month = _this_month()
            self._spent = 0
            self._save()

    # --- persistence --------------------------------------------------------

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"month": self._month, "characters": self._spent}, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except (OSError, ValueError):
            pass

    def summary(self) -> str:
        """One line for the dashboard and the log."""
        return f"{self.spent:,} / {self.cap:,} characters this month"
