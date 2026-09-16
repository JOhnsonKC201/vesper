"""Assembling machine state into something worth putting in a prompt.

Two very different consumers:

  `context_block()` rides along with every spoken turn, so it must be tiny.
  Roughly eighty tokens. It is the difference between Vesper answering "what is
  this error" usefully and asking "which error?".

  `changes_since()` feeds the proactive loop, and describes what is *different*
  rather than what is true. Nobody wants to be told the CPU is at twelve percent;
  they want to be told it just went to ninety nine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from .system import Vitals, vitals
from .window import ActiveWindow, active_window, idle_seconds


@dataclass(frozen=True)
class Change:
    """One observed difference, and whether it is worth spending a turn on.

    Window switches and idle transitions happen constantly. Consulting Claude
    about every one of them would cost more per hour than the assistant is
    worth, so they are recorded as context but never trigger a check on their
    own. Something notable has to happen first; the ambient changes then ride
    along to give that check its setting.
    """

    text: str
    notable: bool = False

    def __str__(self) -> str:  # keeps existing string formatting working
        return self.text


@dataclass(frozen=True)
class Snapshot:
    at: float = field(default_factory=time.time)
    window: ActiveWindow = field(default_factory=ActiveWindow)
    vitals: Vitals = field(default_factory=Vitals)
    idle_s: float = 0.0

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.at)


def take(*, include_processes: bool = False) -> Snapshot:
    return Snapshot(
        window=active_window(),
        vitals=vitals(include_processes=include_processes),
        idle_s=idle_seconds(),
    )


def _spoken_time(moment: datetime) -> str:
    """A clock reading a person would actually say."""
    return moment.strftime("%A %d %B, %H:%M")


def _spoken_idle(seconds: float) -> str:
    if seconds < 30:
        return "at the keyboard"
    if seconds < 120:
        return f"idle {int(seconds)} seconds"
    if seconds < 3600:
        return f"idle {int(seconds // 60)} minutes"
    return f"idle {seconds / 3600:.1f} hours"


def context_block(snapshot: Snapshot | None = None, *, location: str = "") -> str:
    """The compact ambient context attached to every spoken turn.

    `location` is where the user said they are, from config. It rides beside
    the clock so weather and anything local can be answered without asking,
    and it is absent when blank, because a line saying "where: unknown" is a
    question the model would then ask out loud.
    """
    snapshot = snapshot or take()
    lines = [f"time: {_spoken_time(snapshot.when)}"]
    where = " ".join((location or "").split())
    if where:
        lines.append(f"where: {where}")
    if snapshot.window.known:
        lines.append(f"focused window: {snapshot.window.describe()}")
    lines.append(f"user: {_spoken_idle(snapshot.idle_s)}")
    lines.append(snapshot.vitals.describe())
    return "\n".join(lines)


def changes_since(previous: Snapshot | None, current: Snapshot) -> list[Change]:
    """Notable differences between two snapshots, in plain language.

    Thresholds are deliberately blunt. The point is to surface things a person
    would notice, not to build a monitoring system.
    """
    if previous is None:
        return []

    notes: list[Change] = []
    minutes = max(0.0, (current.at - previous.at) / 60.0)

    before, after = previous.window, current.window
    if after.known and before.describe() != after.describe():
        if before.process and after.process and before.process != after.process:
            notes.append(Change(f"switched from {before.process} to {after.process}"))
        else:
            notes.append(Change(f"now looking at {after.describe()}"))

    old, new = previous.vitals, current.vitals

    if new.cpu_percent - old.cpu_percent > 40 and new.cpu_percent > 70:
        busiest = new.top_processes[0][0] if new.top_processes else "something"
        notes.append(
            Change(
                f"cpu jumped from {old.cpu_percent:.0f}% to {new.cpu_percent:.0f}%, "
                f"mostly {busiest}",
                notable=True,
            )
        )

    if new.ram_percent > 90 and old.ram_percent <= 90:
        notes.append(Change(f"memory is at {new.ram_percent:.0f}%", notable=True))

    if new.battery_percent is not None and old.battery_percent is not None:
        if old.on_ac_power and new.on_ac_power is False:
            notes.append(Change(f"unplugged from power at {new.battery_percent}%", notable=True))
        elif old.on_ac_power is False and new.on_ac_power:
            notes.append(Change("plugged back in"))
        for threshold in (20, 10, 5):
            if old.battery_percent > threshold >= new.battery_percent:
                notes.append(Change(f"battery down to {new.battery_percent}%", notable=True))
                break

    if new.disk_free_gb < 10 and old.disk_free_gb >= 10:
        notes.append(Change(f"only {new.disk_free_gb:.0f}gb left on C", notable=True))

    if previous.idle_s > 600 and current.idle_s < 30:
        notes.append(Change(f"back at the keyboard after {previous.idle_s / 60:.0f} minutes away"))
    elif current.idle_s > 1800 and previous.idle_s <= 1800:
        notes.append(Change(f"away from the keyboard for {current.idle_s / 60:.0f} minutes"))

    if minutes > 0 and notes:
        notes.append(Change(f"(over the last {minutes:.0f} minutes)"))
    return notes
