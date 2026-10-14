"""The calendar, read from a file on this machine.

`scripts/calendar_sync.py` downloads the calendar's private iCal address to
`var/calendar.ics`. That script is outside the package on purpose: the package
keeps its rule that only the cloud voice touches the network, and this module
only ever opens a local file. Which is also why `vasper calendar` can sit on
the allowlist with the other read-only commands.

Recurring events are expanded by `recurring_ical_events`, since a daily routine
stored once with an RRULE is exactly the shape a hand-rolled parser gets wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path

# The longest window one call may ask for. A month is plenty to answer "when is
# my next exam"; past it the printout stops being something to read aloud from.
MAX_DAYS = 31
# Older than this and the printout says when it was last synced, so a sync
# that quietly stopped does not pass for an empty afternoon.
STALE_AFTER_S = 2 * 3600

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

NO_FILE = (
    "No calendar has been synced yet. Set it up with scripts/calendar_sync.py; "
    "its header says where the private iCal address goes."
)


@dataclass(frozen=True)
class Event:
    start: datetime
    end: datetime
    summary: str
    location: str = ""
    all_day: bool = False


def clamp_days(days: int) -> int:
    return max(1, min(int(days), MAX_DAYS))


def resolve_day(word: str, today: date) -> date:
    """today, tomorrow, a weekday name (this one or the next), or YYYY-MM-DD."""
    word = (word or "today").strip().lower()
    if word == "today":
        return today
    if word == "tomorrow":
        return today + timedelta(days=1)
    if word in _WEEKDAYS:
        ahead = (_WEEKDAYS.index(word) - today.weekday()) % 7
        return today + timedelta(days=ahead)
    return date.fromisoformat(word)


def _localize(value: datetime, tz: tzinfo | None) -> datetime:
    """An aware local datetime. With no `tz`, the system zone AS IT IS ON THAT
    DATE: `astimezone()` with no argument looks the offset up per instant, so a
    November event is not shown with October's daylight saving offset. A
    naive value (a floating time, or a midnight built here) is local time."""
    if tz is None:
        return value.astimezone()
    if value.tzinfo is None:
        value = value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _as_local(value, tz: tzinfo | None) -> tuple[datetime, bool]:
    """A DTSTART or DTEND value as a local datetime, and whether it was a date."""
    if isinstance(value, datetime):
        return _localize(value, tz), False
    return _localize(datetime(value.year, value.month, value.day), tz), True


def occurs_on(event: "Event", day: date) -> bool:
    """Does the event take up any of this day? A DTEND is exclusive, so an
    event ending at midnight, or an all-day event's next-day DTEND, does not
    spill onto the following day."""
    first = event.start.date()
    if event.end <= event.start:
        return day == first
    last = (event.end - timedelta(microseconds=1)).date()
    return first <= day <= last


def events_on(path: Path, first: date, days: int = 1, tz: tzinfo | None = None) -> list[Event]:
    """Every event touching the window, all-day ones first within each day."""
    import icalendar
    import recurring_ical_events

    days = clamp_days(days)
    midnight = datetime(first.year, first.month, first.day)
    start = _localize(midnight, tz)
    end = _localize(midnight + timedelta(days=days), tz)
    calendar = icalendar.Calendar.from_ical(Path(path).read_bytes())
    found = []
    for component in recurring_ical_events.of(calendar).between(start, end):
        begins, all_day = _as_local(component.decoded("DTSTART"), tz)
        raw_end = component.get("DTEND")
        finishes = _as_local(component.decoded("DTEND"), tz)[0] if raw_end else begins
        found.append(Event(
            start=begins,
            end=finishes,
            summary=str(component.get("SUMMARY") or "(no title)").strip(),
            location=str(component.get("LOCATION") or "").strip(),
            all_day=all_day,
        ))
    return sorted(found, key=lambda e: (e.start.date(), not e.all_day, e.start, e.summary))


def _age_note(path: Path, now_s: float) -> str:
    age = now_s - path.stat().st_mtime
    if age < STALE_AFTER_S:
        return ""
    hours = int(age // 3600)
    return f"(last synced {hours} hours ago, so this may be out of date)"


def render(
    path: Path,
    first: date,
    days: int = 1,
    tz: tzinfo | None = None,
) -> str:
    """The printout `vasper calendar` gives Claude: one line per event."""
    path = Path(path)
    if not path.exists():
        return NO_FILE
    try:
        events = events_on(path, first, days, tz)
    except Exception as exc:  # noqa: BLE001 - any parse failure is one message
        return f"The synced calendar could not be read ({type(exc).__name__}). Run scripts/calendar_sync.py again."
    lines = []
    note = _age_note(path, time.time())
    if note:
        lines.append(note)
    for offset in range(clamp_days(days)):
        day = first + timedelta(days=offset)
        lines.append(day.strftime("%a %b %d").replace(" 0", " "))
        todays = [e for e in events if occurs_on(e, day)]
        if not todays:
            lines.append("  nothing scheduled")
        for event in todays:
            if event.all_day:
                when = "all day"
            elif event.start.date() < day:
                when = f"until {event.end:%H:%M}" if event.end.date() == day else "all day"
            else:
                when = f"{event.start:%H:%M}-{event.end:%H:%M}"
            where = f" ({event.location})" if event.location else ""
            lines.append(f"  {when}  {event.summary}{where}")
    return "\n".join(lines)
