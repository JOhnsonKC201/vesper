"""Reading the calendar from a file on this machine.

The calendar reaches Vasper as `var/calendar.ics`, written by
scripts/calendar_sync.py, which lives outside the package on purpose: the
package keeps its rule that only the cloud voice may touch the network. Reading
that file is all `vasper calendar` does, which is why it can be on the
allowlist with the other read-only commands.
"""

import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from vesper import schedule as cal

EASTERN = ZoneInfo("America/New_York")

ICS = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//EN
BEGIN:VEVENT
UID:drill@test
DTSTART;TZID=America/New_York:20261001T110000
DTEND;TZID=America/New_York:20261001T113000
RRULE:FREQ=DAILY
SUMMARY:Daily DSA Drill
END:VEVENT
BEGIN:VEVENT
UID:lab@test
DTSTART;TZID=America/New_York:20261014T140000
DTEND;TZID=America/New_York:20261014T153000
SUMMARY:Lab meeting with Dr. Lee
LOCATION:Room 210
END:VEVENT
BEGIN:VEVENT
UID:hw@test
DTSTART;VALUE=DATE:20261014
DTEND;VALUE=DATE:20261015
SUMMARY:ITEC 433 assignment due
END:VEVENT
END:VCALENDAR
"""


def _file(tmp_path, text=ICS, age_s=0.0):
    path = tmp_path / "calendar.ics"
    path.write_text(text, encoding="utf-8")
    if age_s:
        then = time.time() - age_s
        os.utime(path, (then, then))
    return path


def test_a_day_lists_its_events_in_order_with_all_day_first(tmp_path):
    events = cal.events_on(_file(tmp_path), date(2026, 10, 14), days=1, tz=EASTERN)
    assert [e.summary for e in events] == [
        "ITEC 433 assignment due", "Daily DSA Drill", "Lab meeting with Dr. Lee",
    ]
    assert events[0].all_day is True


def test_a_recurring_event_appears_on_every_day_it_repeats(tmp_path):
    events = cal.events_on(_file(tmp_path), date(2026, 10, 20), days=3, tz=EASTERN)
    assert [e.summary for e in events] == ["Daily DSA Drill"] * 3


def test_times_are_shown_in_the_local_zone(tmp_path):
    pacific = ZoneInfo("America/Los_Angeles")
    events = cal.events_on(_file(tmp_path), date(2026, 10, 15), days=1, tz=pacific)
    assert events[0].start.hour == 8  # 11:00 Eastern


def test_the_printout_is_one_line_per_event(tmp_path):
    text = cal.render(_file(tmp_path), date(2026, 10, 14), days=1, tz=EASTERN)
    lines = text.splitlines()
    assert lines[0] == "Wed Oct 14"
    assert "  all day  ITEC 433 assignment due" in lines
    assert "  11:00-11:30  Daily DSA Drill" in lines
    assert "  14:00-15:30  Lab meeting with Dr. Lee (Room 210)" in lines


def test_an_empty_day_says_so(tmp_path):
    empty = ICS.replace("RRULE:FREQ=DAILY\n", "")
    text = cal.render(_file(tmp_path, empty), date(2026, 11, 1), days=1, tz=EASTERN)
    assert "nothing scheduled" in text


def test_no_file_says_how_to_get_one(tmp_path):
    text = cal.render(tmp_path / "missing.ics", date(2026, 10, 14), days=1, tz=EASTERN)
    assert "No calendar has been synced" in text
    assert "calendar_sync.py" in text


def test_a_stale_file_is_flagged(tmp_path):
    path = _file(tmp_path, age_s=5 * 3600)
    text = cal.render(path, date(2026, 10, 14), days=1, tz=EASTERN)
    assert "last synced 5 hours ago" in text


def test_a_fresh_file_says_nothing_about_age(tmp_path):
    text = cal.render(_file(tmp_path), date(2026, 10, 14), days=1, tz=EASTERN)
    assert "synced" not in text


def test_a_broken_file_is_reported_not_raised(tmp_path):
    text = cal.render(_file(tmp_path, "garbage"), date(2026, 10, 14), days=1, tz=EASTERN)
    assert "could not be read" in text


def test_the_window_is_bounded():
    assert cal.clamp_days(0) == 1
    assert cal.clamp_days(400) == cal.MAX_DAYS


def test_day_words_resolve_against_today():
    today = date(2026, 10, 14)
    assert cal.resolve_day("today", today) == today
    assert cal.resolve_day("tomorrow", today) == today + timedelta(days=1)
    assert cal.resolve_day("2026-12-01", today) == date(2026, 12, 1)
    assert cal.resolve_day("friday", today) == date(2026, 10, 16)
    assert cal.resolve_day("wednesday", today) == today


# --- the command -------------------------------------------------------------

from vesper import tool  # noqa: E402


def test_vasper_calendar_prints_the_synced_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "CALENDAR", _file(tmp_path))
    assert tool.main(["calendar", "2026-10-14"]) == 0
    assert "Lab meeting with Dr. Lee" in capsys.readouterr().out


def test_vasper_calendar_refuses_a_day_it_cannot_read(capsys):
    assert tool.main(["calendar", "someday"]) == 2
    assert "not a day" in capsys.readouterr().out
