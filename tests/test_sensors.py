"""Machine awareness: what Vesper knows without being told."""

from datetime import datetime

import pytest

from vesper.sensors.snapshot import Change, Snapshot, changes_since, context_block
from vesper.sensors.system import Vitals, vitals
from vesper.sensors.window import ActiveWindow, active_window, idle_seconds


def snap(*, cpu=10.0, ram=50.0, disk=500.0, battery=80, ac=True, idle=5.0,
         window="Code.exe", title="main.py", at=1000.0, top=()):
    return Snapshot(
        at=at,
        window=ActiveWindow(title=title, process=window),
        vitals=Vitals(
            cpu_percent=cpu, ram_percent=ram, disk_free_gb=disk,
            battery_percent=battery, on_ac_power=ac, top_processes=top,
        ),
        idle_s=idle,
    )


def texts(changes):
    return [str(c) for c in changes]


def notable(changes):
    return [str(c) for c in changes if c.notable]


# --- live sensors -----------------------------------------------------------


def test_active_window_returns_something_usable():
    window = active_window()
    assert isinstance(window, ActiveWindow)
    if window.known:
        assert isinstance(window.describe(), str)


def test_idle_seconds_is_a_sane_number():
    value = idle_seconds()
    assert value >= 0
    assert value < 86400 * 7


def test_vitals_are_populated_on_this_machine():
    reading = vitals()
    assert 0 <= reading.cpu_percent <= 100 * 64
    assert 0 < reading.ram_percent <= 100
    assert reading.disk_free_gb > 0
    assert "cpu" in reading.describe()


def test_system_idle_process_never_appears_as_a_top_consumer():
    """It reports unused CPU summed across cores, so it always wins and means
    nothing."""
    names = [name.lower() for name, _ in vitals(include_processes=True).top_processes]
    assert "system idle process" not in names


def test_context_block_is_spoken_english_not_a_data_dump():
    block = context_block()
    assert "time:" in block
    assert "cpu" in block
    assert "{" not in block and "}" not in block


def test_context_block_reads_the_clock_the_way_a_person_says_it():
    moment = datetime(2026, 8, 28, 14, 5).timestamp()
    block = context_block(snap(at=moment))
    assert "Friday" in block and "14:05" in block


# --- change detection -------------------------------------------------------


def test_no_previous_snapshot_yields_nothing():
    assert changes_since(None, snap()) == []


def test_identical_snapshots_yield_nothing():
    identical = snap()
    assert changes_since(identical, identical) == []


def test_switching_app_is_noticed_but_is_not_notable():
    changes = changes_since(snap(window="chrome.exe"), snap(window="Code.exe"))
    assert "switched from chrome.exe to Code.exe" in texts(changes)
    assert notable(changes) == [], "an app switch must never cost a Claude turn"


def test_a_cpu_spike_is_notable():
    changes = changes_since(
        snap(cpu=10), snap(cpu=95, top=(("ffmpeg.exe", 95.0),))
    )
    assert any("cpu jumped" in c for c in notable(changes))
    assert "ffmpeg.exe" in " ".join(notable(changes))


def test_a_small_cpu_rise_is_not_reported():
    assert changes_since(snap(cpu=10), snap(cpu=35)) == []


def test_a_high_but_steady_cpu_is_not_reported():
    """Only the change is news. Constant load is the user's own doing."""
    assert changes_since(snap(cpu=88), snap(cpu=92)) == []


def test_unplugging_is_notable():
    changes = changes_since(snap(battery=80, ac=True), snap(battery=79, ac=False))
    assert any("unplugged" in c for c in notable(changes))


def test_plugging_back_in_is_noticed_but_not_worth_interrupting():
    changes = changes_since(snap(battery=40, ac=False), snap(battery=41, ac=True))
    assert "plugged back in" in texts(changes)
    assert notable(changes) == []


@pytest.mark.parametrize("before,after", [(30, 19), (15, 9), (8, 4)])
def test_battery_thresholds_are_notable(before, after):
    changes = changes_since(
        snap(battery=before, ac=False), snap(battery=after, ac=False)
    )
    assert any("battery down to" in c for c in notable(changes))


def test_battery_drifting_within_a_band_is_not_reported():
    assert changes_since(snap(battery=70, ac=False), snap(battery=65, ac=False)) == []


def test_disk_filling_up_is_notable():
    changes = changes_since(snap(disk=50), snap(disk=4))
    assert any("4gb left" in c for c in notable(changes))


def test_disk_staying_low_is_reported_only_once():
    assert changes_since(snap(disk=6), snap(disk=5)) == []


def test_memory_pressure_is_notable():
    changes = changes_since(snap(ram=70), snap(ram=95))
    assert any("memory is at 95%" in c for c in notable(changes))


def test_returning_to_the_keyboard_is_noticed_but_not_notable():
    changes = changes_since(snap(idle=1800), snap(idle=5))
    assert any("back at the keyboard" in c for c in texts(changes))
    assert notable(changes) == []


def test_going_away_is_noticed_but_not_notable():
    changes = changes_since(snap(idle=10), snap(idle=3600))
    assert any("away from the keyboard" in c for c in texts(changes))
    assert notable(changes) == []


def test_elapsed_time_is_appended_when_something_changed():
    changes = changes_since(snap(at=0, cpu=10), snap(at=300, cpu=95))
    assert any("over the last 5 minutes" in c for c in texts(changes))


def test_change_stringifies_to_its_text():
    assert str(Change("something happened", notable=True)) == "something happened"


def test_ambient_changes_alone_never_trigger_a_consultation():
    """The gate that keeps the ambient loop from costing money all day."""
    ambient = changes_since(
        snap(window="chrome.exe", idle=1800), snap(window="Code.exe", idle=5)
    )
    assert ambient, "these are still worth recording as context"
    assert not any(c.notable for c in ambient)
