"""Machine vitals.

Everything here is cheap enough to sample on every turn. Anything that needs a
subprocess or a network round trip does not belong in this file; Claude has a
shell and can go and get it when actually asked.

Values are formatted the way a person says them, because they end up being read
aloud. "About six hundred gigabytes free" beats "598.13 GB".
"""

from __future__ import annotations

import time
from dataclasses import dataclass


_NOT_REAL_LOAD = frozenset({"system idle process", "idle", ""})


@dataclass(frozen=True)
class Vitals:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_free_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    battery_percent: int | None = None
    on_ac_power: bool | None = None
    battery_minutes_left: int | None = None
    uptime_hours: float = 0.0
    top_processes: tuple[tuple[str, float], ...] = ()

    def describe(self) -> str:
        """A compact one-line summary for the context block."""
        parts = [f"cpu {self.cpu_percent:.0f}%", f"ram {self.ram_percent:.0f}%"]
        if self.battery_percent is not None:
            state = "on ac" if self.on_ac_power else "on battery"
            parts.append(f"battery {self.battery_percent}% {state}")
        parts.append(f"disk {self.disk_free_gb:.0f}gb free")
        return ", ".join(parts)


def _top_processes(limit: int = 3) -> tuple[tuple[str, float], ...]:
    """The heaviest processes by CPU.

    cpu_percent() without an interval returns usage since the last call for that
    process object, so the first sample is always zero. Callers that care sample
    twice; for ambient context a slightly stale ranking is fine and a blocking
    interval is not.
    """
    try:
        import psutil

        rows: dict[str, float] = {}
        for process in psutil.process_iter(["name", "cpu_percent"]):
            info = process.info
            name = info.get("name") or "?"
            # "System Idle Process" reports the CPU that is *not* being used,
            # summed across cores, so it always wins and always means nothing.
            if name.lower() in _NOT_REAL_LOAD:
                continue
            rows[name] = rows.get(name, 0.0) + (info.get("cpu_percent") or 0.0)
        ranked = sorted(rows.items(), key=lambda kv: kv[1], reverse=True)
        return tuple((name, round(pct, 1)) for name, pct in ranked[:limit] if pct > 1.0)
    except Exception:
        return ()


def vitals(*, include_processes: bool = True) -> Vitals:
    try:
        import psutil
    except ImportError:
        return Vitals()

    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\" if _is_windows() else "/")

        battery_percent = on_ac = minutes_left = None
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                battery_percent = int(battery.percent)
                on_ac = bool(battery.power_plugged)
                secs = battery.secsleft
                if secs is not None and secs >= 0:
                    minutes_left = int(secs // 60)
        except Exception:
            pass

        return Vitals(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_percent=memory.percent,
            ram_free_gb=memory.available / 1e9,
            disk_free_gb=disk.free / 1e9,
            disk_percent=disk.percent,
            battery_percent=battery_percent,
            on_ac_power=on_ac,
            battery_minutes_left=minutes_left,
            uptime_hours=max(0.0, (time.time() - psutil.boot_time()) / 3600.0),
            top_processes=_top_processes() if include_processes else (),
        )
    except Exception:
        return Vitals()


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def prime() -> None:
    """Take a throwaway CPU sample so the first real reading is not zero."""
    try:
        import psutil

        psutil.cpu_percent(interval=None)
        for process in psutil.process_iter(["cpu_percent"]):
            pass
    except Exception:
        pass
