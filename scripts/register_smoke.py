"""See what a turn actually carries, in each situation that changes it.

    python scripts/register_smoke.py

Prints the real context block for a handful of sessions, so you can read what
the model reads. Nothing is mocked except the clock and the session history:
the sensor readings are this machine, right now.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper import register
from vesper.register import LONG_SESSION_S, Register
from vesper.sensors import snapshot as sensors


def at(hour: int, minute: int = 0) -> time.struct_time:
    return time.struct_time((2026, 9, 3, hour, minute, 0, 2, 246, -1))


def show(title: str, reg: Register, *, now: float, clock: time.struct_time) -> None:
    block = register.attach(sensors.context_block(), reg.line(now=now, clock=clock))
    print(f"\n\033[1m{title}\033[0m")
    print("-" * 68)
    for line in block.splitlines():
        marker = "  > " if line.startswith("how to pitch") else "    "
        print(f"{marker}{line}")
    print(f"    [{len(block)} chars]")


def main() -> int:
    print("what one turn carries, in six situations\n")
    print("the first four lines of each block are live readings from this machine.")
    print("the marked line is the only thing the register adds.")

    show("1. ordinary afternoon turn (the common case: it adds nothing)",
         Register(started_at=0.0), now=30.0, clock=at(14, 20))

    mid = Register(started_at=0.0)
    mid.note_turn("Code.exe", now=100.0)
    mid.note_turn("Code.exe", now=110.0)
    show("2. mid exchange, you came straight back",
         mid, now=118.0, clock=at(14, 22))

    back = Register(started_at=0.0)
    back.note_idle(1800.0)
    show("3. back after half an hour away",
         back, now=40.0, clock=at(15, 0))

    late = Register(started_at=0.0)
    show("4. quarter to midnight, four hours in",
         late, now=LONG_SESSION_S + 900, clock=at(23, 45))

    stuck = Register(started_at=0.0)
    stuck.note_failure()
    stuck.note_failure()
    show("5. two failures running (crowds everything else out)",
         stuck, now=LONG_SESSION_S + 900, clock=at(23, 50))

    heads_down = Register(started_at=0.0)
    for tick in range(5):
        heads_down.note_turn("Code.exe", now=float(tick))
    show("6. five turns without leaving the editor",
         heads_down, now=600.0, clock=at(10, 30))

    print("\nnote what is not in any of them: how you feel, or how it should feel")
    print("back. Every line is an instruction about delivery.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
