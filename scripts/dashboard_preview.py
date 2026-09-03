"""Open the real dashboard with made up data, to look at it.

    python scripts/dashboard_preview.py            a normal session
    python scripts/dashboard_preview.py asking     one waiting on your yes
    python scripts/dashboard_preview.py fresh      nothing has happened yet
    python scripts/dashboard_preview.py unasked    a grant spent on something else

Nothing here talks to Claude, the microphone or the speakers. It is the same
widget the tray opens, fed a dictionary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper.ui.dashboard import Dashboard

CHAT = [
    ("you", "what is my battery doing"),
    ("vasper", "Full, and on mains."),
    ("you", "which windows do I have open"),
    ("vasper", "Chrome, Windows Terminal, VS Code and Settings."),
    ("you", "open spotify"),
    ("vasper", "Opened Spotify."),
    ("you", "what am I looking at"),
    ("vasper", "The dashboard you are building, in a terminal."),
]

SCENES = {
    "normal": (dict(paused=False, awake=True, uptime_s=8130, turns=18,
                    cost_usd=0.34, approvals=2, refusals=1, undos=0,
                    learned=3, pending=""), CHAT),
    "asking": (dict(paused=False, awake=True, uptime_s=622, turns=6,
                    cost_usd=0.11, approvals=0, refusals=0, learned=1,
                    pending="create notes dot txt, which lets me write to "
                            "other files too"),
               CHAT[:4] + [("you", "save that as a note")]),
    "fresh": (dict(paused=False, awake=False, uptime_s=12, turns=0,
                   cost_usd=0.0, pending=""), []),
    "unasked": (dict(paused=False, awake=False, uptime_s=4210, turns=11,
                     cost_usd=0.22, approvals=1, unasked=2, pending=""), CHAT),
}


def main() -> int:
    scene = (sys.argv[1] if len(sys.argv) > 1 else "normal").lower()
    if scene not in SCENES:
        print(f"unknown scene {scene!r}. try: " + ", ".join(SCENES))
        return 1
    data, chat = SCENES[scene]

    board = Dashboard(lambda: data, name="Vasper",
                      transcript=lambda: tuple(chat),
                      on_quit=lambda: board.close())
    board.show()
    print(f"showing '{scene}'. close the window to exit.")
    try:
        while board.open:
            time.sleep(0.2)
    except KeyboardInterrupt:
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
