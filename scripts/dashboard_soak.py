"""Open and close the dashboard until it breaks, or prove it does not.

The failure this hunts cannot be caught by pytest. A Tk object touched from the
wrong thread makes Tcl call `Tcl_Panic`, which does not raise anything: it
aborts the process. There is no traceback and no exception to assert on, so the
only honest test is to do it many times and check the process is still alive.

That failure has happened here before, and took three wrong fixes to
understand: it is not just `destroy()`, it is *any* cross-thread Tk call,
including `after()`. Every rule in `ui/dashboard.py` exists because of it, and
the voice picker added widgets, a `StringVar` and a worker thread to the file
that learned those rules the hard way.

    python scripts/dashboard_soak.py            # 20 cycles
    python scripts/dashboard_soak.py --cycles 5 # fewer, windows do flash

An abort shows up as this script dying without printing "survived". A clean run
prints the count and exits 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper.ui.dashboard import Dashboard  # noqa: E402


class FakePanel:
    """A voice panel that answers instantly, so the soak measures Tk only."""

    def __init__(self):
        self.chosen = []

    def list(self):
        from vesper.tts import eleven_api

        return list(eleven_api.KNOWN_VOICES)

    def current(self):
        return "George"

    def choose(self, voice_id):
        self.chosen.append(voice_id)

    def preview(self, voice_id):
        return "previewed"

    def budget(self):
        return 1_234, 9_000

    def status(self):
        return "7,766 of 9,000 characters left this month"


class FakeLocalPanel(FakePanel):
    """The local picker: many rows, and no allowance, so no spend bar.

    Worth soaking separately. The bar is now built only when there is a budget,
    and a widget that exists on one path and not the other is exactly the shape
    of mistake that leaves something outside `_fields` and finalises it on the
    wrong thread later.
    """

    def list(self):
        from vesper import config
        from vesper.tts import catalog

        return list(catalog.discover(config.Config().voices_path()))

    def current(self):
        rows = self.list()
        return rows[0].label if rows else ""

    def budget(self):
        return 0, 0

    def status(self):
        return "5 voices here, and none of them need the network"


def snapshot():
    return {
        "paused": False, "uptime_s": 42, "turns": 3, "cost_usd": 0.12,
        "interruptions": 0, "approvals": 1, "refusals": 0, "undos": 0,
        "voice_rejections": 0, "echo_rejections": 0,
        "last_heard": "what is the time", "pending": "", "voice": "eleven:George",
        "learned": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--no-voices", action="store_true",
                        help="soak without the picker, to isolate it")
    parser.add_argument("--local", action="store_true",
                        help="soak the local picker, which draws no spend bar")
    args = parser.parse_args()

    panel = None
    if not args.no_voices:
        panel = FakeLocalPanel() if args.local else FakePanel()
    print(f"opening and closing {args.cycles} times, "
          f"{'without' if panel is None else 'with'} the voice picker")

    for cycle in range(1, args.cycles + 1):
        board = Dashboard(snapshot, voices=panel, name="Vesper soak")
        board.show()

        # Let the window build and tick at least once, so _refresh and
        # _apply_voice both actually run rather than the window being torn
        # down before it drew anything.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not board.open:
            time.sleep(0.02)
        time.sleep(0.45)

        # Raising is a cross-thread request, which is the exact operation the
        # Event-flag discipline exists for. Do it every time.
        board.show()
        time.sleep(0.2)

        board.close(timeout=5.0)
        if board.open:
            print(f"cycle {cycle}: the thread did not stop, which is the "
                  f"condition that finalises Tk objects on the wrong thread")
            return 1
        print(f"  cycle {cycle} ok", flush=True)

    print(f"survived {args.cycles} cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
