"""Vasper's hands: one command line tool, called through the shell.

Claude Code already has a terminal. What it does not have is a mouse, a
keyboard, or any idea which windows are open, and that gap is the whole reason
this file exists. `vasper windows`, `vasper open`, `vasper screenshot` and the
rest give it those, as ordinary shell commands.

## Why a shell command rather than a tool schema

Because the CLI transport does not take custom tools, and because routing them
through the shell means they inherit the consent gate for free instead of
needing a second one written beside it. `vasper click` is refused by
`--permission-mode manual` exactly like `rm` is, becomes the spoken question
"I want to run vasper click, do I do this for you", and lands in the audit log
with everything else. A separate tool layer would have had to reimplement all
of that and would have got it wrong.

`vasper` is registered in `_SUBCOMMAND_TOOLS`, so a grant is
`Bash(vasper click:*)` rather than bare `Bash(vasper:*)`. Approving a click
cannot be spent on typing.

## Why win32 to look and UIA only to touch

Measured on this machine, and the numbers are not close:

    pywinauto Desktop(backend="uia").windows()   120,902 ms
    win32gui.EnumWindows                              0.6 ms
    UIA attached to one window by handle             21   ms
    mss grab of a 2560x1600 screen                   33   ms

Enumerating the desktop through UIA is two minutes, which is not slow, it is
broken for anything a person is waiting on. So nothing here ever asks UIA for a
list. Windows are found with win32, and UIA is attached to a single window by
handle only once there is something to press.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .sensors.window import _SENSITIVE

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "var" / "shots"

# Start Menu trees, most specific first: something you pinned yourself should
# win over the same name shipped for every account on the machine.
START_MENUS = (
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
)

# Below this a "match" is a coincidence. Tuned so "chrome" finds Google Chrome
# and "asdf" finds nothing rather than the alphabetically nearest thing.
MATCH_FLOOR = 0.45


@dataclass(frozen=True)
class Window:
    handle: int
    pid: int
    process: str
    title: str


# --- looking ----------------------------------------------------------------


def visible_windows() -> list[Window]:
    """Every top level window with a title, via win32. Sub-millisecond."""
    import win32gui
    import win32process

    found: list[Window] = []

    def collect(handle: int, _) -> None:
        if not win32gui.IsWindowVisible(handle):
            return
        title = win32gui.GetWindowText(handle)
        if not title.strip():
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(handle)
        except Exception:
            pid = 0
        found.append(Window(handle, pid, _process_name(pid), title))

    win32gui.EnumWindows(collect, None)
    return found


def _process_name(pid: int) -> str:
    if not pid:
        return ""
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:
        return ""


def _safe_title(title: str) -> str:
    """Same redaction the ambient context uses.

    A window list is read far more often than it is acted on, and a password
    manager's title is exactly as revealing here as it is there.
    """
    return "[hidden]" if _SENSITIVE.search(title) else title


def find_window(needle: str) -> Window | None:
    """The best window for a substring, preferring an exact-ish title match."""
    needle = needle.strip().lower()
    if not needle:
        return None
    windows = visible_windows()
    exact = [w for w in windows if needle in w.title.lower()]
    if exact:
        # Shortest title wins: "Notepad" over "Untitled - Notepad - Save As".
        return min(exact, key=lambda w: len(w.title))
    scored = [
        (difflib.SequenceMatcher(None, needle, w.title.lower()).ratio(), w)
        for w in windows
    ]
    best_score, best = max(scored, key=lambda pair: pair[0], default=(0.0, None))
    return best if best_score >= MATCH_FLOOR else None


# --- the start menu ---------------------------------------------------------


def installed_apps() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for menu in START_MENUS:
        if not menu.is_dir():
            continue
        for link in menu.rglob("*.lnk"):
            key = link.stem.lower()
            if key not in seen:
                seen.add(key)
                out.append(link)
    return sorted(out, key=lambda p: p.stem.lower())


def match_app(name: str) -> tuple[Path | None, list[Path]]:
    """Best Start Menu entry for a spoken name, plus the runners up.

    Returns the runners up as well because "open code" is genuinely ambiguous
    and a wrong launch is more annoying than a question.
    """
    name = name.strip().lower()
    apps = installed_apps()
    if not name or not apps:
        return None, []

    starts = [a for a in apps if a.stem.lower().startswith(name)]
    contains = [a for a in apps if name in a.stem.lower() and a not in starts]
    ranked = starts + contains
    if not ranked:
        scored = sorted(
            apps,
            key=lambda a: difflib.SequenceMatcher(None, name, a.stem.lower()).ratio(),
            reverse=True,
        )
        top = scored[0]
        if difflib.SequenceMatcher(None, name, top.stem.lower()).ratio() < MATCH_FLOOR:
            return None, []
        ranked = scored[:4]
    return ranked[0], ranked[1:4]


# --- commands ---------------------------------------------------------------


def cmd_windows(args) -> int:
    windows = visible_windows()
    if not windows:
        print("no visible windows")
        return 0
    for w in windows:
        print(f"{w.handle}\t{w.process}\t{_safe_title(w.title)}")
    return 0


def cmd_apps(args) -> int:
    if args.query:
        best, others = match_app(args.query)
        if best is None:
            print(f"nothing installed matching {args.query!r}")
            return 1
        print(best.stem)
        for other in others:
            print(f"  also: {other.stem}")
        return 0
    for app in installed_apps():
        print(app.stem)
    return 0


def cmd_open(args) -> int:
    best, others = match_app(args.name)
    if best is None:
        print(f"nothing installed matching {args.name!r}")
        return 1
    try:
        os.startfile(str(best))  # noqa: S606 - a Start Menu shortcut the user asked for
    except OSError as exc:
        print(f"could not open {best.stem}: {exc}")
        return 1
    print(f"opened {best.stem}")
    if others:
        print("  (also matched: " + ", ".join(o.stem for o in others) + ")")
    return 0


def cmd_focus(args) -> int:
    import win32con
    import win32gui

    window = find_window(args.title)
    if window is None:
        print(f"no window matching {args.title!r}")
        return 1
    try:
        if win32gui.IsIconic(window.handle):
            win32gui.ShowWindow(window.handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window.handle)
    except Exception:
        # Windows refuses foreground changes from a process that does not own
        # the current foreground window, and reports it as errno 2, "the system
        # cannot find the file specified", which is nonsense and would be
        # repeated to the user as though the window had vanished. Say what
        # actually happened instead.
        print(f"windows would not bring {_safe_title(window.title)!r} to the "
              "front from the background. it is still open, and clicking it "
              "once will do it.")
        return 1
    print(f"focused {_safe_title(window.title)}")
    return 0


def cmd_screenshot(args) -> int:
    import mss

    SHOTS.mkdir(parents=True, exist_ok=True)
    # Milliseconds, not seconds. Two captures in the same second is not an edge
    # case here: "screenshot the screen, now screenshot that window" is one
    # spoken request, and at second resolution the second one silently
    # overwrote the first and both paths pointed at the same image.
    target = SHOTS / f"shot-{time.time_ns() // 1_000_000}.png"

    region = None
    if args.window:
        window = find_window(args.window)
        if window is None:
            print(f"no window matching {args.window!r}")
            return 1
        import win32gui

        if win32gui.IsIconic(window.handle):
            # A minimized window's rect is parked off screen at around
            # (-21333, -21333), so grabbing it produced a 158x26 sliver of
            # nothing and wrote a 91 byte png that looked like a screenshot.
            # Saying so lets the caller focus it and ask again, which is the
            # only thing that can actually work.
            print(f"{_safe_title(window.title)!r} is minimized, focus it first")
            return 1

        left, top, right, bottom = win32gui.GetWindowRect(window.handle)
        width, height = right - left, bottom - top
        if width < 8 or height < 8:
            print(f"{_safe_title(window.title)!r} has no visible area to capture")
            return 1
        region = {"left": left, "top": top, "width": width, "height": height}

    with mss.MSS() as sct:
        shot = sct.grab(region or sct.monitors[1])
        mss.tools.to_png(shot.rgb, shot.size, output=str(target))

    _prune_shots()
    # The path, and only the path. Claude reads it with the Read tool, which is
    # how a screenshot reaches the model without the transport carrying images.
    print(target)
    return 0


def _prune_shots(keep: int = 20) -> None:
    """Screens are 2560x1600 and this runs whenever asked. Do not fill the disk."""
    shots = sorted(SHOTS.glob("shot-*.png"), key=lambda p: p.stat().st_mtime)
    for stale in shots[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasper", description="Vasper's hands on this machine."
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("windows", help="list visible windows").set_defaults(run=cmd_windows)

    apps = subs.add_parser("apps", help="list or search installed apps")
    apps.add_argument("query", nargs="?", default="")
    apps.set_defaults(run=cmd_apps)

    opener = subs.add_parser("open", help="launch an app by name")
    opener.add_argument("name")
    opener.set_defaults(run=cmd_open)

    focus = subs.add_parser("focus", help="bring a window to the front")
    focus.add_argument("title")
    focus.set_defaults(run=cmd_focus)

    shot = subs.add_parser("screenshot", help="capture the screen or one window")
    shot.add_argument("--window", default="")
    shot.set_defaults(run=cmd_screenshot)

    return parser


def _match_the_real_pixels() -> None:
    """Tell Windows this process counts pixels the way the screen does.

    Measured here: this display runs at 150%. A DPI unaware process is told a
    window is at (234, 234) 392 wide, while it is really at (351, 351) and 588
    wide, because Windows hands out logical coordinates and quietly scales.
    mss grabs physical pixels. So `screenshot --window` was capturing a region
    a third too small, offset up and to the left, and returning a picture of
    whatever happened to be there instead: usually the window behind.

    Deliberately called from `main` rather than at import. This runs as its own
    short lived process through the shim, so the setting dies with it. Making
    the assistant itself DPI aware would change nothing about what it hears and
    would render the tkinter dashboard at a third of its intended size.
    """
    import ctypes

    for attempt in (
        # Per monitor v2, the only one that is right on a mixed DPI desktop.
        lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)),
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
        lambda: ctypes.windll.user32.SetProcessDPIAware(),
    ):
        try:
            if attempt():
                return
        except Exception:
            continue


def _force_utf8() -> None:
    """Window titles are full of unicode and Windows still defaults to cp1252.

    Real titles on this machine right now carry U+2733 and U+25D1, and printing
    them through a redirected pipe raised UnicodeEncodeError and returned a
    failure for a command that had worked. The brain sets PYTHONIOENCODING for
    its own child, but this runs as a grandchild through a .cmd shim, so it
    settles the question itself rather than trusting what it inherited.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _match_the_real_pixels()
    _force_utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except Exception as exc:
        # This is read by a model, not a person. A traceback would be parsed as
        # output and reported as a result.
        print(f"{args.command} failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
