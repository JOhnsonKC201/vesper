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
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import console
from .sensors.window import _SENSITIVE
from . import hands

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "var" / "shots"
# The app list, cached between runs because `vasper` is a fresh process
# every time and rebuilding it costs 0.66s. See `installed_apps`.
APP_INDEX = ROOT / "var" / "apps.json"

# Start Menu trees, most specific first: something you pinned yourself should
# win over the same name shipped for every account on the machine.
START_MENUS = (
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
)

# Below this a "match" is a coincidence. Substring matches never reach this
# floor; it only decides the fuzzy fallback for typos. It was 0.45, and at 0.45
# "notepad" scored 0.57 against "OneNote" and launched it, live, in front of
# the user (2026-09-05). A wrong launch is worse than "nothing matches", so the
# floor sits where "chorme" still finds Chrome (0.83) and OneNote does not.
MATCH_FLOOR = 0.66


@dataclass(frozen=True)
class App:
    """Something Windows will launch by name: a Start Menu shortcut, or an
    installed app the Start menu lists without a shortcut, such as the Store
    apps Notepad and Calculator are on Windows 11.

    `stem` matches `Path.stem` so the matcher can treat shortcuts and apps the
    same way; `str()` is what `os.startfile` needs.
    """

    name: str
    target: str

    @property
    def stem(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.target


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


def store_apps() -> list[App]:
    """Every app the Start menu itself lists, from the shell's AppsFolder.

    This is where Windows 11 keeps the apps that have no .lnk anywhere:
    Notepad, Calculator, Terminal, Photos, and everything from the Store. Each
    comes with an application user model id, and `shell:AppsFolder\\<id>` is
    how the Start menu launches it, which is exactly what `vasper open` is
    allowed to be: the same double click the user could make.
    """
    try:
        import win32com.client

        shell = win32com.client.Dispatch("Shell.Application")
        items = shell.NameSpace("shell:AppsFolder").Items()
        apps: list[App] = []
        for index in range(items.Count):
            item = items.Item(index)
            name, path = str(item.Name or "").strip(), str(item.Path or "").strip()
            if name and path:
                apps.append(App(name, "shell:AppsFolder\\" + path))
        return apps
    except Exception:
        # No shell COM, or a locked down profile. The Start Menu shortcuts
        # still work on their own, as they did before this existed.
        return []


def _scan_apps() -> list[Path | App]:
    """Start Menu shortcuts first, then the apps only the AppsFolder knows.

    A shortcut wins over an AppsFolder entry of the same name because the
    shortcut is the thing the user, or the installer, put there on purpose.
    """
    out: list[Path | App] = []
    seen: set[str] = set()
    for menu in START_MENUS:
        if not menu.is_dir():
            continue
        for link in menu.rglob("*.lnk"):
            key = link.stem.lower()
            if key not in seen:
                seen.add(key)
                out.append(link)
    for app in store_apps():
        key = app.stem.lower()
        if key not in seen:
            seen.add(key)
            out.append(app)
    return sorted(out, key=lambda p: p.stem.lower())


def _menus_stamp() -> str:
    """A cheap fingerprint of the Start Menu trees, for knowing when to rescan.

    The top directory mtimes rather than a walk. Windows touches a directory
    when an entry is added or removed under it, which is when this index goes
    wrong, and it is two stat calls rather than a recursive scan. An install
    that only changes a file deep in the tree without touching these will be
    missed until something else does; a wrong app name for an hour is a much
    smaller cost than 0.66s on every open.
    """
    parts = []
    for menu in START_MENUS:
        try:
            parts.append(f"{menu}:{menu.stat().st_mtime_ns}")
        except OSError:
            parts.append(f"{menu}:none")
    return "|".join(parts)


def installed_apps() -> list[Path | App]:
    """The app list, read from `var/apps.json` unless the Start Menu changed.

    `match_app` calls this, `cmd_open` calls `match_app`, and `vasper` is a
    fresh process every time, so nothing could be kept in memory between calls.
    Measured on this machine: `vasper apps` 0.66s against `vasper windows`
    0.10s, and the whole difference is this walking two Start Menu trees and
    enumerating the shell AppsFolder over COM.

    That was paid before every single `vasper open`, with somebody waiting to
    hear the app start.
    """
    stamp = _menus_stamp()
    cached = _read_app_index(stamp)
    if cached is not None:
        return cached
    apps = _scan_apps()
    _write_app_index(stamp, apps)
    return apps


def _read_app_index(stamp: str) -> list[Path | App] | None:
    """The saved list, or None if it is missing, stale, or unreadable."""
    try:
        with APP_INDEX.open(encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or saved.get("stamp") != stamp:
        return None
    out: list[Path | App] = []
    for entry in saved.get("apps", []):
        try:
            if entry["kind"] == "link":
                out.append(Path(entry["path"]))
            else:
                out.append(App(entry["name"], entry["target"]))
        except (KeyError, TypeError):
            return None  # damaged, or written by an older version
    return out


def _write_app_index(stamp: str, apps: list[Path | App]) -> None:
    """Save the list. Never raises: a cache that cannot be written is not a
    reason to fail to open an app."""
    entries = [
        {"kind": "app", "name": a.name, "target": a.target}
        if isinstance(a, App)
        else {"kind": "link", "path": str(a)}
        for a in apps
    ]
    try:
        APP_INDEX.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed, so a reader never sees half a file. Two
        # of these really can overlap: Claude runs `vasper` from a shell.
        temporary = APP_INDEX.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump({"stamp": stamp, "apps": entries}, handle)
        temporary.replace(APP_INDEX)
    except (OSError, ValueError):
        pass


def _similarity(needle: str, name: str) -> float:
    """How close a spoken name is to an app name, or to any one word of it.

    Whole-name only, "chorme" scores 0.53 against "google chrome" and a typo
    stops finding Chrome; word by word it scores 0.83 against "chrome". The
    per-word view does not rescue "notepad" against "onenote" (0.57), which is
    the wrong launch the floor exists to prevent.
    """
    candidates = [name, *name.split()]
    return max(difflib.SequenceMatcher(None, needle, c).ratio() for c in candidates)


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
        scored = sorted(apps, key=lambda a: _similarity(name, a.stem.lower()), reverse=True)
        top = scored[0]
        if _similarity(name, top.stem.lower()) < MATCH_FLOOR:
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


# An address the browser should open. Two shapes: a full one with a scheme, or
# a bare host with a dot in it, optionally with a path or a query. Only http and
# https: `file:` would open a local file and `javascript:` would run code, and
# neither is "go to a site". The scheme is composed rather than written out
# because `tests/test_privacy.py` scans this package for baked-in destinations
# and the marker it looks for is the scheme plus the slashes; a scheme on its
# own names no destination, and keeping the marker out keeps that check honest.
_FULL_ADDRESS = re.compile(r"^(?P<scheme>[a-z][a-z0-9+-]*):(?P<rest>.*)$", re.I)
_BARE_HOST = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(:\d+)?(/\S*)?$", re.I)


def web_address(name: str) -> str | None:
    """The address `open` should hand to the browser, or None for an app name.

    "Go to my Google Calendar" (2026-09-08 09:57) went through ctrl+t, a
    permission question and a lapse, and never arrived. An address is a double
    click the user could make, so it is free, like opening an app.
    """
    text = (name or "").strip().strip("\"'")
    if not text or any(ch.isspace() for ch in text):
        return None
    full = _FULL_ADDRESS.match(text)
    if full:
        scheme = full.group("scheme").lower()
        rest = full.group("rest")
        if scheme in ("http", "https") and rest.startswith("//") and len(rest) > 2:
            return text
        return None
    if _BARE_HOST.match(text):
        return "https:" + "//" + text
    return None


def cmd_open(args) -> int:
    address = web_address(args.name)
    if address is not None:
        # Before the app matcher on purpose: "google.com" scores well above the
        # fuzzy floor against "Google Chrome", and would have launched the
        # browser with no page in it.
        try:
            os.startfile(address)  # noqa: S606
        except OSError as exc:
            print(f"could not open {address}: {exc}")
            return 1
        print(f"opened {address} in the browser")
        return 0
    best, others = match_app(args.name)
    if best is None:
        print(f"nothing installed matching {args.name!r}")
        return 1
    try:
        # A Start Menu shortcut, or shell:AppsFolder\<id> for an app the Start
        # menu lists without one. Both are the double click the user could make.
        os.startfile(str(best))  # noqa: S606
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
    # Check rather than trust. The call above rarely raises even when Windows
    # ignores it, and on 2026-09-05 that read as "focused LinkedIn" while the
    # terminal stayed on top, so the brain went on to describe a screenshot of
    # the wrong window. Polled briefly, because the switch is not instant.
    deadline = time.time() + 0.4
    while win32gui.GetForegroundWindow() != window.handle:
        if time.time() >= deadline:
            front = win32gui.GetWindowText(win32gui.GetForegroundWindow()) or "another window"
            print(f"{_safe_title(window.title)!r} is open, but windows kept "
                  f"{_safe_title(front)!r} in front. clicking it once will do it.")
            return 1
        time.sleep(0.05)
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


# --- the hands: look, then act, in front of the user -------------------------
#
# Every command below except `look` is refused by the brain's allowlist and
# becomes the spoken question "use the mouse and keyboard to ...". A yes covers
# all of them for one turn. Each one is performed so the person can see it
# coming: the cursor glides to its target before a click, and text is typed a
# character at a time. That is deliberate. A click that lands from nowhere
# cannot be stopped; one you watch travel across the screen can.


def _front_window() -> tuple[int, str]:
    import win32gui

    handle = win32gui.GetForegroundWindow()
    return handle, (win32gui.GetWindowText(handle) if handle else "")


def _elements_of(handle: int, limit: int = hands.MAX_ELEMENTS) -> list[hands.Element]:
    return hands.walk(hands.attach(handle), max_elements=limit)


def cmd_look(args) -> int:
    """What is on the front window, numbered, with where each thing is."""
    if args.window:
        window = find_window(args.window)
        if window is None:
            print(f"no window matching {args.window!r}")
            return 1
        handle, title = window.handle, window.title
    else:
        handle, title = _front_window()
    if not handle:
        print("no window in front")
        return 1
    if _SENSITIVE.search(title):
        # The same line the window list and the ambient context draw. A
        # password manager's controls are exactly as revealing as its title.
        print(f"the window in front is {_safe_title(title)}, and I do not read those")
        return 1
    if args.find:
        # A web page has hundreds of controls and the list is capped, so a
        # search reads deeper than the plain list and keeps only what matches.
        needle = args.find.lower()
        elements = [e for e in _elements_of(handle, 600) if needle in e.name.lower()][:40]
        print(f"window: {title}")
        if not elements:
            print(f"  nothing named like {args.find!r} in it")
            return 0
        for element in elements:
            print("  " + element.describe())
        return 0
    elements = _elements_of(handle, args.max)
    print(f"window: {title}")
    if not elements:
        print("  nothing readable in it, a canvas or a custom drawn surface; "
              "take a screenshot instead")
        return 0
    for element in elements:
        print("  " + element.describe())
    return 0


def cmd_read(args) -> int:
    """The text of the front window, or a named one: a web page, a document."""
    if args.window:
        window = find_window(args.window)
        if window is None:
            print(f"no window matching {args.window!r}")
            return 1
        handle, title = window.handle, window.title
    else:
        handle, title = _front_window()
    if not handle:
        print("no window in front")
        return 1
    if _SENSITIVE.search(title):
        print(f"the window in front is {_safe_title(title)}, and I do not read those")
        return 1
    text = hands.page_text(hands.attach(handle), max_chars=args.max_chars)
    print(f"window: {title}")
    if not text:
        print("  no readable text in it; take a screenshot instead")
        return 0
    print(text)
    return 0


def _resolve_target(tokens: list[str]) -> tuple[tuple[int, int] | None, str]:
    """`X Y` is a point; anything else is the name of a control on the front
    window, looked up fresh, because positions from an earlier look go stale
    the moment the window moves."""
    if len(tokens) == 2 and all(t.lstrip("-").isdigit() for t in tokens):
        return (int(tokens[0]), int(tokens[1])), f"{tokens[0]},{tokens[1]}"
    name = " ".join(tokens).strip().strip("\"'")
    handle, title = _front_window()
    if not handle:
        return None, "no window in front"
    if _SENSITIVE.search(title):
        return None, f"the window in front is {_safe_title(title)}, and I do not touch those"
    found, others = hands.find_element(_elements_of(handle), name)
    if found is not None:
        return found.center, repr(found.name or found.control_type)
    if others:
        return None, hands.ambiguity_line(name, title, others)
    return None, f"nothing called {name!r} on {title!r}. Run vasper look and pick from the list."


def _glide_to(point: tuple[int, int]) -> None:
    """Move the cursor to the point the way a hand would, so it is seen coming."""
    import win32api

    for step in hands.glide_path(tuple(win32api.GetCursorPos()), point):
        win32api.SetCursorPos(step)
        time.sleep(hands.GLIDE_STEP_MS / 1000.0)


def cmd_move(args) -> int:
    point, label = _resolve_target(args.target)
    if point is None:
        print(label)
        return 1
    _glide_to(point)
    print(f"pointing at {label}, {point[0]},{point[1]}")
    return 0


def cmd_click(args) -> int:
    from pywinauto import mouse

    point, label = _resolve_target(args.target)
    if point is None:
        print(label)
        return 1
    _glide_to(point)
    button = "right" if args.right else "left"
    if args.double:
        mouse.double_click(button=button, coords=point)
    else:
        mouse.click(button=button, coords=point)
    kind = ("double " if args.double else "") + ("right " if args.right else "")
    print(f"{kind}clicked {label} at {point[0]},{point[1]}")
    return 0


def cmd_type(args) -> int:
    """Type where the keyboard focus is, one character at a time."""
    from pywinauto import keyboard

    text = args.text
    if not text:
        print("nothing to type")
        return 1
    keyboard.send_keys(
        hands.escape_text(text), with_spaces=True, with_newlines=True,
        pause=hands.TYPE_PAUSE_S,
    )
    if args.enter:
        keyboard.send_keys("{ENTER}", pause=hands.TYPE_PAUSE_S)
    print(f"typed {len(text)} characters" + (" and pressed enter" if args.enter else ""))
    return 0


def cmd_key(args) -> int:
    from pywinauto import keyboard

    try:
        keys = hands.to_send_keys(args.combo)
    except ValueError as exc:
        print(f"cannot press {args.combo!r}: {exc}")
        return 1
    keyboard.send_keys(keys, pause=0.02)
    print(f"pressed {args.combo}")
    return 0


def cmd_scroll(args) -> int:
    import win32api
    from pywinauto import mouse

    notches = max(1, min(int(args.times), 20))
    distance = notches if args.direction == "up" else -notches
    mouse.scroll(coords=tuple(win32api.GetCursorPos()), wheel_dist=distance)
    print(f"scrolled {args.direction} {notches}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasper", description="Vasper's hands on this machine."
    )
    subs = parser.add_subparsers(dest="command", required=True)

    look = subs.add_parser("look", help="what is on the front window, numbered, with positions")
    look.add_argument("--window", default="", help="look at this window instead of the front one")
    look.add_argument("--max", type=int, default=hands.MAX_ELEMENTS)
    look.add_argument("--find", default="", help="only controls whose name contains this")
    look.set_defaults(run=cmd_look)

    reader = subs.add_parser("read", help="the text of the front window: a page, a document")
    reader.add_argument("--window", default="")
    reader.add_argument("--max-chars", type=int, default=6000)
    reader.set_defaults(run=cmd_read)

    click = subs.add_parser("click", help="glide to a control by name, or to X Y, and click")
    click.add_argument("target", nargs="+")
    click.add_argument("--right", action="store_true")
    click.add_argument("--double", action="store_true")
    click.set_defaults(run=cmd_click)

    move = subs.add_parser("move", help="point the cursor at a control or X Y without clicking")
    move.add_argument("target", nargs="+")
    move.set_defaults(run=cmd_move)

    typer = subs.add_parser("type", help="type text where the focus is, visibly")
    typer.add_argument("text")
    typer.add_argument("--enter", action="store_true", help="press enter afterwards")
    typer.set_defaults(run=cmd_type)

    key = subs.add_parser("key", help="press a key or combination: enter, esc, ctrl+l, alt+f4")
    key.add_argument("combo")
    key.set_defaults(run=cmd_key)

    scroll = subs.add_parser("scroll", help="scroll under the cursor")
    scroll.add_argument("direction", choices=("up", "down"))
    scroll.add_argument("--times", type=int, default=3)
    scroll.set_defaults(run=cmd_scroll)

    subs.add_parser("windows", help="list visible windows").set_defaults(run=cmd_windows)

    apps = subs.add_parser("apps", help="list or search installed apps")
    apps.add_argument("query", nargs="?", default="")
    apps.set_defaults(run=cmd_apps)

    opener = subs.add_parser(
        "open", help="launch an app by name, or open a web address in the browser"
    )
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


def main(argv: list[str] | None = None) -> int:
    _match_the_real_pixels()
    console.force_utf8()
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
