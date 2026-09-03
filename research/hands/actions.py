"""The action space: click, type, hotkey, launch.

Four verbs, matching the brief. Every one of them goes through the guard before
it touches the machine, and every one of them lands in the trace whether it ran
or not. A blocked action is a recorded event, not a silent no-op, because a run
whose success depended on being stopped from deleting something is a different
result than one that was never asked to.

Coordinates rather than control handles for the click. A handle is only
available in UIA mode, and Experiment 1 compares UIA against a screenshot
parser that has nothing but a box. Clicking by centre point is the only method
all three modes can share, so it is the only method any of them may use;
letting mode (a) click by handle would be measuring two different action
layers, not two different observations.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .elements import Element
from .guard import Guard

# pywinauto's send_keys treats these as syntax. Typing a literal one means
# wrapping it in braces, and forgetting to is how "50% off" becomes an alt
# chord that opens a menu.
_SEND_KEYS_SPECIAL = "^%+~(){}[]"

# "ctrl" and friends spelled the way a planner writes them, mapped to the way
# send_keys wants them.
_MODIFIERS = {"ctrl": "^", "control": "^", "alt": "%", "shift": "+", "win": "{VK_LWIN}"}

# Keys that need braces. Single printable characters do not.
_NAMED_KEYS = {
    "enter": "{ENTER}", "return": "{ENTER}", "tab": "{TAB}", "esc": "{ESC}",
    "escape": "{ESC}", "space": "{SPACE}", "backspace": "{BACKSPACE}",
    "delete": "{DELETE}", "del": "{DELETE}", "home": "{HOME}", "end": "{END}",
    "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
    "pgup": "{PGUP}", "pgdn": "{PGDN}",
    **{f"f{n}": "{F%d}" % n for n in range(1, 13)},
}


def escape_text(text: str) -> str:
    """Make arbitrary text safe to hand to send_keys."""
    return "".join("{%s}" % ch if ch in _SEND_KEYS_SPECIAL else ch for ch in text)


def to_send_keys(combo: str) -> str:
    """"ctrl+shift+n" into send_keys syntax.

    Raises on an unrecognised key rather than guessing. A silently mistyped
    hotkey lands somewhere unpredictable, and in an unattended run there is
    nobody to notice the window that just opened.
    """
    parts = [part.strip().lower() for part in combo.split("+") if part.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    prefix = ""
    for part in parts[:-1]:
        if part not in _MODIFIERS:
            raise ValueError(f"{part!r} is not a modifier in {combo!r}")
        prefix += _MODIFIERS[part]
    final = parts[-1]
    if final in _NAMED_KEYS:
        return prefix + _NAMED_KEYS[final]
    if len(final) == 1:
        return prefix + final
    raise ValueError(f"{final!r} is not a key this understands, in {combo!r}")


@dataclass
class Result:
    ok: bool
    detail: str = ""
    latency_s: float = 0.0
    blocked: bool = False


@dataclass
class Actions:
    """The hands. Holds the guard and the trace so no call site can skip them."""

    guard: Guard
    trace: object = None
    task_id: str = ""
    dry_run: bool = False
    performed: list[dict] = field(default_factory=list)

    def _record(self, step: int, verb: str, args: dict, element, result: Result):
        # The element travels with the record, not just its id, because the
        # distiller has to store *what* was clicked. An eid is meaningless by
        # the next observation and a coordinate is meaningless by the next
        # window, so a skill built from either would be broken on arrival.
        self.performed.append({
            "verb": verb,
            "args": args,
            "ok": result.ok,
            "element": element.to_json() if element is not None else None,
        })
        if self.trace is None:
            return
        if result.blocked:
            self.trace.blocked(self.task_id, step, verb, args, result.detail)
        else:
            self.trace.action(self.task_id, step, verb, args, element=element,
                              latency_s=result.latency_s, ok=result.ok,
                              detail=result.detail)

    def _run(self, step: int, verb: str, args: dict, element, body) -> Result:
        decision = self.guard.check(verb, args, element)
        if not decision:
            result = Result(False, decision.reason, blocked=True)
            self._record(step, verb, args, element, result)
            return result

        if self.dry_run:
            result = Result(True, "dry run, nothing performed")
            self._record(step, verb, args, element, result)
            return result

        started = time.perf_counter()
        try:
            body()
            result = Result(True, decision.reason, time.perf_counter() - started)
        except Exception as exc:
            result = Result(False, f"{type(exc).__name__}: {exc}",
                            time.perf_counter() - started)
        self._record(step, verb, args, element, result)
        return result

    def click(self, element: Element, step: int = 0, *, button: str = "left") -> Result:
        x, y = element.rect.center

        def body():
            from pywinauto import mouse

            mouse.click(button=button, coords=(x, y))

        return self._run(step, "click", {"eid": element.eid, "at": [x, y],
                                         "button": button}, element, body)

    def type_text(self, text: str, step: int = 0) -> Result:
        def body():
            from pywinauto import keyboard

            keyboard.send_keys(escape_text(text), with_spaces=True,
                               pause=0.01)

        return self._run(step, "type", {"text": text}, None, body)

    def hotkey(self, combo: str, step: int = 0) -> Result:
        # Translated before the guard runs, so an unparseable combo fails as a
        # bad argument rather than being waved through as a non-destructive
        # string the guard did not recognise.
        keys = to_send_keys(combo)

        def body():
            from pywinauto import keyboard

            keyboard.send_keys(keys, pause=0.01)

        return self._run(step, "hotkey", {"keys": combo, "sent": keys}, None, body)

    def launch(self, target: str, step: int = 0) -> Result:
        """Start something, then make sure it is actually the window in front.

        The waiting and focusing are not politeness. The first live run failed
        exactly here: a progress dialog belonging to something else held the
        foreground, Notepad opened behind it, and because every observation
        reads the foreground window the agent could not see what it had just
        started. It launched Notepad three times and burned eight of its ten
        steps before the window happened to surface. UFO2 answers this with an
        isolated virtual desktop; this is the cheap version of the same idea.
        """
        def body():
            import win32gui

            before = set()

            def collect(handle, into):
                if win32gui.IsWindowVisible(handle) and win32gui.GetWindowText(handle):
                    into.add(handle)
                return True

            win32gui.EnumWindows(collect, before)

            # startfile rather than spawning a shell: it takes a document or an
            # app the same way the Start menu does, and it gives a planner no
            # way to smuggle a command line through a filename.
            os.startfile(target)  # noqa: S606

            # Poll for a window that was not there before, rather than sleeping
            # a fixed guess. A cold Notepad and a warm one differ by seconds.
            deadline = time.time() + 8.0
            fresh = None
            while time.time() < deadline:
                time.sleep(0.25)
                now = set()
                win32gui.EnumWindows(collect, now)
                new = now - before
                if new:
                    fresh = max(new, key=lambda h: len(win32gui.GetWindowText(h)))
                    break

            if fresh is None:
                # It may have reused an existing window, which is normal for
                # apps with a single instance. Not an error, but the caller
                # gets no focus guarantee either.
                return

            try:
                from pywinauto import Desktop

                # set_focus does the restore-and-raise dance that a bare
                # SetForegroundWindow is not allowed to do from a background
                # process.
                Desktop(backend="uia").window(handle=fresh).set_focus()
            except Exception:
                # Focus is best effort. Failing to raise a window is worth far
                # less than failing the whole action for it.
                pass
            time.sleep(0.4)

        return self._run(step, "launch", {"target": target}, None, body)

    def wait(self, seconds: float, step: int = 0) -> Result:
        seconds = max(0.0, min(float(seconds), 10.0))

        def body():
            time.sleep(seconds)

        return self._run(step, "wait", {"seconds": seconds}, None, body)
