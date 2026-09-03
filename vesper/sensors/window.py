"""What the user is actually looking at.

This is the single highest-value signal for an assistant that is supposed to
know what is going on. "What is this error" only works if Vesper can see that
the foreground window is VS Code with auth.py open.

Window titles are also the most privacy-sensitive thing Vesper reads: they leak
document names, browser tabs and chat contents. So there is a redaction hook
here, and titles never leave the machine except inside the conversation with
Claude that the user deliberately started.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveWindow:
    title: str = ""
    process: str = ""
    pid: int = 0

    @property
    def known(self) -> bool:
        return bool(self.title or self.process)

    def describe(self) -> str:
        if not self.known:
            return "unknown window"
        if self.title and self.process:
            return f"{self.process}: {self.title}"
        return self.title or self.process


# Titles matching these are reported as the app name only. Cheap protection
# against the assistant reciting a password manager entry or a private chat.
# `inprivate` is Edge's word for incognito and was missing, so an Edge private
# window was reported by its real title while the Chrome equivalent was hidden.
# That mattered more once `vasper windows` began listing every window rather
# than only the focused one: the hole went from one window to all of them.
_SENSITIVE = re.compile(
    r"(password|passwd|secret|token|api[_ -]?key|credential|bitwarden|1password"
    r"|lastpass|keepass|banking|incognito|inprivate|private browsing"
    r"|private window)",
    re.IGNORECASE,
)


def redact(window: ActiveWindow) -> ActiveWindow:
    if window.title and _SENSITIVE.search(window.title):
        return ActiveWindow(title="[hidden]", process=window.process, pid=window.pid)
    return window


def active_window() -> ActiveWindow:
    """The foreground window, or an empty record if it cannot be read."""
    try:
        import win32gui
        import win32process
    except ImportError:
        return ActiveWindow()

    try:
        handle = win32gui.GetForegroundWindow()
        if not handle:
            return ActiveWindow()
        title = win32gui.GetWindowText(handle) or ""
        pid = 0
        process = ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(handle)
            if pid:
                import psutil

                process = psutil.Process(pid).name()
        except Exception:
            pass
        return redact(ActiveWindow(title=title.strip(), process=process, pid=pid))
    except Exception:
        return ActiveWindow()


def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input, system wide.

    Uses GetLastInputInfo rather than a global input hook. A hook would work but
    installing one to answer "are they at the desk" is disproportionate, and it
    is exactly the kind of thing that makes an assistant feel like spyware.
    """
    try:
        import ctypes

        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        ticks = ctypes.windll.kernel32.GetTickCount()
        return max(0.0, (ticks - info.dwTime) / 1000.0)
    except Exception:
        return 0.0
