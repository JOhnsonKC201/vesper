"""Writing a Windows shortcut, which is COM whichever way you cut it.

Two places want one and they are the same six lines with a different
destination: the Startup folder, so Vesper comes back after a reboot, and the
desktop, so it can be started by hand. `autostart.py` had them first; this is
where they live now so that fixing one fixes both. The desktop icon is the
reason it moved, and it immediately paid for itself, because it needed one
thing the startup shortcut never did: an icon.

Every shortcut points at `run_silent.vbs` rather than at `run.bat`, so nothing
flashes up a console window. That matters more for the desktop icon than for
autostart: a black window that appears and vanishes on a double click reads as
a crash, and Vesper's real answer is the tray icon that appears a moment later.
"""

from __future__ import annotations

import os
from pathlib import Path

LAUNCHER = "run_silent.vbs"


def write(target: Path, root: Path, *, description: str,
          icon: Path | None = None) -> tuple[bool, str]:
    """Point `target` at Vesper's silent launcher. Returns (ok, what to say).

    Never raises. Every caller is either a CLI flag or a self check, and both
    would rather print why than hand back a traceback.
    """
    launcher = Path(root) / LAUNCHER
    if not launcher.exists():
        # A shortcut to nothing is worse than no shortcut: it fails silently,
        # and for the autostart one that is at login, where nobody is watching.
        return False, f"{LAUNCHER} is missing from {root}"

    target = Path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot reach {target.parent}: {exc}"

    try:
        import win32com.client
    except ImportError:
        return False, "pywin32 is not available, so the shortcut cannot be created"

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        link = shell.CreateShortCut(str(target))
        # wscript, not the .vbs directly, so the association cannot be hijacked
        # by whatever else has claimed .vbs on this machine.
        link.TargetPath = str(Path(os.environ.get("WINDIR", "C:/Windows")) /
                              "System32" / "wscript.exe")
        link.Arguments = f'"{launcher}"'
        link.WorkingDirectory = str(root)
        link.Description = description
        if icon is not None and Path(icon).exists():
            # Without this the icon is wscript's, which is a generic scroll and
            # says nothing about what the thing does. `,0` is the index into
            # the file, and an .ico has exactly one image group.
            link.IconLocation = f"{icon},0"
        link.save()
    except Exception as exc:
        return False, f"could not create the shortcut: {exc}"

    return True, str(target)


def remove(target: Path) -> tuple[bool, str]:
    """Delete a shortcut. Succeeds if it was already gone, which is the same
    end state and not worth an error."""
    target = Path(target)
    if not target.exists():
        return True, "was not installed"
    try:
        target.unlink()
    except OSError as exc:
        return False, f"could not remove it: {exc}"
    return True, str(target)
