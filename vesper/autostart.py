"""Starting with Windows, by putting one shortcut in one folder.

The Startup folder rather than the registry's `Run` key, deliberately. Both
work. The folder is somewhere you can look, and delete by hand, without knowing
that `HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run` exists. An
assistant that listens to your microphone from login should be easy to remove by
someone who has forgotten how it was installed, and this machine already has
`Echo Flow.lnk` installed exactly this way.

The shortcut points at `run_silent.vbs`, which was written for this and has been
sitting unused since the project started: it launches `run.bat` with a hidden
window so nothing flashes up at login.
"""

from __future__ import annotations

import os
from pathlib import Path

SHORTCUT_NAME = "Vesper.lnk"
LAUNCHER = "run_silent.vbs"


def startup_dir() -> Path:
    """Where Windows looks for things to run at login."""
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path:
    return startup_dir() / SHORTCUT_NAME


def is_installed() -> bool:
    return shortcut_path().exists()


def install(root: Path) -> tuple[bool, str]:
    """Create the login shortcut. Returns (ok, something to tell the user)."""
    launcher = Path(root) / LAUNCHER
    if not launcher.exists():
        return False, f"{LAUNCHER} is missing from {root}"

    target = shortcut_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot reach the startup folder: {exc}"

    try:
        import win32com.client
    except ImportError:
        return False, "pywin32 is not available, so the shortcut cannot be created"

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(target))
        # wscript, not the .vbs directly, so the association cannot be hijacked
        # by whatever else has claimed .vbs on this machine.
        shortcut.TargetPath = str(Path(os.environ.get("WINDIR", "C:/Windows")) /
                                  "System32" / "wscript.exe")
        shortcut.Arguments = f'"{launcher}"'
        shortcut.WorkingDirectory = str(root)
        shortcut.Description = "Vesper, a voice copilot"
        shortcut.save()
    except Exception as exc:
        return False, f"could not create the shortcut: {exc}"

    return True, str(target)


def uninstall() -> tuple[bool, str]:
    """Remove the login shortcut. Succeeds if it was already gone."""
    target = shortcut_path()
    if not target.exists():
        return True, "was not installed"
    try:
        target.unlink()
    except OSError as exc:
        return False, f"could not remove it: {exc}"
    return True, str(target)


def describe() -> str:
    """One line for the self check."""
    return f"starts at login  ({shortcut_path()})" if is_installed() else "not installed"
