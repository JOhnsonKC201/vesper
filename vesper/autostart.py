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

Writing the shortcut itself is `shortcut.py`, shared with `desktop.py`, because
the two differ only in where the file lands and what picture it wears.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import shortcut

SHORTCUT_NAME = "Vesper.lnk"
DESCRIPTION = "Vesper, a voice copilot"


def startup_dir() -> Path:
    """Where Windows looks for things to run at login."""
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path:
    return startup_dir() / SHORTCUT_NAME


def is_installed() -> bool:
    return shortcut_path().exists()


def install(root: Path) -> tuple[bool, str]:
    """Create the login shortcut. Returns (ok, something to tell the user).

    No icon, deliberately: nothing ever looks at the Startup folder, and a
    shortcut nobody sees does not need a picture.
    """
    return shortcut.write(shortcut_path(), Path(root), description=DESCRIPTION)


def uninstall() -> tuple[bool, str]:
    """Remove the login shortcut. Succeeds if it was already gone."""
    return shortcut.remove(shortcut_path())


def describe() -> str:
    """One line for the self check."""
    return f"starts at login  ({shortcut_path()})" if is_installed() else "not installed"
