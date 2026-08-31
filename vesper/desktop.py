"""An icon on the desktop, so Vesper can be started by hand.

Autostart covers the machine starting itself. This covers the other half: you
quit Vesper to stop it listening during a call, and then you want it back, and
"open a terminal, cd to the project folder, run run.bat" is not a way to start
an assistant.

There was already a `Vesper.lnk` on the desktop here, made by hand when the
project started. It ran `run.bat` directly, so it left a console window open
for the whole session, and it wore `SndVol.exe`'s icon, the Windows volume
mixer, which is why it was not recognisable as Vesper on a desktop full of
shortcuts. Both of those are what this fixes.

The whole file is really about one line, `desktop_dir`. Writing the shortcut is
`shortcut.py`'s job and it is the same code the Startup folder uses.
"""

from __future__ import annotations

from pathlib import Path

from . import shortcut

SHORTCUT_NAME = "Vesper.lnk"
DESCRIPTION = "Vesper, a voice copilot"
# 16 and 32 for the taskbar and alt-tab if it is ever pinned there, 48 for the
# desktop's medium icons, 96 and 256 for large and extra large.
DESKTOP_SIZES = (16, 32, 48, 96, 256)


def desktop_dir() -> Path:
    """Where this machine actually draws the desktop.

    Not `~/Desktop`. With OneDrive's "back up my desktop" turned on, which is
    the default on a new Windows install and is on here, the real desktop is
    `~/OneDrive/Desktop`, and the old `~/Desktop` still exists with years of
    leftovers in it. Writing the icon there would put it somewhere the person
    who asked for it never looks, and nothing would report a failure.

    So the shell is asked, which reads the same registry value Explorer does.
    `~/Desktop` is the fallback for a machine without pywin32, where it is the
    best guess available rather than a good one.
    """
    found = shell_folder("Desktop")
    return Path(found) if found else Path.home() / "Desktop"


def shell_folder(name: str) -> str:
    """Ask the shell where one of its special folders is. "" if it cannot."""
    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
        return str(shell.SpecialFolders(name) or "")
    except Exception:
        return ""


def shortcut_path() -> Path:
    return desktop_dir() / SHORTCUT_NAME


def is_installed() -> bool:
    return shortcut_path().exists()


def draw_icon(root: Path) -> Path | None:
    """Vesper's star, at the sizes a desktop actually asks for.

    The tray set stops at 48 pixels, which is everything a tray needs and also
    what Explorer uses for medium icons. Set the desktop to large or extra
    large and Windows asks for 96 or 256, and an upscaled 48 looks exactly as
    soft as it sounds. Its own file rather than a bigger tray icon, because
    `ui.icon.ensure` only redraws what is missing, so widening the tray sizes
    would leave every existing install with the old ones forever.

    375ms and 316KB, paid once, when the icon is installed.

    None if it cannot be drawn. A shortcut with wscript's generic scroll on it
    still starts Vesper, and refusing to make one over its picture would be the
    wrong trade.
    """
    try:
        from .ui.icon import STATES, write_ico

        background, foreground = STATES["listening"]
        return write_ico(
            Path(root) / "var" / "icons" / "vesper-desktop.ico",
            background, foreground, sizes=DESKTOP_SIZES,
        )
    except Exception:
        return None


def install(root: Path) -> tuple[bool, str]:
    """Put the icon on the desktop, drawing it first."""
    root = Path(root)
    return shortcut.write(shortcut_path(), root, description=DESCRIPTION,
                          icon=draw_icon(root))


def uninstall() -> tuple[bool, str]:
    return shortcut.remove(shortcut_path())


def describe() -> str:
    """One line for the self check."""
    return f"on the desktop  ({shortcut_path()})" if is_installed() else "not installed"
