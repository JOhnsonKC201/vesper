"""Download the calendar to var/calendar.ics, for `vasper calendar` to read.

Lives outside the `vesper` package on purpose. The package's rule is that only
the cloud voice touches the network (tests/test_privacy.py), and the calendar
does not get to be a second exception: this script fetches, the package reads a
file. It is also the only thing that ever reads the private address, so the
address never reaches Claude, the log or the dashboard.

Setup, once:
  1. Google Calendar > Settings > your calendar > "Secret address in iCal
     format". Copy it.
  2. Put it on one line in var/calendar_url.txt (var/ is gitignored).
  3. python scripts/calendar_sync.py            one sync now
     python scripts/calendar_sync.py --install  then every 15 minutes
     python scripts/calendar_sync.py --uninstall

Anyone with that address can read the calendar. Reset it in the same settings
page if it ever leaks, then paste the new one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL_FILE = ROOT / "var" / "calendar_url.txt"
OUT_FILE = ROOT / "var" / "calendar.ics"
TASK_NAME = "Vesper calendar sync"
EVERY_MINUTES = 15
TIMEOUT_S = 30
# A year of a busy personal calendar is well under a megabyte. Anything past
# this is not a calendar, or not one worth holding in memory to parse.
MAX_BYTES = 8 * 1024 * 1024


class SyncError(Exception):
    pass


def read_url(path: Path = URL_FILE) -> str:
    """The first non-blank, non-comment line of the address file."""
    if not path.exists():
        raise SyncError(f"no address yet: put the private iCal address in {path}")
    # utf-8-sig: Notepad can save with a byte order mark, which would otherwise
    # sit in front of "https://" and fail the check below.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Google hands out webcal:// for subscribing; it is the same file.
            if line.lower().startswith("webcal://"):
                line = "https://" + line[len("webcal://"):]
            if not line.lower().startswith("https://"):
                raise SyncError("the address must start with https:// or webcal://")
            return line
    raise SyncError(f"{path} has no address in it")


def host_of(url: str) -> str:
    """The host alone, for messages. The path is the secret part."""
    return urllib.parse.urlsplit(url).hostname or "the calendar host"


def validate(body: bytes) -> bytes:
    if len(body) > MAX_BYTES:
        raise SyncError("the download is too large to be a calendar")
    if not body.lstrip().upper().startswith(b"BEGIN:VCALENDAR"):
        raise SyncError("the download is not an iCal calendar (check the address)")
    return body


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vesper-calendar-sync"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise SyncError(f"{host_of(url)} answered {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SyncError(f"could not reach {host_of(url)}: {reason}") from None


def write_atomically(body: bytes, out: Path = OUT_FILE) -> None:
    """Never leave a half-written file for `vasper calendar` to trip over."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_bytes(body)
    temp.replace(out)


def sync(url_file: Path = URL_FILE, out: Path = OUT_FILE, fetcher=fetch) -> int:
    url = read_url(url_file)
    body = validate(fetcher(url))
    write_atomically(body, out)
    return len(body)


def _pythonw() -> str:
    """The venv's windowless interpreter, so the task never flashes a console."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate if candidate.exists() else sys.executable)


def install_argv() -> list[str]:
    action = f'"{_pythonw()}" "{Path(__file__).resolve()}"'
    return [
        "schtasks", "/Create", "/F", "/TN", TASK_NAME,
        "/SC", "MINUTE", "/MO", str(EVERY_MINUTES), "/TR", action,
    ]


def uninstall_argv() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", action="store_true", help=f"sync every {EVERY_MINUTES} minutes")
    group.add_argument("--uninstall", action="store_true", help="stop the scheduled sync")
    args = parser.parse_args(argv)

    if args.install or args.uninstall:
        command = install_argv() if args.install else uninstall_argv()
        done = subprocess.run(command, capture_output=True, text=True)
        print((done.stdout or done.stderr).strip())
        return done.returncode

    try:
        size = sync()
    except SyncError as exc:
        print(f"calendar sync failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Most often the file is open in `vasper calendar` at that moment. The
        # next run in 15 minutes tries again; this only keeps it from being a
        # traceback in a window nobody sees.
        print(f"calendar sync failed: could not write the file ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"calendar synced: {size} bytes to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
