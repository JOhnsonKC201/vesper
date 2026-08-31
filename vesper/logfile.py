"""Somewhere for things to go when nobody is watching.

Vesper was built to be run from a terminal, so everything it has to say goes to
a rich console: warnings, errors, the brain's stderr, tracebacks. Start it from
your login instead and there is no console, which means an unplugged microphone,
a failed model load or a crashed CLI child all happen in complete silence. You
would find out by noticing it had stopped answering.

So the same callbacks that feed the terminal also append here. Deliberately not
the `logging` module's own config machinery: this process shares an interpreter
with torch, onnxruntime and ctranslate2, all of which have opinions about the
root logger, and a stray `basicConfig` would either lose our lines or flood the
file with theirs.

Rotation is size based and small. Nobody reads a hundred megabytes of this, and
an assistant that fills a disk while trying to be helpful has failed twice.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
KEEP_ROTATIONS = 3


class LogFile:
    """Append-only text log with size-based rotation. Never raises.

    Losing a log line must never take down a conversation, so every failure
    here is swallowed the same way `audit.record` swallows its own.
    """

    def __init__(self, path: Path | str | None, *, max_bytes: int = MAX_BYTES) -> None:
        self.path = Path(path) if path else None
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, level: str, message: str) -> None:
        if self.path is None:
            return
        line = "{when}  {level:<5} {message}\n".format(
            when=datetime.now().isoformat(timespec="seconds"),
            level=level.upper()[:5],
            message=" ".join(str(message).split()),
        )
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, ValueError):
            # ValueError covers a path the OS will not even parse. Both mean
            # the same thing: no log, and the assistant carries on regardless.
            pass

    def _rotate_if_needed(self) -> None:
        if self.path is None or not self.path.exists():
            return
        if self.path.stat().st_size < self.max_bytes:
            return
        # vesper.log -> vesper.log.1 -> vesper.log.2, oldest dropped.
        for index in range(KEEP_ROTATIONS - 1, 0, -1):
            older = self.path.with_suffix(self.path.suffix + f".{index}")
            newer = self.path.with_suffix(self.path.suffix + f".{index + 1}")
            if older.exists():
                older.replace(newer)
        self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))

    # --- convenience --------------------------------------------------------

    def info(self, message: str) -> None:
        self.write("info", message)

    def warn(self, message: str) -> None:
        self.write("warn", message)

    def error(self, message: str) -> None:
        self.write("error", message)


def attach(ui, log: LogFile):
    """Tee a TerminalUI's diagnostics into the log without changing the UI.

    Wrapping rather than editing `TerminalUI`: the terminal is the primary
    surface when there is one, and this must not change what it prints. It also
    keeps the log working for any other UI object, which is what the tests use.
    """
    if not log.enabled:
        return ui

    for level in ("info", "warn", "error"):
        original = getattr(ui, level, None)
        if original is None:
            continue

        def tee(message, _original=original, _level=level):
            log.write(_level, message)
            return _original(message)

        setattr(ui, level, tee)

    # What it heard, and whether it thought you were talking to it. This is the
    # only thing that answers "I said the wake word and nothing happened":
    # without it there is no way to tell a microphone that heard nothing from a
    # transcription that came out wrong from a wake gate that said no.
    heard = getattr(ui, "heard", None)
    if heard is not None:

        def tee_heard(text, addressed, _original=heard):
            log.write("heard", f"{'->' if addressed else '  '} {text}")
            return _original(text, addressed)

        ui.heard = tee_heard

    # And what it threw away, which is the other half of the same question.
    discarded = getattr(ui, "discarded", None)
    if discarded is not None:

        def tee_discarded(reason, _original=discarded):
            log.write("drop", str(reason))
            return _original(reason)

        ui.discarded = tee_discarded

    return ui
