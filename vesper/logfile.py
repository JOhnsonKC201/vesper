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
        # Every line used to cost a mkdir, an exists, a stat, and then the open
        # and the write. Four of those five answer a question we already know
        # the answer to. The directory is made once, and the size is carried
        # forward, so the file is only measured when the running total says it
        # is worth measuring. The handle is still opened and closed per line:
        # holding it open would be cheaper again, and on Windows it would also
        # stop `_rotate_if_needed` renaming the file out from under itself.
        self._made_parent = False
        self._written = -1

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
                if not self._made_parent:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._made_parent = True
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                self._written += len(line.encode("utf-8"))
        except (OSError, ValueError):
            # ValueError covers a path the OS will not even parse. Both mean
            # the same thing: no log, and the assistant carries on regardless.
            # The byte count is left alone, so the next write measures the file
            # rather than trusting a total that may now be wrong.
            self._made_parent = False
            self._written = -1

    def _rotate_if_needed(self) -> None:
        if self.path is None:
            return
        if self._written < 0:
            # First write of this process, or straight after a failure. This is
            # the one place the file is actually measured.
            self._written = self.path.stat().st_size if self.path.exists() else 0
        if self._written < self.max_bytes:
            return
        if not self.path.exists():
            self._written = 0
            return
        # vesper.log -> vesper.log.1 -> vesper.log.2, oldest dropped.
        for index in range(KEEP_ROTATIONS - 1, 0, -1):
            older = self.path.with_suffix(self.path.suffix + f".{index}")
            newer = self.path.with_suffix(self.path.suffix + f".{index + 1}")
            if older.exists():
                older.replace(newer)
        self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        self._written = 0

    # --- convenience --------------------------------------------------------

    def info(self, message: str) -> None:
        self.write("info", message)

    def warn(self, message: str) -> None:
        self.write("warn", message)

    def error(self, message: str) -> None:
        self.write("error", message)


def attach(ui, log: LogFile, *, transcripts: bool = True):
    """Tee a TerminalUI's diagnostics into the log without changing the UI.

    Wrapping rather than editing `TerminalUI`: the terminal is the primary
    surface when there is one, and this must not change what it prints. It also
    keeps the log working for any other UI object, which is what the tests use.

    `transcripts=False` keeps the words out. Everything the microphone hears is
    written down, addressed to Vesper or not, and one week of var/vesper.log
    holds 975 transcriptions of which 238 were addressed to him. The rest is a
    record of a room: other people, a television, half of a phone call.
    `sensors/window.py` already redacts sensitive window titles, and there was
    no equivalent for what was said out loud.

    On by default, because it is the only thing that answers "I said the wake
    word and nothing happened", which is a real question with no other source.
    Off is for a shared room.
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
            if transcripts:
                log.write("heard", f"{'->' if addressed else '  '} {text}")
            else:
                # That something was heard, and whether it was for him, without
                # what it was. Enough to answer "the wake word did nothing".
                log.write("heard", f"{'->' if addressed else '  '} [{len(text)} chars]")
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
