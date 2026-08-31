"""Taking back the last thing you said yes to.

Consent given out loud is easy to give. "Yes" takes a quarter of a second, and
the thing it approves can be a file you had spent an afternoon on. The gate
stops Vesper acting without agreement; it does nothing about agreeing to the
wrong thing, which is the more likely mistake by far.

So before an approved action runs, whatever it is about to touch is copied
aside. "Undo that" puts it back.

The honest scope, stated plainly because a half-working undo is worse than none:
this covers the file tools, and shell commands whose targets can be read off the
command line with certainty, which in practice means `rm` and `del`. A general
shell command can do anything at all, and pretending otherwise would be a
promise this cannot keep. When there is no snapshot, undo says so rather than
claiming success.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Copying aside a very large file to make a voice command reversible is a bad
# trade. Above this, no snapshot is taken and undo says why.
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024

# Snapshots kept before the oldest is deleted. Undo only ever reaches the most
# recent one; the rest are there so a mistake noticed late is still recoverable
# by hand from var/undo/.
KEEP = 20

# Shell verbs whose file arguments are unambiguous. Deliberately short: this is
# the list of commands where getting it wrong is not possible, not the list of
# commands that touch files.
_REMOVERS = {"rm", "del", "erase", "unlink"}
# Unix style, and the `/f /q` form that `del` actually takes on Windows. The
# slash form is deliberately one letter only: `/tmp/x` is a path, `/f` is not.
_FLAG = re.compile(r"^(-{1,2}[A-Za-z][A-Za-z-]*|/[A-Za-z])$")


@dataclass(frozen=True)
class Saved:
    """One file as it was, or the fact that it did not exist."""

    original: str
    copy: str = ""          # empty means the file did not exist
    existed: bool = True

    def to_dict(self) -> dict:
        return {"original": self.original, "copy": self.copy, "existed": self.existed}

    @staticmethod
    def from_dict(data: dict) -> "Saved":
        return Saved(
            original=str(data.get("original") or ""),
            copy=str(data.get("copy") or ""),
            existed=bool(data.get("existed", True)),
        )


@dataclass
class Snapshot:
    """Everything kept aside for one approved action."""

    action: str
    at: float = field(default_factory=time.time)
    files: list[Saved] = field(default_factory=list)
    # Why nothing was kept, when nothing was kept. Spoken back on an undo, so
    # the answer to "undo that" is never a bare no.
    reason: str = ""

    @property
    def reversible(self) -> bool:
        return bool(self.files)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "at": self.at,
            "reason": self.reason,
            "files": [f.to_dict() for f in self.files],
        }

    @staticmethod
    def from_dict(data: dict) -> "Snapshot":
        return Snapshot(
            action=str(data.get("action") or ""),
            at=float(data.get("at") or 0.0),
            reason=str(data.get("reason") or ""),
            files=[Saved.from_dict(f) for f in data.get("files") or []],
        )


# --- working out what an action will touch ----------------------------------


def targets(tool: str, tool_input: dict) -> tuple[list[str], str]:
    """Files an approved action is about to change.

    Returns the paths and, when there are none, why not. The second half is the
    point: "I can't undo a shell command" is a useful answer, and silence is not.
    """
    if not isinstance(tool_input, dict):
        return [], "there was nothing to keep a copy of"

    if tool in {"Write", "Edit", "MultiEdit"}:
        path = tool_input.get("file_path")
        return ([str(path)], "") if path else ([], "no file was named")
    if tool == "NotebookEdit":
        path = tool_input.get("notebook_path") or tool_input.get("file_path")
        return ([str(path)], "") if path else ([], "no file was named")

    if tool == "Bash":
        command = str(tool_input.get("command") or "")
        found = _removal_targets(command)
        if found:
            return found, ""
        return [], "that was a shell command, and I can only undo file changes"

    return [], f"{tool} is not something I can undo"


def _removal_targets(command: str) -> list[str]:
    """Paths a plain removal command will delete, or nothing if it is not plain.

    Anything with a pipe, a redirect, a chain, a variable or a glob is treated
    as not plain. A wrong answer here means undo silently misses a file, so the
    bar is certainty rather than coverage.

    Parsed with posix=False, which matters more than it looks. In posix mode
    shlex treats a backslash as an escape, so `del C:\\Users\\johns\\notes.txt`
    came apart as `C:Usersjohnsnotes.txt`. That path does not exist, so the file
    was recorded as one that had never existed, the real file was deleted, and
    undo then reported "Put notes.txt back" while the original was gone for
    good. Silent data loss announced as success, on the ordinary Windows way of
    writing a delete.
    """
    if not command or any(ch in command for ch in "|&;<>$`*?"):
        return []
    try:
        import shlex

        tokens = shlex.split(command, posix=False)
    except ValueError:
        return []
    tokens = [_unquote(t) for t in tokens]
    if not tokens or tokens[0].replace("\\", "/").split("/")[-1].lower() not in _REMOVERS:
        return []

    targets: list[str] = []
    end_of_flags = False
    for token in tokens[1:]:
        if not end_of_flags and token == "--":
            # Everything after this is a filename, even if it starts with a
            # dash. Dropping it as a flag loses the very file it protects.
            end_of_flags = True
            continue
        if not end_of_flags and _FLAG.match(token):
            continue
        targets.append(token)
    return targets


def _unquote(token: str) -> str:
    """Strip the quotes posix=False leaves in place."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


# --- keeping and restoring --------------------------------------------------


class UndoStore:
    """Snapshots on disk, newest last."""

    def __init__(self, directory: Path | str, *, keep: int = KEEP) -> None:
        self.directory = Path(directory)
        self.keep = keep

    @property
    def ledger(self) -> Path:
        return self.directory / "ledger.json"

    def _load(self) -> list[Snapshot]:
        try:
            raw = json.loads(self.ledger.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
            # Inside the guard, not after it. A ledger that is valid json but
            # the wrong shape, say a `files` value that is a string, raised an
            # uncaught AttributeError here and took the whole assistant down on
            # the next "undo that". Nothing up the call chain catches it.
            return [Snapshot.from_dict(item) for item in raw if isinstance(item, dict)]
        except (OSError, ValueError, TypeError, AttributeError):
            return []

    def _save(self, snapshots: list[Snapshot]) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.ledger.write_text(
                json.dumps([s.to_dict() for s in snapshots], indent=2),
                encoding="utf-8",
            )
        except (OSError, ValueError):
            pass

    def latest(self) -> Snapshot | None:
        snapshots = self._load()
        return snapshots[-1] if snapshots else None

    def _free_name(self, basename: str) -> Path:
        """A copy path nothing else is using.

        The name used to be the millisecond plus the basename, which collides
        whenever one action touches two files with the same name in different
        folders. Both records then pointed at one copy: the first file came back
        holding the second one's contents, the second was unrecoverable, and
        undo announced success for both.
        """
        stamp = int(time.time() * 1000)
        for attempt in range(10000):
            candidate = self.directory / f"{stamp}-{attempt}-{basename}"
            if not candidate.exists():
                return candidate
        raise OSError("no free snapshot name")

    def keep_copy(self, action: str, tool: str, tool_input: dict) -> Snapshot:
        """Copy aside whatever this action is about to change.

        Never raises. A failed snapshot must not block an action the user has
        already approved out loud; it only means undo will say it cannot help.
        """
        paths, reason = targets(tool, tool_input)
        snapshot = Snapshot(action=action, reason=reason)

        for raw in paths:
            try:
                source = Path(raw)
                if not source.exists():
                    # Worth recording: undoing a file that did not exist before
                    # means deleting the one that does now.
                    snapshot.files.append(Saved(str(source), existed=False))
                    continue
                if source.is_dir():
                    snapshot.reason = "that is a folder, and I only copy files aside"
                    continue
                if source.stat().st_size > MAX_SNAPSHOT_BYTES:
                    snapshot.reason = "the file was too big to keep a copy of"
                    continue
                self.directory.mkdir(parents=True, exist_ok=True)
                copy = self._free_name(source.name)
                shutil.copy2(source, copy)
                snapshot.files.append(Saved(str(source), str(copy)))
            except (OSError, ValueError) as exc:
                snapshot.reason = f"I could not keep a copy: {exc}"

        snapshots = self._load()
        snapshots.append(snapshot)
        self._prune(snapshots)
        self._save(snapshots)
        return snapshot

    def _prune(self, snapshots: list[Snapshot]) -> None:
        while len(snapshots) > self.keep:
            oldest = snapshots.pop(0)
            for saved in oldest.files:
                if not saved.copy:
                    continue
                try:
                    Path(saved.copy).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass

    def undo_latest(self) -> tuple[bool, str]:
        """Put the most recent approved change back.

        Returns (did_something, sentence to say). The sentence is the whole
        point: every failure mode here has a specific and useful explanation,
        and "no" on its own has none.
        """
        snapshots = self._load()
        if not snapshots:
            return False, "There's nothing to undo."

        snapshot = snapshots[-1]
        if not snapshot.reversible:
            reason = snapshot.reason or "I did not keep a copy of that one"
            return False, f"I can't undo that, {reason}."

        restored, failed = 0, ""
        for saved in snapshot.files:
            try:
                target = Path(saved.original)
                if saved.existed and saved.copy:
                    if not Path(saved.copy).exists():
                        # The copy is gone: quarantined, pruned, deleted by
                        # hand. Claiming this one back would be a lie.
                        failed = "the copy I kept is missing"
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved.copy, target)
                    Path(saved.copy).unlink(missing_ok=True)
                    restored += 1
                elif not saved.existed:
                    # It did not exist before the action, so putting things back
                    # means removing what the action created.
                    if target.exists() and target.is_file():
                        target.unlink()
                    restored += 1
            except (OSError, ValueError) as exc:
                failed = str(exc)

        if not restored:
            # The entry stays. Popping it on failure burned the only chance to
            # try again, so a locked file or a passing antivirus scan cost you
            # the recovery permanently.
            return False, f"I tried to undo that and couldn't, {failed}."

        snapshots.pop()
        self._save(snapshots)
        name = Path(snapshot.files[0].original).name
        if restored == 1:
            return True, f"Put {name} back."
        return True, f"Put {restored} files back."
