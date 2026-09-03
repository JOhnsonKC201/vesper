"""Every action, on one line, forever.

A brief constraint, and independently the only way Experiment 1 produces a
result anyone can check. Success rate alone cannot tell you why (a) beat (b);
the trace can, because it holds the observation size, the element that was
chosen, and how long each stage took.

JSON Lines rather than one JSON document, because a run that crashes halfway
still leaves a readable file, and because appending never has to rewrite what
is already on disk. A crashed run is exactly the run you most want to read.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trace:
    """An append-only JSONL sink for one experiment run."""

    path: Path
    run_id: str = ""
    # Kept open for the life of the run. Reopening per line is safer against a
    # hard kill but costs a syscall per action, and the flush below already
    # covers the case that actually happens.
    _handle: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.run_id:
            self.run_id = time.strftime("%Y%m%dT%H%M%S")

    def open(self) -> "Trace":
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __enter__(self) -> "Trace":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, kind: str, **fields) -> dict:
        """Append one record. Returns it, which is what makes this testable
        without reading the file back."""
        record = {
            "ts": time.time(),
            "run": self.run_id,
            "kind": kind,
            **fields,
        }
        if self._handle is None:
            self.open()
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._handle.write(line + "\n")
        # Flushed every line on purpose. An unflushed buffer is how a run that
        # hangs leaves behind a trace that stops well before the interesting
        # part, which is the part you opened the file to read.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return record

    # The vocabulary. Free functions would do, but naming the record kinds here
    # keeps the reader of a trace file from having to guess the schema.

    def task_start(self, task_id: str, mode: str, goal: str) -> dict:
        return self.write("task_start", task=task_id, mode=mode, goal=goal)

    def observation(self, task_id: str, step: int, obs) -> dict:
        return self.write(
            "observation",
            task=task_id,
            step=step,
            mode=obs.mode,
            elements=len(obs.elements),
            latency_s=round(obs.latency_s, 4),
            window=obs.window_title,
            sources=sorted({e.source for e in obs.elements}),
        )

    def action(self, task_id: str, step: int, verb: str, args: dict,
               element=None, latency_s: float = 0.0, ok: bool = True,
               detail: str = "") -> dict:
        return self.write(
            "action",
            task=task_id,
            step=step,
            verb=verb,
            args=args,
            element=element.to_json() if element is not None else None,
            latency_s=round(latency_s, 4),
            ok=ok,
            detail=detail,
        )

    def blocked(self, task_id: str, step: int, verb: str, args: dict,
                reason: str) -> dict:
        """A destructive action the guard stopped. Recorded as loudly as a
        performed one: a run whose success came from being allowed to delete
        something is a different result than one that was not."""
        return self.write("blocked", task=task_id, step=step, verb=verb,
                          args=args, reason=reason)

    def task_end(self, task_id: str, mode: str, ok: bool, steps: int,
                 wall_s: float, detail: str = "") -> dict:
        return self.write("task_end", task=task_id, mode=mode, ok=ok,
                          steps=steps, wall_s=round(wall_s, 3), detail=detail)


def read(path: Path) -> list[dict]:
    """Load a trace back. Tolerates a truncated last line, which is what a
    killed run leaves behind."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Only ever the final line, and only after a hard kill. Dropping it
            # silently would hide that the run did not finish.
            records.append({"kind": "truncated", "raw": line})
    return records
