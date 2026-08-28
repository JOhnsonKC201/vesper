"""Remembering the conversation across restarts.

Claude Code keeps session state server side, so passing `--resume <id>` picks up
a conversation exactly where it left off. Persisting that id is the difference
between an assistant that greets you as a stranger every morning and one that
remembers what you were doing yesterday.

There is a cost to that, which is why this is bounded rather than unconditional:
a resumed session carries its whole history, so it starts more expensive and
gets slower the longer it runs. Sessions older than the configured window are
dropped rather than resumed, so the context cannot grow without limit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    updated_at: float
    turns: int = 0
    cost_usd: float = 0.0

    def age_hours(self, now: float | None = None) -> float:
        return max(0.0, ((now or time.time()) - self.updated_at) / 3600.0)


class SessionStore:
    """A single JSON file holding the last conversation's id."""

    def __init__(self, path: str | Path, *, max_age_hours: float = 12.0) -> None:
        self.path = Path(path)
        self.max_age_hours = max_age_hours

    def load(self, now: float | None = None) -> StoredSession | None:
        """The previous session, if there is one and it is recent enough."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict) or not raw.get("session_id"):
            return None

        try:
            stored = StoredSession(
                session_id=str(raw["session_id"]),
                updated_at=float(raw.get("updated_at", 0.0)),
                turns=int(raw.get("turns", 0)),
                cost_usd=float(raw.get("cost_usd", 0.0)),
            )
        except (TypeError, ValueError):
            return None

        if stored.age_hours(now) > self.max_age_hours:
            return None
        return stored

    def save(self, session_id: str, *, turns: int = 0, cost_usd: float = 0.0) -> None:
        if not session_id:
            return
        record = StoredSession(
            session_id=session_id,
            updated_at=time.time(),
            turns=turns,
            cost_usd=cost_usd,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write then replace, so a crash mid-write cannot leave a truncated
            # file that silently loses the conversation.
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            pass  # remembering is a convenience, never a reason to fail

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
