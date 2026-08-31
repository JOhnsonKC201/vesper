"""Which voice you picked, remembered across restarts.

This exists because `config.yaml` is not a settings file that a program should
write. It is 8.5KB of which the large majority is comments explaining why each
value is what it is, and `yaml.safe_dump` would erase every one of them on the
first save. There is no comment-preserving YAML writer in this project's
dependencies and adding one to store a single string would be absurd.

So the split is: `config.yaml` is what you set by hand, and this is what the
dashboard sets by clicking. This wins when both name a voice, because a click
is more recent and more deliberate than a file you edited last month.

It follows the pattern already used by `var/session.json` and
`var/voiceprint.json`: small, machine-local, gitignored, temp-file-and-rename
so a crash mid-write cannot corrupt it, and unreadable means "no opinion"
rather than an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceChoice:
    """A voice picked in the UI. Empty means nothing was ever picked."""

    voice_id: str = ""
    name: str = ""
    # Which backend the id belongs to: piper, sapi or elevenlabs. Files written
    # before the local picker existed have no engine, and an ElevenLabs id is
    # the only thing they can contain, so blank reads as elevenlabs.
    engine: str = ""

    @property
    def chosen(self) -> bool:
        return bool(self.voice_id)

    @property
    def local(self) -> bool:
        return self.engine in ("piper", "sapi")


def load(path: Path | str | None) -> VoiceChoice:
    """Read the saved choice. Any problem at all reads as no choice."""
    if not path:
        return VoiceChoice()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return VoiceChoice()
    if not isinstance(raw, dict):
        return VoiceChoice()
    return VoiceChoice(
        voice_id=str(raw.get("voice_id") or ""),
        name=str(raw.get("name") or ""),
        engine=str(raw.get("engine") or ""),
    )


def save(path: Path | str | None, voice_id: str, name: str = "",
         engine: str = "") -> bool:
    """Persist a choice. Returns whether it landed."""
    if not path or not voice_id:
        return False
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"voice_id": voice_id, "name": name, "engine": engine},
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
        return True
    except (OSError, ValueError):
        return False


def clear(path: Path | str | None) -> None:
    """Forget the choice, so config.yaml takes over again."""
    if not path:
        return
    try:
        Path(path).unlink()
    except (OSError, ValueError):
        pass
