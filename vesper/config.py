"""Configuration.

Two files by design, copied from Echo Flow's split because it works: the tracked
default lives in packaging/default/config.yaml and is never edited; the live
config sits at the project root, is gitignored, and is created from the default
on first run. That way updating the app never silently overwrites your settings,
and your settings never end up in version control.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time as clock
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_CONFIG_PATH = ROOT / "packaging" / "default" / "config.yaml"


@dataclass
class Identity:
    name: str = "Vesper"
    user: str = "the user"
    wake_words: tuple[str, ...] = ("vesper", "jarvis")
    # Free text appended to the system prompt. The place to tell Vesper to be
    # funnier, terser, or to always call you by name.
    personality: str = ""


@dataclass
class BrainSettings:
    executable: str = "claude"
    model: str = "sonnet"
    # Where Claude's shell starts. Home gives it reach across your projects.
    cwd: str = ""
    add_dirs: tuple[str, ...] = ()
    tools: tuple[str, ...] = ("Bash", "Read", "Grep", "Glob", "WebSearch")
    # Pre-approved, read-only commands. Anything not listed prompts, which the
    # protocol turns into a spoken request for permission.
    allowed_tools: tuple[str, ...] = (
        "Bash(git status*)",
        "Bash(git log*)",
        "Bash(git diff*)",
        "Bash(git branch*)",
        "Bash(python*)",
        "Bash(dir*)",
        "Bash(ls*)",
        "Bash(cat*)",
        "Bash(type*)",
        "Bash(powershell -NoProfile -Command Get-*)",
        "Bash(tasklist*)",
        "Bash(systeminfo*)",
        "Bash(where*)",
        "Bash(findstr*)",
        "Read",
        "Grep",
        "Glob",
    )
    turn_timeout_s: float = 180.0
    # Resume the previous conversation on startup, so Vesper remembers
    # yesterday. Bounded, because a resumed session carries its whole history.
    remember_across_restarts: bool = True
    remember_max_age_hours: float = 12.0


@dataclass
class VoiceSettings:
    engine: str = "piper"  # piper | sapi | none
    model: str = "en_GB-alan-medium"
    voices_dir: str = "var/voices"
    speed: float = 1.15
    volume: float = 0.9
    sapi_voice_hint: str = ""


@dataclass
class ListeningSettings:
    device: int | str | None = None
    whisper_model: str = "base.en"
    wake_required: bool = True
    follow_up_window_s: float = 25.0
    end_silence_ms: int = 700
    max_utterance_s: float = 30.0
    half_duplex: bool = False
    barge_in_blocks: int = 3
    self_mute_ms: int = 250


@dataclass
class ProactiveSettings:
    enabled: bool = True
    check_interval_s: float = 120.0
    min_interval_s: float = 900.0
    quiet_hours: str = "23:30-08:00"


@dataclass
class UISettings:
    greet_on_start: bool = True
    show_cost: bool = True


@dataclass
class Config:
    identity: Identity = field(default_factory=Identity)
    brain: BrainSettings = field(default_factory=BrainSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    listening: ListeningSettings = field(default_factory=ListeningSettings)
    proactive: ProactiveSettings = field(default_factory=ProactiveSettings)
    ui: UISettings = field(default_factory=UISettings)

    def voices_path(self) -> Path:
        path = Path(self.voice.voices_dir)
        return path if path.is_absolute() else ROOT / path

    def brain_cwd(self) -> str:
        return self.brain.cwd or str(Path.home())

    def session_path(self) -> Path:
        return ROOT / "var" / "session.json"


# --- loading ----------------------------------------------------------------


def _coerce(default: Any, value: Any) -> Any:
    """Turn YAML lists into tuples where the default is a tuple.

    `from __future__ import annotations` makes dataclass field types strings at
    runtime, so this compares against the default value rather than the
    annotation.
    """
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _build(cls, data: dict | None):
    if not isinstance(data, dict):
        return cls()
    blank = cls()
    kwargs = {}
    for spec in fields(cls):
        if spec.name not in data:
            continue
        kwargs[spec.name] = _coerce(getattr(blank, spec.name), data[spec.name])
    return cls(**kwargs)


def load(path: Path | str | None = None) -> Config:
    """Read config.yaml, seeding it from the packaged default on first run."""
    target = Path(path) if path else CONFIG_PATH

    if not target.exists() and DEFAULT_CONFIG_PATH.exists() and target == CONFIG_PATH:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_CONFIG_PATH, target)

    if not target.exists():
        return Config()

    import yaml

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if raw is None:
        return Config()
    if not isinstance(raw, dict):
        raise ValueError(f"{target} must contain a mapping, got {type(raw).__name__}")

    return Config(
        identity=_build(Identity, raw.get("identity")),
        brain=_build(BrainSettings, raw.get("brain")),
        voice=_build(VoiceSettings, raw.get("voice")),
        listening=_build(ListeningSettings, raw.get("listening")),
        proactive=_build(ProactiveSettings, raw.get("proactive")),
        ui=_build(UISettings, raw.get("ui")),
    )


# --- quiet hours ------------------------------------------------------------


def parse_quiet_hours(spec: str) -> tuple[clock, clock] | None:
    """Parse "23:30-08:00" into two times. Returns None if unset or malformed."""
    if not spec or "-" not in spec:
        return None
    start_text, _, end_text = spec.partition("-")
    try:
        start_h, start_m = (int(x) for x in start_text.strip().split(":"))
        end_h, end_m = (int(x) for x in end_text.strip().split(":"))
        return clock(start_h, start_m), clock(end_h, end_m)
    except (ValueError, TypeError):
        return None


def in_quiet_hours(spec: str, now: clock) -> bool:
    window = parse_quiet_hours(spec)
    if window is None:
        return False
    start, end = window
    if start <= end:
        return start <= now < end
    # The window crosses midnight, which is the normal case for sleep.
    return now >= start or now < end
