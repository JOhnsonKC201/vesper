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
    name: str = "Vasper"
    user: str = "the user"
    wake_words: tuple[str, ...] = ("vasper", "vesper", "jarvis")
    # Free text appended to the system prompt. The place to tell Vesper to be
    # funnier, terser, or to always call you by name.
    personality: str = ""


@dataclass
class BrainSettings:
    executable: str = "claude"
    model: str = "opus"
    # Where Claude's shell starts. Home gives it reach across your projects.
    cwd: str = ""
    add_dirs: tuple[str, ...] = ()
    # Write and Edit are present so Vesper *can* act once you say yes. They are
    # not usable until then: permission_mode below refuses anything missing
    # from allowed_tools.
    tools: tuple[str, ...] = (
        "Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebSearch",
    )
    # Everything Vasper may do without asking. The rule for this list was "no
    # entry may be able to change anything", and it has been widened once,
    # deliberately, rather than quietly broken.
    #
    # The rule now: nothing here may change data, reach the network, or run a
    # program of its own choosing. Launching a Start Menu entry and changing
    # which window has focus are allowed, because both are immediately visible,
    # trivially reversible, and exactly what the user meant by "open Chrome".
    # `vasper open` takes a Start Menu shortcut and passes it no arguments, so
    # it can do no more than the user double clicking the same icon. Clicking
    # and typing are NOT here, and must not be added: a click can press Send,
    # and typing can write anything into anything.
    #
    # A review of this list found six more entries that could change things,
    # each of which meant an action nobody was ever asked about:
    #   find      GNU findutils is on PATH here, so -delete and -exec run
    #   wmic      `wmic process call create` starts arbitrary processes
    #   git branch  -D deletes a branch
    #   git remote  set-url rewrites .git/config
    #   nvidia-smi  -pl, -pm and --gpu-reset change device state
    #   powershell -Command Get-*  a wildcard on an interpreter's argument
    # WebSearch came off too. It cannot touch the disk, but it sends text off
    # the machine, and "what leaves this machine" is a promise this list keeps.
    #
    # `Bash(python*)` used to be here and had to come off: `python -c` writes
    # files, deletes them and reaches the network, so allowing it quietly
    # allowed all three. Any interpreter, package manager or editor belongs on
    # the far side of the gate for the same reason. Bare `Bash` most of all:
    # with Write refused, Claude will reach for `printf > file` instead, so a
    # list containing bare Bash is not a gate at all.
    allowed_tools: tuple[str, ...] = (
        "Read",
        "Grep",
        "Glob",
        "Bash(git status:*)",
        "Bash(git log:*)",
        "Bash(git diff:*)",
        "Bash(git show:*)",
        "Bash(ls:*)",
        "Bash(dir:*)",
        "Bash(cat:*)",
        "Bash(head:*)",
        "Bash(tail:*)",
        "Bash(type:*)",
        "Bash(findstr:*)",
        "Bash(where:*)",
        "Bash(which:*)",
        "Bash(wc:*)",
        "Bash(du:*)",
        "Bash(df:*)",
        "Bash(tasklist:*)",
        "Bash(systeminfo:*)",
        # Vasper's own hands. Looking, and the two changes that are only ever
        # visible ones. See vesper/tool.py.
        "Bash(vasper windows:*)",
        "Bash(vasper apps:*)",
        "Bash(vasper screenshot:*)",
        "Bash(vasper focus:*)",
        "Bash(vasper open:*)",
        "Bash(date:*)",
        "Bash(whoami:*)",
        "Bash(hostname:*)",
        "Bash(ipconfig:*)",
    )
    # `manual` refuses anything not allowlisted and reports it, which is what
    # gives you the spoken yes-or-no. `auto` is the CLI default and lets file
    # writes under the working directory happen with no announcement at all.
    permission_mode: str = "manual"
    turn_timeout_s: float = 180.0
    # Resume the previous conversation on startup, so Vesper remembers
    # yesterday. Bounded, because a resumed session carries its whole history.
    remember_across_restarts: bool = True
    remember_max_age_hours: float = 12.0


@dataclass
class ElevenSettings:
    """Cloud speech. Off unless a key is present, and Piper is always behind it.

    Turning this on is the one place Vesper sends anything anywhere other than
    Claude. What goes is the text of the replies, which can contain the
    contents of your files. Microphone audio never does.
    """

    # Left blank in the packaged default on purpose. Set it in config.yaml,
    # which is gitignored, or in VESPER_ELEVEN_API_KEY, which wins over the
    # file so the key need not sit on disk at all.
    api_key: str = ""
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George
    # Half the credit cost of the standard models and the lowest latency.
    model_id: str = "eleven_flash_v2_5"
    # Under the free tier's 10,000 rather than on it, so the ceiling is found
    # here rather than by ElevenLabs. Over it, Piper speaks instead.
    monthly_characters: int = 9_000
    timeout_s: float = 20.0
    cache_dir: str = "var/voice-cache"
    budget: str = "var/voice-budget.json"
    choice: str = "var/voice-choice.json"
    # Synthesize the stock phrases once at startup so the fillers are instant
    # and free. About 350 characters, paid once.
    prewarm: bool = True


@dataclass
class VoiceSettings:
    engine: str = "piper"  # piper | sapi | elevenlabs | none
    model: str = "en_GB-alan-medium"
    voices_dir: str = "var/voices"
    speed: float = 1.15
    volume: float = 0.9
    sapi_voice_hint: str = ""
    # How the voice is finished: natural, jarvis or broadcast. See
    # vesper/tts/shaping.py. `natural` is Piper untouched.
    character: str = "jarvis"
    # Where the dashboard writes the voice you clicked. Here rather than under
    # `eleven` because every engine can be picked in the window now, and a
    # choice that outlives a restart should not live inside the section for the
    # one engine that is off by default. `eleven.choice` is still read when
    # this is blank, so files written before the move keep working.
    choice: str = "var/voice-choice.json"
    eleven: ElevenSettings = field(default_factory=ElevenSettings)


@dataclass
class ListeningSettings:
    device: int | str | None = None
    # small.en: base.en could not reliably hear the wake word on a
    # real voice, waking on only two of nine attempts.
    whisper_model: str = "small.en"
    # Where transcription runs. `auto` asks CTranslate2, which is the runtime
    # that actually does it, and uses the gpu when the gpu genuinely works.
    # `cpu` pins it down. `cuda` demands the gpu and still falls back rather
    # than leaving you with an assistant that cannot hear.
    whisper_device: str = "auto"
    # `auto` picks float16 on the gpu and int8 on the cpu.
    whisper_compute: str = "auto"
    wake_required: bool = True
    # How long he stays awake after you speak, with the clock reset by every
    # thing you say. 0 means the wake word is required every single time, which
    # is what this used to be, and it disagreed with WakeConfig's own default.
    follow_up_window_s: float = 25.0
    end_silence_ms: int = 700
    max_utterance_s: float = 30.0
    # True by default because most people are on laptop speakers, where
    # full duplex means it cuts itself off on almost every reply.
    half_duplex: bool = True
    barge_in_blocks: int = 3
    self_mute_ms: int = 250
    tail_mute_ms: int = 350
    # Cores transcription may use. Unset it takes every physical core,
    # and that burst is the one moment an always-on assistant is felt.
    cpu_threads: int = 4


@dataclass
class ProactiveSettings:
    enabled: bool = True
    check_interval_s: float = 120.0
    min_interval_s: float = 900.0
    quiet_hours: str = "23:30-08:00"


@dataclass
class ConsentSettings:
    """Asking before changing anything.

    Disabling this does not make Vesper act freely, it makes it unable to act:
    the CLI still refuses, there is just no longer a way to say yes.
    """

    enabled: bool = True
    # How long a spoken yes still refers to the thing that was asked. Only
    # speech addressed to Vesper can answer, so a bare "yes" works inside the
    # follow-up window and takes the wake word after it.
    window_s: float = 45.0
    # Every approval and refusal, appended as one line. Blank disables it.
    log: str = "var/actions.log"
    # Copies of files an approved action is about to change, so "undo that"
    # can put them back. Blank makes every approved change permanent.
    undo_dir: str = "var/undo"


@dataclass
class RuntimeSettings:
    """How it behaves when nobody launched it from a terminal."""

    # A notification-area icon. Not decoration: without a console there is no
    # other way to pause or stop it short of Task Manager, which skips the exit
    # path and leaves the claude child running.
    tray: bool = True
    # Where diagnostics go when there is no terminal to print them to. Blank
    # disables, which means a hidden failure is invisible.
    log: str = "var/vesper.log"
    # Your enrolled voice. Blank, or never enrolled, means every voice is
    # accepted, which is exactly how it behaved before this existed.
    voiceprint: str = "var/voiceprint.json"
    # Instructions you have given him that should outlive the session. Blank
    # disables learning entirely and he forgets at every restart, as before.
    lessons: str = "var/lessons.json"


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
    consent: ConsentSettings = field(default_factory=ConsentSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    ui: UISettings = field(default_factory=UISettings)

    def voices_path(self) -> Path:
        path = Path(self.voice.voices_dir)
        return path if path.is_absolute() else ROOT / path

    def brain_cwd(self) -> str:
        return self.brain.cwd or str(Path.home())

    def session_path(self) -> Path:
        return ROOT / "var" / "session.json"

    def audit_path(self) -> Path | None:
        if not self.consent.log:
            return None
        path = Path(self.consent.log)
        return path if path.is_absolute() else ROOT / path

    def undo_path(self) -> Path | None:
        if not self.consent.undo_dir:
            return None
        path = Path(self.consent.undo_dir)
        return path if path.is_absolute() else ROOT / path

    def log_path(self) -> Path | None:
        return self._under_root(self.runtime.log)

    def voiceprint_path(self) -> Path | None:
        return self._under_root(self.runtime.voiceprint)

    def lessons_path(self) -> Path | None:
        return self._under_root(self.runtime.lessons)

    def voice_cache_path(self) -> Path | None:
        return self._under_root(self.voice.eleven.cache_dir)

    def voice_budget_path(self) -> Path | None:
        return self._under_root(self.voice.eleven.budget)

    def voice_choice_path(self) -> Path | None:
        return self._under_root(self.voice.choice or self.voice.eleven.choice)

    def eleven_key(self) -> str:
        """The ElevenLabs key, environment first.

        The environment wins so the key never has to be written to disk. It is
        read here rather than at import so a test can set it, and so rotating
        it means restarting rather than editing code.
        """
        import os

        return (os.environ.get("VESPER_ELEVEN_API_KEY") or self.voice.eleven.api_key or "").strip()

    def _under_root(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else ROOT / path


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
        default = getattr(blank, spec.name)
        # A section inside a section, like voice.eleven. Without this the raw
        # YAML dict is assigned straight onto the field, and every attribute
        # access on it fails later at the point of use rather than here.
        if is_dataclass(default) and not isinstance(default, type):
            kwargs[spec.name] = _build(type(default), data[spec.name])
        else:
            kwargs[spec.name] = _coerce(default, data[spec.name])
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
        consent=_build(ConsentSettings, raw.get("consent")),
        runtime=_build(RuntimeSettings, raw.get("runtime")),
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
