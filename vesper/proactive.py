"""Speaking without being spoken to.

This is the difference between a tool you invoke and something that is present.
It is also the fastest way to build something insufferable, so nearly all of this
file is restraint:

  - It never decides on its own what is worth saying. It gathers what changed and
    asks Claude "is this worth interrupting for?", which answers SILENT almost
    every time. A hand-written rule set would either miss the interesting cases
    or fire constantly.
  - One remark per fifteen minutes, maximum, regardless of how much happens.
  - Silent during quiet hours, while you are already talking to it, while you are
    away from the keyboard, and for anything it has already mentioned.
  - One spoken word turns it off.

The default posture is silence. Every gate below defaults to not speaking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime

from .brain.persona import PROACTIVE_PROMPT, clean_for_speech
from .brain.protocol import BrainError, TextDelta, TurnComplete
from .config import in_quiet_hours
from .sensors import snapshot as sensors

SILENT = "SILENT"


@dataclass
class ProactiveConfig:
    enabled: bool = True
    check_interval_s: float = 120.0
    min_interval_s: float = 900.0
    quiet_hours: str = "23:30-08:00"
    # Do not interrupt someone who is not there.
    max_idle_s: float = 900.0
    # Remember recent remarks so it cannot say the same thing twice.
    memory_size: int = 8


class ProactiveLoop:
    """Watches the machine and occasionally has something to say."""

    def __init__(
        self,
        *,
        brain,
        speak,
        config: ProactiveConfig | None = None,
        ui=None,
        now=time.time,
    ) -> None:
        self.brain = brain
        self._speak = speak
        self.config = config or ProactiveConfig()
        self.ui = ui
        self._now = now

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._muted = not self.config.enabled
        self._last_spoke_at = 0.0
        self._previous: sensors.Snapshot | None = None
        self._recent: list[str] = []
        self.remarks = 0
        self.checks = 0

    # --- control ------------------------------------------------------------

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False
        # Do not let unmuting immediately dump a backlog.
        self._last_spoke_at = self._now()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._previous = sensors.take(include_processes=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="proactive")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    # --- loop ---------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self.config.check_interval_s):
            try:
                self.tick()
            except Exception as exc:  # never let ambient chatter kill the app
                if self.ui:
                    self.ui.warn(f"proactive check failed: {exc}")

    def tick(self) -> str:
        """One check. Returns what was said, or "" for silence.

        Public and synchronous so the whole policy is testable without threads.
        """
        self.checks += 1
        current = sensors.take(include_processes=True)
        previous, self._previous = self._previous, current

        blocked = self._blocked(current)
        if blocked:
            return ""

        signals = sensors.changes_since(previous, current)
        # Window switches alone are context, not news. Without this gate a
        # normal working hour would spend a Claude turn every two minutes.
        if not any(getattr(s, "notable", False) for s in signals):
            return ""

        remark = self._consult([str(s) for s in signals])
        if not remark:
            return ""

        self._last_spoke_at = self._now()
        self._remember(remark)
        self.remarks += 1
        if self.ui:
            self.ui.proactive(remark)
        self._speak(remark)
        return remark

    # --- gates --------------------------------------------------------------

    def _blocked(self, current: sensors.Snapshot) -> str:
        if self._muted:
            return "muted"
        if in_quiet_hours(self.config.quiet_hours, datetime.now().time()):
            return "quiet hours"
        if self._now() - self._last_spoke_at < self.config.min_interval_s:
            return "too soon"
        if current.idle_s > self.config.max_idle_s:
            return "away"
        # Never talk over a conversation already in progress.
        if getattr(self.brain, "busy", False):
            return "mid conversation"
        return ""

    def _remember(self, remark: str) -> None:
        self._recent.append(remark)
        del self._recent[: -self.config.memory_size]

    # --- asking claude ------------------------------------------------------

    def _consult(self, signals: list[str]) -> str:
        """Ask whether any of this is worth saying. Usually the answer is no."""
        lines = ["what changed:"] + [f"- {s}" for s in signals]
        if self._recent:
            lines.append("")
            lines.append("you already said these recently, do not repeat them:")
            lines += [f"- {r}" for r in self._recent]

        prompt = PROACTIVE_PROMPT.format(signals="\n".join(lines))

        collected: list[str] = []
        final = ""
        for event in self.brain.ask(prompt):
            if isinstance(event, TextDelta):
                collected.append(event.text)
            elif isinstance(event, TurnComplete):
                final = event.text or "".join(collected)
            elif isinstance(event, BrainError):
                return ""

        remark = clean_for_speech(final or "".join(collected)).strip()
        if not remark:
            return ""
        # Accept only a clear non-SILENT answer. Anything that merely contains
        # the word, or that rambles, is treated as a refusal to interrupt.
        if SILENT in remark.upper():
            return ""
        if len(remark) > 220:
            return ""
        return remark
