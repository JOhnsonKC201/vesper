"""Terminal interface.

Deliberately a scrolling log rather than a full-screen layout. Vesper is a thing
you talk to while doing something else; a live-updating dashboard would demand
attention it should not be asking for, and it fights badly with the several
background threads writing to it.

The screen channel is the reason this exists at all. Vesper says one sentence
out loud and puts the file path, the command output, the list of five things
here, where it can be read and copied.
"""

from __future__ import annotations

import threading

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_LABEL_WIDTH = 8


class TerminalUI:
    """Prints the conversation as it happens."""

    def __init__(self, *, show_cost: bool = True, console: Console | None = None) -> None:
        self.console = console or Console(highlight=False)
        self.show_cost = show_cost
        self._lock = threading.Lock()
        self.session_cost = 0.0
        self.turns = 0

    # --- helpers ------------------------------------------------------------

    def _line(self, label: str, body, style: str = "", label_style: str = "dim") -> None:
        with self._lock:
            text = Text()
            text.append(f"{label:<{_LABEL_WIDTH}}", style=label_style)
            if isinstance(body, Text):
                text.append_text(body)
            else:
                text.append(str(body), style=style)
            self.console.print(text)

    # --- lifecycle ----------------------------------------------------------

    def banner(self, *, brain: str, voice: str, ears: str, wake: str, cwd: str) -> None:
        body = Text()
        body.append("brain   ", style="dim")
        body.append(f"{brain}\n", style="bold cyan")
        body.append("voice   ", style="dim")
        body.append(f"{voice}\n", style="bold magenta")
        body.append("ears    ", style="dim")
        body.append(f"{ears}\n", style="bold green")
        body.append("wake    ", style="dim")
        body.append(f'say "{wake}", then just keep talking\n', style="white")
        body.append("shell   ", style="dim")
        body.append(cwd, style="white")
        self.console.print()
        self.console.print(Panel(body, title="Vesper", border_style="cyan", expand=False))
        self.console.print()

    def ready(self, message: str = "listening") -> None:
        self._line("", Text(message, style="dim italic"))

    # --- conversation -------------------------------------------------------

    def heard(self, text: str, addressed: bool) -> None:
        if addressed:
            self._line("you", text, style="bold white", label_style="bold blue")
        else:
            self._line("", Text(f"(not addressed) {text}", style="dim"))

    def thinking(self, what: str) -> None:
        pass  # too noisy to print; the tool and answer lines carry the signal

    def tool(self, name: str, detail: str) -> None:
        body = Text()
        body.append(name, style="yellow")
        if detail:
            body.append("  ")
            body.append(detail, style="dim")
        self._line("run", body)

    def screen(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self.console.print()
            self.console.print(
                Panel(Text(text), border_style="dim", expand=False, padding=(0, 1))
            )

    def permission(self, tool: str, detail: str) -> None:
        body = Text()
        body.append("needs your ok: ", style="bold yellow")
        body.append(tool, style="yellow")
        if detail:
            body.append(f"  {detail}", style="dim")
        self._line("hold", body)

    def decision(self, decision: str, action: str) -> None:
        """What was actually decided about a request, and what it now permits.

        Printed in full rather than in the spoken shorthand: "run git commit"
        is the right thing to hear and the wrong thing to have as the only
        record of what ran.
        """
        style = {
            "approved": "bold green",
            "declined": "bold red",
            "undone": "bold yellow",
        }.get(decision, "dim")
        label = {"approved": "ok", "declined": "no", "undone": "undo"}.get(decision, "")
        body = Text()
        body.append(decision, style=style)
        body.append(f"  {action}", style="dim")
        self._line(label, body)

    def spoke(self, text: str) -> None:
        self._line("vesper", text, style="bold cyan", label_style="bold cyan")

    def answered(
        self,
        turn,
        total_s: float,
        first_speech_s: float | None,
        *,
        stt_s: float = 0.0,
        model: str = "",
    ) -> None:
        # `stt_s` and `model` are for the log's turn line, teed in logfile.py.
        # The console line is deliberately unchanged: it is read at a glance.
        del stt_s, model
        self.turns += 1
        self.session_cost += getattr(turn, "cost_usd", 0.0)
        if not self.show_cost:
            return
        parts = []
        if first_speech_s is not None:
            parts.append(f"{first_speech_s:.1f}s to first word")
        parts.append(f"{total_s:.1f}s total")
        if getattr(turn, "turns", 0) > 1:
            parts.append(f"{turn.turns} steps")
        parts.append(f"${turn.cost_usd:.3f}")
        parts.append(f"session ${self.session_cost:.2f}")
        self._line("", Text(" · ".join(parts), style="dim"))

    def interrupted(self, dropped: int) -> None:
        note = "cut off" if not dropped else f"cut off, dropped {dropped} queued"
        self._line("", Text(note, style="dim italic yellow"))

    def discarded(self, reason: str) -> None:
        # "heard myself" is worth showing; the rest is routine noise gating.
        if reason in ("heard myself",):
            self._line("", Text(reason, style="dim italic"))

    def proactive(self, text: str) -> None:
        self._line("vesper", Text(text, style="bold magenta"), label_style="bold magenta")

    # --- diagnostics --------------------------------------------------------

    def error(self, message: str) -> None:
        self._line("error", message, style="bold red", label_style="bold red")

    def warn(self, message: str) -> None:
        self._line("warn", message, style="yellow", label_style="yellow")

    def info(self, message: str) -> None:
        self._line("", Text(message, style="dim"))

    def farewell(self) -> None:
        self.console.print()
        self._line(
            "",
            Text(
                f"{self.turns} turns, ${self.session_cost:.2f} this session.",
                style="dim",
            ),
        )
        self.console.print()
