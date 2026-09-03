"""How to pitch this turn.

Vasper has had sensors since the beginning and has never once let them change
how it talks. It knows the clock, how long it has been running, how long you
have been away and what you are looking at, and it says the same thing at one
in the morning after four hours as it does at ten with a coffee.

This closes that gap, and the shape of it matters more than the contents.

**Stated emotion repels.** An assistant that says "that sounds frustrating"
is worse than one that says nothing, because the sympathy is obviously
synthetic and the words are spent on nothing. What reads as human is not
feeling described, it is *having noticed*: "third failure on the same thing,
want the diff" lands because it proves attention. So nothing here ever tells
the model how the user feels or how to feel back. Every line is an instruction
about delivery, and the delivery is what carries it.

**Silence is the default.** Like `proactive.py`, most of this file is restraint.
On an ordinary turn it produces nothing at all, and that is the correct answer:
a register note on every turn is a tic, and the model would start performing
attentiveness instead of paying it. The lines below fire on real signals or not
at all, and they are short because they ride on every turn that earns one.

The cost ceiling is the same as the rest of the context block: this is spoken
into the same eighty tokens that `sensors.context_block` is held to, and a test
holds it there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# A gap shorter than this means the exchange is still going: you asked, it
# answered, you came straight back. Longer and it is a fresh approach.
MID_FLOW_GAP_S = 90.0

# When "it is late" starts. Deliberately not midnight: the hour that changes how
# someone wants to be spoken to is the one where they should have stopped.
LATE_HOUR = 23
EARLY_HOUR = 5

# How long a sitting has to run before length starts costing more than it buys.
LONG_SESSION_S = 3 * 3600

# Long enough away that picking up mid-thought would be confusing.
AWAY_S = 600.0

# Two is a coincidence. Three is a pattern worth changing tack over.
REPEATED_FAILURES = 2


@dataclass
class Register:
    """Session state, and the one line it is worth saying about it.

    Fed once per turn from the conversation loop. Holds only scalars, because
    the dashboard reads the conversation's status across a thread boundary and
    nothing here should be the reason that stops being safe.
    """

    started_at: float = field(default_factory=time.monotonic)
    turns: int = 0
    last_turn_at: float = 0.0
    failures_in_a_row: int = 0
    last_window: str = ""
    turns_on_this_window: int = 0
    was_away: bool = False

    # --- observation --------------------------------------------------------

    def note_turn(self, window: str = "", *, now: float | None = None) -> None:
        """One spoken exchange happened. Called before the turn is framed."""
        now = time.monotonic() if now is None else now
        self.turns += 1
        self.last_turn_at = now
        if window and window == self.last_window:
            self.turns_on_this_window += 1
        else:
            self.last_window = window
            self.turns_on_this_window = 1

    def note_failure(self) -> None:
        """Something it tried did not work. Refused, errored, or came back empty."""
        self.failures_in_a_row += 1

    def note_success(self) -> None:
        self.failures_in_a_row = 0

    def note_idle(self, idle_s: float) -> None:
        """Remember that they were away, so the next turn can know they are back.

        Read from the snapshot rather than measured here, because idle time is a
        system-wide fact about the keyboard and mouse, not about this process.
        """
        if idle_s > AWAY_S:
            self.was_away = True

    # --- the line ------------------------------------------------------------

    def line(self, *, now: float | None = None, clock: time.struct_time | None = None) -> str:
        """How to pitch the turn about to happen. Usually empty, on purpose."""
        now = time.monotonic() if now is None else now
        clock = time.localtime() if clock is None else clock
        notes: list[str] = []

        if self.failures_in_a_row >= REPEATED_FAILURES:
            # First, and alone if it fires. When something is not working, how
            # long the session has run stops being the interesting fact.
            return (
                f"the last {self.failures_in_a_row} attempts at this did not work. "
                "Change approach rather than retrying, and say what you are trying "
                "instead. No reassurance."
            )

        late = clock.tm_hour >= LATE_HOUR or clock.tm_hour < EARLY_HOUR
        long_session = (now - self.started_at) > LONG_SESSION_S
        if late and long_session:
            notes.append("it is late and this has run long, so be shorter than usual")
        elif late:
            notes.append("it is late, keep it short")

        if self.was_away:
            # Cleared here rather than in note_idle: it is true for exactly the
            # one turn where they come back, and stale after that.
            self.was_away = False
            notes.append("they have been away from the keyboard, do not recap")
        elif self.turns > 1 and (now - self.last_turn_at) < MID_FLOW_GAP_S:
            notes.append("this is mid exchange, no preamble")

        if self.turns_on_this_window >= 4 and not late:
            notes.append("they have been on one thing a while, stay out of the way")

        return ", ".join(notes)


def attach(context: str, register_line: str) -> str:
    """Add the register to the machine context block, if there is one to add.

    Labelled separately from the sensor readings above it because they are two
    different kinds of thing: one is what is true, the other is what to do about
    it, and blurring them invites the model to read the CPU load as an
    instruction.
    """
    if not register_line.strip():
        return context
    if not context.strip():
        return f"how to pitch this: {register_line}"
    return f"{context.rstrip()}\nhow to pitch this: {register_line}"
