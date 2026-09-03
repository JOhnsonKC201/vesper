"""Confirm before anything destructive, and default to refusing.

A brief constraint: delete, send, purchase. The trap in writing one of these is
matching on the planner's stated intent, which is the one input an agent that
has gone wrong controls completely. So this matches on two things the agent
cannot talk its way around: the verb it is actually about to perform, and the
text of the control it is about to press.

Refusing by default matters more than the word list. An unrecognised verb is
not automatically safe, and a run where the guard silently allowed something is
worse than a run that stopped early, because the first one still reports a
success rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matched against the visible label of the control being pressed, lowercased.
# Deliberately blunt. A false stop costs a step and shows up in the trace; a
# false pass costs whatever was on the other side of the button.
DESTRUCTIVE_WORDS = (
    "delete", "remove", "erase", "discard", "trash", "empty bin",
    "send", "reply", "reply all", "forward", "post", "publish", "share",
    "buy", "purchase", "order", "checkout", "pay", "confirm order",
    "uninstall", "format", "wipe", "reset", "restore defaults",
    "shut down", "sign out", "log out", "unsubscribe",
    "overwrite", "replace", "merge", "commit", "push", "force",
)

# Verbs in the action space that are destructive whatever they are pointed at.
DESTRUCTIVE_VERBS = frozenset({"delete_file", "run_shell"})

# Keys that destroy without any label to read. Typing into a focused list and
# pressing one of these is the classic way an agent deletes something while
# technically only having "typed".
DESTRUCTIVE_KEYS = frozenset({"delete", "del", "shift+delete", "ctrl+d"})

_WORD = re.compile(r"[a-z0-9 ]+")


def _normalise(text: str) -> str:
    """Lowercase, strip accelerators and punctuation, collapse spaces.

    Windows labels carry ampersand accelerators ("&Delete") and ellipses
    ("Delete..."), and matching without stripping them misses the exact buttons
    this exists to catch.
    """
    text = (text or "").lower().replace("&", "")
    text = " ".join(_WORD.findall(text))
    return " ".join(text.split())


@dataclass
class Decision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class Guard:
    """Decides whether an action may proceed.

    `confirm` is injected rather than assumed: the harness runs unattended and
    must refuse, while a person driving it interactively can be asked. Defaults
    to refusing so that forgetting to wire it up fails closed.
    """

    confirm: object = None
    words: tuple[str, ...] = DESTRUCTIVE_WORDS
    verbs: frozenset = DESTRUCTIVE_VERBS
    keys: frozenset = DESTRUCTIVE_KEYS
    # Everything the guard stopped, for the report.
    blocked: list[dict] = field(default_factory=list)

    def is_destructive(self, verb: str, args: dict, element=None) -> str:
        """Returns the reason it is destructive, or "" if it is not."""
        if verb in self.verbs:
            return f"{verb} is destructive by definition"

        if verb == "hotkey":
            pressed = _normalise(str(args.get("keys", ""))).replace(" ", "+")
            if pressed in self.keys:
                return f"hotkey {pressed} destroys without a label to read"

        # The label of the thing being pressed, whether it came from the
        # element or from the planner naming a target by text.
        label = ""
        if element is not None:
            label = element.name
        label = _normalise(label or str(args.get("target", "")))
        if label:
            for word in self.words:
                # Word-boundary, not substring: "undelete" and "resend" are not
                # the same as "delete" and "send", and neither is "shared".
                if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", label):
                    return f"the control reads {label!r}, which matches {word!r}"
        return ""

    def check(self, verb: str, args: dict, element=None) -> Decision:
        reason = self.is_destructive(verb, args, element)
        if not reason:
            return Decision(True)

        if self.confirm is None:
            self.blocked.append({"verb": verb, "args": args, "reason": reason})
            return Decision(False, f"refused, no confirmer wired: {reason}")

        question = f"Allow {verb} {args}? {reason}"
        try:
            said_yes = bool(self.confirm(question))
        except Exception as exc:
            # A confirmer that raises is not a yes.
            self.blocked.append({"verb": verb, "args": args, "reason": str(exc)})
            return Decision(False, f"refused, confirmer failed: {exc}")

        if said_yes:
            return Decision(True, f"confirmed: {reason}")
        self.blocked.append({"verb": verb, "args": args, "reason": reason})
        return Decision(False, f"declined: {reason}")
