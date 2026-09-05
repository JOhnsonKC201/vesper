"""Naming a turn the CLI could not run, so it is never spoken as an answer.

The CLI reports a turn it failed to run as a normal `result` frame with
`is_error` set and its own explanation in the text. Before this module that text
took the same path as a real reply and was read aloud in Vesper's voice. On
2026-09-04 that was "Failed to authenticate: OAuth session expired and could not
be refreshed", three times, each counted as a success.

Two kinds matter. A dead login (`AUTH`) is fixed by a fresh child process once
the user has logged in again somewhere, so it earns a respawn and a retry. Every
other failure (`OTHER`) is a rate limit, an overloaded upstream or a bug, none of
which a respawn touches, so it earns one plain sentence and nothing else.

Pure functions only. Nothing here talks to a process or a socket.
"""

from __future__ import annotations

import re

from .protocol import TurnComplete

AUTH = "auth"
OTHER = "other"

# The CLI's own error code for a login that could not be refreshed, as it
# appears on the assistant frame. The phrases are a fallback for a build that
# stops sending the code, and are the wording the CLI has actually used.
_AUTH_CODE = "authentication_failed"
_AUTH_PHRASES = re.compile(
    r"failed to authenticate|oauth session expired|login expired|"
    r"please run /login|not logged in",
    re.IGNORECASE,
)

_SPOKEN = {
    AUTH: (
        "I can't reach Claude. The login on this machine has expired. "
        "Open a terminal, run claude login, then talk to me again."
    ),
    OTHER: "That didn't go through on Claude's side. Ask me again in a moment.",
}


def classify(done: TurnComplete) -> str | None:
    """AUTH, OTHER, or None for a turn that was not an error at all.

    Only `is_error` decides whether this is a failure. A real answer that
    happens to mention an expired login is an answer.
    """
    if not done.is_error:
        return None
    if done.api_error == _AUTH_CODE or _AUTH_PHRASES.search(done.text or ""):
        return AUTH
    return OTHER


def spoken_line(kind: str) -> str:
    """What Vesper says instead of the CLI's text. Plain, short, no link."""
    return _SPOKEN.get(kind, _SPOKEN[OTHER])
