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

# The offline brain has its own two ways of being broken, and both read as
# "Claude is down" unless they are named. They are close to the opposite: the
# subscription is fine and the thing on this machine is not running, so the fix
# is a command here rather than a login somewhere else.
NO_LOCAL_SERVER = "no-local-server"
NO_LOCAL_MODEL = "no-local-model"

# Connection refused as an HTTP client reports it, plus the wording a local
# server uses when it is running but has never heard of the model asked for.
_NO_SERVER_PHRASES = re.compile(
    r"econnrefused|connection refused|failed to connect|could not connect|"
    r"connect error|fetch failed|socket hang up",
    re.IGNORECASE,
)
_NO_MODEL_PHRASES = re.compile(
    r"model [\"']?[\w.:\-]+[\"']? not found|no such model|try pulling it|"
    r"unknown model",
    re.IGNORECASE,
)

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
    NO_LOCAL_SERVER: (
        "I'm running offline and the model on this machine isn't answering. "
        "Start it with ollama serve, then ask me again."
    ),
    NO_LOCAL_MODEL: (
        "I'm running offline and the local model is missing. "
        "Check the brain settings, then ask me again."
    ),
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


def classify_local(text: str) -> str | None:
    """Which offline failure this error text describes, if it is one at all.

    Separate from `classify` because it reads raw error text rather than a
    finished turn: a local server that is not listening fails before any turn
    exists, so the message arrives as a BrainError or on stderr.
    """
    if not text:
        return None
    if _NO_SERVER_PHRASES.search(text):
        return NO_LOCAL_SERVER
    if _NO_MODEL_PHRASES.search(text):
        return NO_LOCAL_MODEL
    return None


def spoken_line(kind: str) -> str:
    """What Vesper says instead of the CLI's text. Plain, short, no link."""
    return _SPOKEN.get(kind, _SPOKEN[OTHER])


# Sentences that are already fit to be said out loud. Everything else `ask()`
# produces describes plumbing, and two of them append an OS error string, which
# on this machine means reading a file path aloud in a voice.
_SPEAKABLE_BREAKS = frozenset({"that took too long, so I stopped waiting"})

_BROKEN = "I lost my connection to Claude. Give me a moment and ask me again."


def spoken_break(message: str) -> str:
    """What to say for a BrainError, which is not always what it says.

    The same rule `spoken_line` applies to the CLI's error text, applied to
    ours: what broke belongs in the log, and what the person in the room needs
    to hear is that it broke and what to do about it.
    """
    cleaned = (message or "").strip()
    return cleaned if cleaned in _SPEAKABLE_BREAKS else _BROKEN
