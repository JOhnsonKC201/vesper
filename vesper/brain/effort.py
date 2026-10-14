"""How hard the brain should think about one question.

Chosen here, on this machine, from the words alone. A second model call to
grade the question would cost a round trip on every turn to save one on some
of them, which is the wrong trade for something whose whole point is speed.

The rules are deliberately plain, and they only ever move effort in the
direction the words point:
  asked for care out loud  -> max
  needs reasoning, or long -> high
  short, and none of that  -> low
  anything else            -> whatever config.yaml says
"""

from __future__ import annotations

import json
import re

# Said out loud by someone who wants the slow, careful answer.
_DEEP_CUES = (
    "think hard", "think carefully", "think it through", "think this through",
    "take your time", "really think", "deep dive", "be thorough", "double check",
)

# Words that mean the answer has to be worked out rather than looked up.
_HARD_CUES = (
    "why", "how come", "explain", "plan", "compare", "analyze", "analyse",
    "figure out", "debug", "design", "decide", "should i", "pros and cons",
    "strategy", "draft", "write", "summarize", "summarise", "proof", "prove",
    "calculate", "solve", "research", "review", "difference between",
    "what if", "help me",
)

# Up to this many words with no cue is a quick question: the time, a window,
# a single fact. Past it the configured default stands.
_SHORT_WORDS = 6

# Past this many words a request is treated as hard even without a cue. People
# do not describe a quick thing at length.
_LONG_WORDS = 25


def _has(text: str, cues: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(cue)}\b", text) for cue in cues)


def choose_effort(utterance: str, default: str = "") -> str:
    """The effort level for this utterance, or `default` when nothing decides it."""
    text = re.sub(r"[^a-z' ]", " ", (utterance or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = len(text.split())
    if _has(text, _DEEP_CUES):
        return "max"
    if _has(text, _HARD_CUES) or words > _LONG_WORDS:
        return "high"
    if 0 < words <= _SHORT_WORDS:
        return "low"
    return default


def effort_request(level: str, request_id: str) -> str:
    """The stream-json control frame that changes effort mid-session.

    Blank goes out as null, which puts the session back on the CLI's default:
    the same thing a blank `effort` in config.yaml means at spawn.
    """
    frame = {
        "type": "control_request",
        "request_id": request_id,
        "request": {
            "subtype": "apply_flag_settings",
            "settings": {"effortLevel": level or None},
        },
    }
    return json.dumps(frame) + "\n"
