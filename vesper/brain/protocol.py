"""Parsing of the Claude Code CLI's stream-json protocol.

Vesper drives `claude -p --input-format stream-json --output-format stream-json`
as a long-lived child process. That gives one continuous conversation with full
tool access, running on the user's subscription, with no API key anywhere.

This module is deliberately pure: bytes in, events out, no I/O and no subprocess.
That makes the protocol testable against recorded transcripts, so a change in the
CLI's output shape fails a unit test instead of silently breaking speech.

Observed frame shapes (recorded from claude 2.1.250):

    {"type":"system","subtype":"init","session_id":"...","tools":[...]}
    {"type":"stream_event","event":{"type":"content_block_delta",
        "delta":{"type":"text_delta","text":"Hel"}}}
    {"type":"assistant","message":{"content":[
        {"type":"text","text":"..."},
        {"type":"tool_use","name":"Bash","input":{"command":"..."}}]}}
    {"type":"result","is_error":false,"num_turns":2,"session_id":"...",
        "total_cost_usd":0.008,"duration_api_ms":5470,"usage":{...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator


# --- Events -----------------------------------------------------------------


@dataclass(frozen=True)
class SessionReady:
    """The CLI finished booting and gave us a session id to resume later."""

    session_id: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the spoken answer, as it is generated."""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    """Claude decided to touch the machine. Surfaced so the UI can show it."""

    name: str
    detail: str = ""


@dataclass(frozen=True)
class PermissionNeeded:
    """Claude tried something the allowlist forbids.

    Under the "read freely, ask before changing" policy this is a normal,
    expected outcome, not an error. Vesper should say out loud what it wanted
    to do and wait for a yes.
    """

    tool: str
    detail: str = ""


@dataclass(frozen=True)
class TurnComplete:
    """The turn finished. Carries the cost and latency telemetry we care about."""

    text: str
    session_id: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    ttft_ms: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    is_error: bool = False


@dataclass(frozen=True)
class BrainError:
    """Something went wrong that the user should hear about."""

    message: str


Event = (
    SessionReady
    | TextDelta
    | ToolStarted
    | PermissionNeeded
    | TurnComplete
    | BrainError
)


# --- Outbound framing -------------------------------------------------------


def encode_user_message(text: str) -> str:
    """Frame a user turn for the CLI's stdin. Returns a single newline-terminated line."""
    frame = {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }
    return json.dumps(frame, ensure_ascii=False) + "\n"


# --- Inbound parsing --------------------------------------------------------


def _tool_detail(name: str, tool_input: dict) -> str:
    """One short human phrase describing a tool call, for the terminal."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "file_path", "pattern", "query", "url", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return ""


class StreamParser:
    """Turns CLI stdout lines into Vesper events.

    Holds the small amount of state the protocol needs: the assembled text of the
    current turn. Everything else is stateless.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self.session_id: str = ""

    def reset_turn(self) -> None:
        self._buffer.clear()

    @property
    def turn_text(self) -> str:
        return "".join(self._buffer)

    def feed(self, line: str) -> Iterator[Event]:
        """Parse one line of CLI stdout. Yields zero or more events.

        Non-JSON lines are ignored rather than raised. The CLI occasionally
        interleaves plain text (progress, warnings) and killing the voice loop
        over a stray line would be the wrong trade.
        """
        line = line.strip()
        if not line or not line.startswith("{"):
            return
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(frame, dict):
            return

        kind = frame.get("type")

        if kind == "system":
            if frame.get("subtype") == "init":
                self.session_id = frame.get("session_id", "") or self.session_id
                tools = frame.get("tools") or []
                yield SessionReady(
                    session_id=self.session_id,
                    tools=tuple(t for t in tools if isinstance(t, str)),
                )
            return

        if kind == "stream_event":
            event = frame.get("event") or {}
            if event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        self._buffer.append(text)
                        yield TextDelta(text)
            return

        if kind == "assistant":
            for block in (frame.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = block.get("name", "tool")
                    yield ToolStarted(name, _tool_detail(name, block.get("input") or {}))
            return

        if kind == "result":
            usage = frame.get("usage") or {}
            self.session_id = frame.get("session_id", "") or self.session_id

            for denial in frame.get("permission_denials") or []:
                if isinstance(denial, dict):
                    tool = denial.get("tool_name") or denial.get("tool") or "something"
                    yield PermissionNeeded(
                        tool, _tool_detail(tool, denial.get("tool_input") or {})
                    )

            # `result` is the CLI's own authoritative final text. Prefer it over
            # our assembled deltas: partial-message streaming can be disabled,
            # and on a retry the deltas may contain a discarded first attempt.
            final = frame.get("result")
            if not isinstance(final, str) or not final.strip():
                final = self.turn_text
            yield TurnComplete(
                text=final.strip(),
                session_id=self.session_id,
                cost_usd=float(frame.get("total_cost_usd") or 0.0),
                duration_ms=int(frame.get("duration_api_ms") or 0),
                ttft_ms=int(frame.get("ttft_ms") or 0),
                turns=int(frame.get("num_turns") or 0),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
                is_error=bool(frame.get("is_error")),
            )
            self.reset_turn()
            return
