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

A turn the CLI could not run at all (recorded 2026-09-04, when the saved login
had expired) is one `assistant` frame whose only block is the CLI's own text,
marked "error":"authentication_failed", followed by a `result` frame with
"is_error":true. No deltas arrive. The text is the CLI talking about itself and
must never be spoken as Vesper's answer; `failures.py` names it instead.
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

    Carries the tool's actual input, not just its name, because "I want to run
    a command" is not a question anyone can answer. The input arrives in the
    preceding `assistant` frame rather than in the denial, so the parser
    remembers tool calls by id for the length of a turn and joins them here.
    """

    tool: str
    detail: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_use_id: str = ""
    message: str = ""


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
    # Set only when `is_error` is. The result frame's subtype names how the
    # turn failed (`error_during_execution`); the api error is the code the
    # CLI put on its error message (`authentication_failed`), which is more
    # precise than the text and does not change when the wording does.
    error_subtype: str = ""
    api_error: str = ""


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


# A turn that makes more tool calls than this has gone wrong in some other way;
# the cache is cleared rather than allowed to grow for the life of the process.
_MAX_REMEMBERED_CALLS = 200


def tool_detail(name: str, tool_input: dict) -> str:
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
        # Tool calls seen this turn, by id, so a later denial can say what was
        # actually being attempted. Bounded: a turn that makes hundreds of tool
        # calls should not grow this without limit.
        self._tool_uses: dict[str, tuple[str, dict]] = {}
        self._denied: set[str] = set()
        # The error code from an assistant frame the CLI flagged as an API
        # error, carried to the result so the turn can be named precisely.
        self._api_error: str = ""

    def reset_turn(self) -> None:
        self._buffer.clear()
        self._tool_uses.clear()
        self._denied.clear()
        self._api_error = ""

    @property
    def turn_text(self) -> str:
        return "".join(self._buffer)

    def _denial(
        self,
        *,
        tool: str,
        tool_use_id: str = "",
        message: str = "",
        tool_input: dict | None = None,
    ) -> PermissionNeeded | None:
        """Build one denial event, or None if this call was already reported."""
        if tool_use_id and tool_use_id in self._denied:
            return None
        remembered = self._tool_uses.get(tool_use_id) if tool_use_id else None
        resolved = tool_input if isinstance(tool_input, dict) and tool_input else None
        if resolved is None and remembered is not None:
            resolved = remembered[1]
        resolved = resolved or {}
        if tool_use_id:
            self._denied.add(tool_use_id)
        return PermissionNeeded(
            tool=tool,
            detail=tool_detail(tool, resolved),
            tool_input=resolved,
            tool_use_id=tool_use_id,
            message=message,
        )

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
            subtype = frame.get("subtype")
            if subtype == "init":
                self.session_id = frame.get("session_id", "") or self.session_id
                tools = frame.get("tools") or []
                yield SessionReady(
                    session_id=self.session_id,
                    tools=tuple(t for t in tools if isinstance(t, str)),
                )
            elif subtype == "permission_denied":
                # Real time, mid-turn, one frame per refused call. This is the
                # only moment Vesper learns it was stopped, so it is also the
                # only chance to ask before the turn moves on.
                event = self._denial(
                    tool=frame.get("tool_name") or "something",
                    tool_use_id=frame.get("tool_use_id") or "",
                    message=str(frame.get("message") or ""),
                )
                if event is not None:
                    yield event
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
            # A turn the CLI could not run arrives as an assistant frame whose
            # text is the CLI's explanation, marked with an error code. Recorded
            # 2026-09-04: {"error":"authentication_failed","isApiErrorMessage":true}.
            error = frame.get("error")
            if isinstance(error, str) and error:
                self._api_error = error
            for block in (frame.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = block.get("name", "tool")
                    tool_input = block.get("input") or {}
                    block_id = block.get("id") or ""
                    if block_id and isinstance(tool_input, dict):
                        if len(self._tool_uses) > _MAX_REMEMBERED_CALLS:
                            self._tool_uses.clear()
                        self._tool_uses[block_id] = (name, tool_input)
                    yield ToolStarted(name, tool_detail(name, tool_input))
            return

        if kind == "result":
            usage = frame.get("usage") or {}
            self.session_id = frame.get("session_id", "") or self.session_id

            # The result frame repeats every denial from the turn. Anything
            # already announced live is skipped, so Vesper asks about an action
            # once rather than once per report of it.
            for denial in frame.get("permission_denials") or []:
                if not isinstance(denial, dict):
                    continue
                event = self._denial(
                    tool=denial.get("tool_name") or denial.get("tool") or "something",
                    tool_use_id=denial.get("tool_use_id") or "",
                    message=str(denial.get("message") or ""),
                    tool_input=denial.get("tool_input") or {},
                )
                if event is not None:
                    yield event

            # `result` is the CLI's own authoritative final text. Prefer it over
            # our assembled deltas: partial-message streaming can be disabled,
            # and on a retry the deltas may contain a discarded first attempt.
            final = frame.get("result")
            if not isinstance(final, str) or not final.strip():
                final = self.turn_text
            is_error = bool(frame.get("is_error"))
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
                is_error=is_error,
                error_subtype=str(frame.get("subtype") or "") if is_error else "",
                api_error=self._api_error if is_error else "",
            )
            self.reset_turn()
            return
