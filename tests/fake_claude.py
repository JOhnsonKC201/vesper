"""A stand-in for the `claude` executable.

Speaks the same stream-json protocol as the real CLI, so ClaudeBrain's actual
subprocess machinery (pipes, reader threads, framing, session ids, restarts,
interrupts) is exercised for real without spending money or needing a network.

Behaviour is driven by environment variables so the test can shape one run:

    FAKE_CLAUDE_REPLIES   "|" separated replies, consumed in order
    FAKE_CLAUDE_DELAY_MS  pause between text deltas, for interrupt tests
    FAKE_CLAUDE_CRASH     exit abruptly after this many turns
    FAKE_CLAUDE_SESSION   session id to report
    FAKE_CLAUDE_GARBAGE   emit non-JSON lines too, as the real CLI sometimes does
    FAKE_CLAUDE_TOOL      report a tool call before answering
    FAKE_CLAUDE_AUTH_FAIL fail this many turns first, the way a dead login does
    FAKE_CLAUDE_LOGGED_IN what `auth status` answers: 1 (default) exits 0, 0 exits 1
"""

import json
import os
import sys
import time

CHUNK = 5

# Verbatim from the brain's transcript of 2026-09-04, when the saved login had
# expired. The real CLI sends this as one assistant frame marked with the error
# code, then an error result, and no deltas at all.
AUTH_ERROR = "Failed to authenticate: OAuth session expired and could not be refreshed"


def emit(frame):
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


def main() -> int:
    if sys.argv[1:3] == ["auth", "status"]:
        logged_in = os.environ.get("FAKE_CLAUDE_LOGGED_IN", "1") == "1"
        emit({"loggedIn": logged_in})
        return 0 if logged_in else 1

    replies = [r for r in os.environ.get("FAKE_CLAUDE_REPLIES", "").split("|") if r]
    delay = float(os.environ.get("FAKE_CLAUDE_DELAY_MS", "0")) / 1000.0
    crash_after = int(os.environ.get("FAKE_CLAUDE_CRASH", "0"))
    session = os.environ.get("FAKE_CLAUDE_SESSION", "11111111-2222-3333-4444-555555555555")
    garbage = os.environ.get("FAKE_CLAUDE_GARBAGE") == "1"
    tool = os.environ.get("FAKE_CLAUDE_TOOL", "")
    auth_fail = int(os.environ.get("FAKE_CLAUDE_AUTH_FAIL", "0"))

    resumed = "--resume" in sys.argv

    if garbage:
        sys.stdout.write("starting up, this line is not json\n")
        sys.stdout.flush()

    emit({
        "type": "system",
        "subtype": "init",
        "session_id": session,
        "cwd": os.getcwd(),
        "tools": ["Bash", "Read"],
        "resumed": resumed,
    })

    turns = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue

        turns += 1
        if crash_after and turns > crash_after:
            return 3

        if turns <= auth_fail:
            emit({
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": AUTH_ERROR}]},
                "error": "authentication_failed",
                "isApiErrorMessage": True,
            })
            emit({
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": AUTH_ERROR,
                "session_id": session,
                "num_turns": 1,
                "total_cost_usd": 0.0,
                "duration_api_ms": 0,
                "permission_denials": [],
                "usage": {},
            })
            continue

        reply = replies.pop(0) if replies else "Nothing to report."

        if tool:
            emit({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": tool, "input": {"command": "whoami"}}
                    ]
                },
            })

        if garbage:
            sys.stdout.write("[warn] a stray log line\n")
            sys.stdout.flush()

        for start in range(0, len(reply), CHUNK):
            emit({
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": reply[start : start + CHUNK]},
                },
            })
            if delay:
                time.sleep(delay)

        emit({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": reply,
            "session_id": session,
            "num_turns": 2 if tool else 1,
            "total_cost_usd": 0.008,
            "duration_api_ms": 1200,
            "ttft_ms": 400,
            "permission_denials": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": len(reply) // 4,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 0,
            },
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())
