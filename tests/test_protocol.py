"""Contract tests for the Claude Code stream-json protocol.

The fixture in tests/fixtures/ is a real recorded transcript from the CLI, not a
hand-written guess. If Anthropic changes the frame shape, these fail loudly
instead of Vesper silently going mute.
"""

import json
from pathlib import Path

import pytest

from vesper.brain.protocol import (
    BrainError,
    PermissionNeeded,
    SessionReady,
    StreamParser,
    TextDelta,
    ToolStarted,
    TurnComplete,
    encode_user_message,
)

FIXTURE = Path(__file__).parent / "fixtures" / "turn_with_tool.jsonl"


def parse_all(lines):
    parser = StreamParser()
    events = []
    for line in lines:
        events.extend(parser.feed(line))
    return events


def of_type(events, kind):
    return [e for e in events if isinstance(e, kind)]


# --- outbound ---------------------------------------------------------------


def test_encode_user_message_is_one_json_line():
    line = encode_user_message("what is on my screen")
    assert line.endswith("\n")
    assert "\n" not in line[:-1]
    frame = json.loads(line)
    assert frame["type"] == "user"
    assert frame["message"]["content"][0]["text"] == "what is on my screen"


def test_encode_user_message_keeps_unicode_readable():
    line = encode_user_message("caf\u00e9 na\u00efve")
    assert "caf\u00e9" in line
    assert json.loads(line)["message"]["content"][0]["text"] == "caf\u00e9 na\u00efve"


# --- real transcript --------------------------------------------------------


def test_fixture_exists():
    assert FIXTURE.exists(), "record it with scripts/record_fixture.sh"


def test_real_transcript_yields_a_session_id():
    events = parse_all(FIXTURE.read_text(encoding="utf-8").splitlines())
    ready = of_type(events, SessionReady)
    assert len(ready) == 1
    assert len(ready[0].session_id) == 36  # uuid


def test_real_transcript_reports_the_tool_call():
    events = parse_all(FIXTURE.read_text(encoding="utf-8").splitlines())
    tools = of_type(events, ToolStarted)
    assert [t.name for t in tools] == ["Bash"]
    assert "print(7*6)" in tools[0].detail


def test_real_transcript_completes_with_the_answer():
    events = parse_all(FIXTURE.read_text(encoding="utf-8").splitlines())
    done = of_type(events, TurnComplete)
    assert len(done) == 1
    assert "42" in done[0].text
    assert done[0].is_error is False
    assert done[0].turns >= 2  # tool call plus answer
    assert done[0].cost_usd > 0
    assert done[0].ttft_ms > 0


def test_real_transcript_streams_deltas_before_completion():
    """Deltas must arrive before TurnComplete or there is nothing to speak early."""
    events = parse_all(FIXTURE.read_text(encoding="utf-8").splitlines())
    first_delta = next(i for i, e in enumerate(events) if isinstance(e, TextDelta))
    completion = next(i for i, e in enumerate(events) if isinstance(e, TurnComplete))
    assert first_delta < completion


def test_deltas_reassemble_into_the_final_text():
    events = parse_all(FIXTURE.read_text(encoding="utf-8").splitlines())
    streamed = "".join(e.text for e in of_type(events, TextDelta)).strip()
    final = of_type(events, TurnComplete)[0].text
    assert streamed == final


# --- robustness -------------------------------------------------------------


def test_garbage_lines_are_ignored_not_raised():
    # The CLI interleaves plain text occasionally. Going mute over it is worse
    # than skipping it.
    assert parse_all(["not json", "", "   ", "<html>", "{broken", "null", "[]"]) == []


def test_unknown_frame_types_are_ignored():
    assert parse_all(['{"type":"telemetry","payload":{"x":1}}']) == []


def test_result_prefers_its_own_text_over_assembled_deltas():
    """On a retry the deltas can hold a discarded first attempt."""
    events = parse_all([
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"discarded draft"}}}',
        '{"type":"result","is_error":false,"result":"the real answer",'
        '"session_id":"s1","usage":{}}',
    ])
    assert of_type(events, TurnComplete)[0].text == "the real answer"


def test_result_falls_back_to_deltas_when_result_field_is_empty():
    events = parse_all([
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"assembled"}}}',
        '{"type":"result","is_error":false,"result":"","session_id":"s1","usage":{}}',
    ])
    assert of_type(events, TurnComplete)[0].text == "assembled"


def test_permission_denial_becomes_a_spoken_ask():
    events = parse_all([
        '{"type":"result","is_error":false,"result":"I need permission.",'
        '"session_id":"s1","usage":{},"permission_denials":'
        '[{"tool_name":"Write","tool_input":{"file_path":"C:/notes.txt"}}]}',
    ])
    denials = of_type(events, PermissionNeeded)
    assert len(denials) == 1
    assert denials[0].tool == "Write"
    assert "notes.txt" in denials[0].detail


def test_buffer_resets_between_turns():
    parser = StreamParser()
    for line in [
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"first"}}}',
        '{"type":"result","is_error":false,"result":"","session_id":"s","usage":{}}',
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"second"}}}',
    ]:
        list(parser.feed(line))
    assert parser.turn_text == "second"


def test_error_result_is_flagged():
    events = parse_all([
        '{"type":"result","is_error":true,"result":"boom","session_id":"s","usage":{}}'
    ])
    assert of_type(events, TurnComplete)[0].is_error is True


def test_usage_numbers_are_captured_for_cost_tracking():
    events = parse_all([
        '{"type":"result","is_error":false,"result":"ok","session_id":"s",'
        '"total_cost_usd":0.0081,"duration_api_ms":5470,"ttft_ms":1590,"num_turns":2,'
        '"usage":{"input_tokens":3,"output_tokens":11,'
        '"cache_read_input_tokens":9726,"cache_creation_input_tokens":920}}'
    ])
    done = of_type(events, TurnComplete)[0]
    assert (done.cost_usd, done.duration_ms, done.ttft_ms) == (0.0081, 5470, 1590)
    assert (done.input_tokens, done.output_tokens) == (3, 11)
    assert (done.cache_read_tokens, done.cache_write_tokens) == (9726, 920)


@pytest.mark.parametrize(
    "tool_input,expected",
    [
        ({"command": "ls -la"}, "ls -la"),
        ({"file_path": "C:/a/b.py"}, "C:/a/b.py"),
        ({"pattern": "TODO"}, "TODO"),
        ({"query": "weather"}, "weather"),
        ({}, ""),
        ({"unknown_key": "x"}, ""),
        ("not a dict", ""),
    ],
)
def test_tool_detail_extraction(tool_input, expected):
    frame = json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": "T", "input": tool_input}]},
    })
    tools = of_type(parse_all([frame]), ToolStarted)
    assert tools[0].detail == expected
