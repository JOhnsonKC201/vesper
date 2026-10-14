"""Fanning a request out to parallel subagents.

Measured against the real CLI (2.1.299) on 2026-10-13 before any of this was
written, because every obvious design was wrong in a way only the CLI could say:

- `--safe-mode` ignores `--agents`, so Vesper cannot define its own subagent.
  The built-in ones (Explore, general-purpose, Plan) survive it.
- Two Agent calls in one message ran IN THE BACKGROUND by default: the turn
  ended on "both agents are still counting" and two more `result` frames
  arrived afterwards. Vesper reads a `result` as the end of the spoken turn, so
  the real answer would have landed in the next one.
  CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 made the same request one turn, one
  result, both counts, 4.6 s, $0.04.
- The gate holds inside a subagent. A general-purpose subagent told to
  `echo hi > file` and to Write a file was refused both, and both refusals came
  back in the parent's `permission_denials`, which is what the spoken yes-or-no
  already reads. The Agent tool itself needs no grant.
"""

import json

from vesper.brain.claude import BrainConfig, child_env
from vesper.brain.protocol import StreamParser, TextDelta, ToolStarted, tool_detail


def _config(**kwargs):
    defaults = dict(model="opus", tools=("Bash", "Read", "WebSearch"))
    defaults.update(kwargs)
    return BrainConfig(**defaults)


def _tools(config):
    argv = config.argv()
    return argv[argv.index("--tools") + 1].split(",")


def _feed(parser, frame):
    return list(parser.feed(json.dumps(frame)))


# --- the tool --------------------------------------------------------------


def test_subagents_add_the_agent_tool_to_a_cloud_brain():
    assert "Agent" in _tools(_config(subagents=True))


def test_subagents_off_means_no_agent_tool():
    assert "Agent" not in _tools(_config(subagents=False))


def test_off_is_the_default_so_a_bare_config_changes_nothing():
    assert BrainConfig().subagents is False


def test_a_local_brain_never_fans_out():
    """A 3B model on this machine running several copies of itself is slow at
    best, and nothing was measured for it."""
    assert "Agent" not in _tools(_config(subagents=True, local=True))


def test_agent_is_not_listed_twice_when_config_already_names_it():
    tools = _tools(_config(subagents=True, tools=("Read", "Agent")))
    assert tools.count("Agent") == 1


def test_the_allowlist_is_untouched_by_subagents():
    """The Agent tool needs no grant, and every tool a subagent calls goes
    through the same gate. Nothing here should widen what runs unasked."""
    plain = _config(subagents=False, allowed_tools=("Read",)).argv()
    fanned = _config(subagents=True, allowed_tools=("Read",)).argv()
    flag = "--allowedTools"
    assert plain[plain.index(flag) + 1] == fanned[fanned.index(flag) + 1]


# --- the prompt ------------------------------------------------------------


def test_the_note_rides_along_only_when_the_tool_does():
    on = _config(subagents=True, subagents_note="FAN OUT NOTE")
    off = _config(subagents=False, subagents_note="FAN OUT NOTE")
    local = _config(subagents=True, local=True, subagents_note="FAN OUT NOTE")
    assert "FAN OUT NOTE" in on.prompt_for_spawn()
    assert "FAN OUT NOTE" not in off.prompt_for_spawn()
    assert "FAN OUT NOTE" not in local.prompt_for_spawn()


# --- one spoken question, one result ----------------------------------------


def test_background_tasks_are_always_off_in_the_child():
    assert child_env(_config(subagents=True))["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert child_env(None)["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


# --- a subagent never speaks -------------------------------------------------


def _delta(text, parent=None):
    return {
        "type": "stream_event",
        "parent_tool_use_id": parent,
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    }


def test_a_subagents_streamed_text_is_never_spoken():
    parser = StreamParser()
    assert _feed(parser, _delta("internal notes", parent="toolu_1")) == []
    assert parser.turn_text == ""


def test_the_main_agents_streamed_text_still_is():
    parser = StreamParser()
    assert _feed(parser, _delta("Nine files.")) == [TextDelta("Nine files.")]


def test_an_agent_call_is_described_by_its_description():
    detail = tool_detail(
        "Agent",
        {"description": "Count brain files", "prompt": "a long brief", "subagent_type": "Explore"},
    )
    assert detail == "Count brain files"


def test_a_subagents_own_tool_calls_still_show_on_screen():
    parser = StreamParser()
    frame = {
        "type": "assistant",
        "parent_tool_use_id": "toolu_1",
        "message": {"content": [
            {"type": "tool_use", "id": "toolu_2", "name": "Glob", "input": {"pattern": "**/*.py"}},
        ]},
    }
    assert _feed(parser, frame) == [ToolStarted("Glob", "**/*.py")]
