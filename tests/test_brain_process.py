"""ClaudeBrain against a real subprocess.

Uses tests/fake_claude.py, which speaks the same stream-json protocol as the
real CLI. That means the pipes, reader threads, framing, session handling,
interrupts and crash recovery are all genuinely exercised, at zero cost and
with no network.

The one thing these cannot check is that the real CLI's flags are still valid.
tests/fixtures/turn_with_tool.jsonl covers that side, being a recorded
transcript from the actual binary.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from vesper.brain.claude import BrainConfig, ClaudeBrain
from vesper.brain.protocol import (
    BrainError,
    SessionReady,
    TextDelta,
    ToolStarted,
    TurnComplete,
)

FAKE = Path(__file__).parent / "fake_claude.py"


@pytest.fixture
def brain_factory():
    created = []

    def make(replies=(), **env):
        for key, value in {
            "FAKE_CLAUDE_REPLIES": "|".join(replies),
            **{k: str(v) for k, v in env.items()},
        }.items():
            os.environ[key] = value

        config = BrainConfig(executable=sys.executable, model="fake")
        # Prepend the script so the child is `python fake_claude.py ...`.
        original_argv = config.argv

        def argv(resume_session=""):
            args = original_argv(resume_session)
            return [args[0], str(FAKE)] + args[1:]

        config.argv = argv
        brain = ClaudeBrain(config)
        created.append(brain)
        return brain

    yield make

    for brain in created:
        brain.stop()
    for key in list(os.environ):
        if key.startswith("FAKE_CLAUDE_"):
            del os.environ[key]


def collect(brain, text):
    return list(brain.ask(text))


def of(events, kind):
    return [e for e in events if isinstance(e, kind)]


# --- basics -----------------------------------------------------------------


def test_a_turn_round_trips_through_a_real_subprocess(brain_factory):
    brain = brain_factory(["The disk is fine."])
    events = collect(brain, "how is the disk")

    assert of(events, SessionReady), "no init frame"
    assert "".join(e.text for e in of(events, TextDelta)) == "The disk is fine."
    done = of(events, TurnComplete)[0]
    assert done.text == "The disk is fine."
    assert done.is_error is False
    assert done.cost_usd == 0.008
    assert done.ttft_ms == 400


def test_the_process_survives_across_turns(brain_factory):
    brain = brain_factory(["first", "second", "third"])
    for expected in ("first", "second", "third"):
        assert of(collect(brain, "go"), TurnComplete)[0].text == expected
    assert brain.alive
    assert brain.turn_count == 3


def test_session_id_is_captured_and_reused(brain_factory):
    brain = brain_factory(["hi"], FAKE_CLAUDE_SESSION="abc-123")
    collect(brain, "hello")
    assert brain.session_id == "abc-123"


def test_cost_accumulates_across_the_session(brain_factory):
    brain = brain_factory(["a", "b"])
    collect(brain, "one")
    collect(brain, "two")
    assert brain.total_cost_usd == pytest.approx(0.016)


def test_tool_calls_are_reported(brain_factory):
    brain = brain_factory(["done"], FAKE_CLAUDE_TOOL="Bash")
    tools = of(collect(brain, "run something"), ToolStarted)
    assert [t.name for t in tools] == ["Bash"]
    assert tools[0].detail == "whoami"


def test_empty_input_is_ignored(brain_factory):
    brain = brain_factory(["never"])
    assert collect(brain, "   ") == []


def test_garbage_output_does_not_break_the_stream(brain_factory):
    """The real CLI interleaves plain log lines. Going mute over one is worse
    than skipping it."""
    brain = brain_factory(["still works"], FAKE_CLAUDE_GARBAGE=1)
    assert of(collect(brain, "hello"), TurnComplete)[0].text == "still works"


# --- lifecycle --------------------------------------------------------------


def test_starting_twice_is_harmless(brain_factory):
    brain = brain_factory(["ok"])
    brain.start()
    brain.start()
    assert of(collect(brain, "hi"), TurnComplete)[0].text == "ok"


def test_stop_terminates_the_process(brain_factory):
    brain = brain_factory(["ok"])
    collect(brain, "hi")
    assert brain.alive
    brain.stop()
    assert not brain.alive


def test_ask_starts_the_process_if_it_was_never_started(brain_factory):
    brain = brain_factory(["lazy start"])
    assert not brain.alive
    assert of(collect(brain, "hi"), TurnComplete)[0].text == "lazy start"


def test_a_crash_mid_turn_is_reported_rather_than_hanging(brain_factory):
    brain = brain_factory(["fine"], FAKE_CLAUDE_CRASH=1)
    collect(brain, "first")  # succeeds
    events = collect(brain, "second")  # child exits instead of answering
    assert of(events, BrainError), "a dead brain must surface an error"


def test_restart_keeps_the_session_and_stays_usable(brain_factory):
    """A crashed brain should cost a pause, not the conversation.

    The fake restarts with a fresh script, so it replays from the top; the real
    CLI resumes server side. What is asserted here is what this layer is
    actually responsible for: the session id survives and the brain answers
    again. That --resume is passed is covered by test_resume_is_only_added.
    """
    brain = brain_factory(["one", "two"], FAKE_CLAUDE_SESSION="keep-me")
    collect(brain, "hello")
    assert brain.session_id == "keep-me"

    brain.restart()
    assert brain.alive
    assert brain.session_id == "keep-me"
    assert of(collect(brain, "again"), TurnComplete), "brain is mute after restart"


def test_busy_is_false_when_idle(brain_factory):
    brain = brain_factory(["ok"])
    brain.start()
    assert brain.busy is False


def test_busy_is_true_during_a_turn(brain_factory):
    """The ambient loop reads this to avoid talking over a conversation."""
    brain = brain_factory(["a slow answer"], FAKE_CLAUDE_DELAY_MS=40)
    seen = []

    def watch():
        time.sleep(0.15)
        seen.append(brain.busy)

    watcher = threading.Thread(target=watch)
    brain.start()
    watcher.start()
    collect(brain, "take your time")
    watcher.join()
    assert seen == [True]


# --- interruption -----------------------------------------------------------


def test_interrupt_stops_yielding_events(brain_factory):
    brain = brain_factory(["one two three four five six seven"], FAKE_CLAUDE_DELAY_MS=30)
    received = []
    for event in brain.ask("say a lot"):
        received.append(event)
        if len(of(received, TextDelta)) == 2:
            brain.interrupt()
    assert len(of(received, TextDelta)) == 2, "kept yielding after interrupt"
    assert of(received, TurnComplete) == [], "completion must be suppressed too"


def test_the_stream_stays_usable_after_an_interrupt(brain_factory):
    """The abandoned turn is still drained, so the next one starts clean."""
    brain = brain_factory(
        ["first long answer here", "second answer"], FAKE_CLAUDE_DELAY_MS=20
    )
    for event in brain.ask("first"):
        if isinstance(event, TextDelta):
            brain.interrupt()
    assert of(collect(brain, "second"), TurnComplete)[0].text == "second answer"


def test_note_sends_a_turn_without_returning_anything(brain_factory):
    brain = brain_factory(["ignored", "real answer"])
    brain.note("[system] the user cut you off")
    assert of(collect(brain, "next"), TurnComplete)[0].text == "real answer"


# --- argv -------------------------------------------------------------------


def test_safe_mode_is_always_passed():
    """Without it a spoken sentence costs a dollar. This must never regress."""
    assert "--safe-mode" in BrainConfig().argv()


def test_streaming_flags_are_present():
    argv = BrainConfig().argv()
    for flag in (
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
    ):
        assert flag in argv


def test_no_api_key_flag_is_ever_passed():
    """The whole point is running on the subscription."""
    argv = " ".join(BrainConfig().argv())
    assert "--bare" not in argv, "--bare forces ANTHROPIC_API_KEY auth"
    assert "api-key" not in argv.lower()


def test_resume_is_only_added_when_asked():
    assert "--resume" not in BrainConfig().argv()
    assert "--resume" in BrainConfig().argv("session-1")


def test_tools_and_allowlist_are_comma_joined():
    argv = BrainConfig(
        tools=("Bash", "Read"), allowed_tools=("Bash(git status*)", "Read")
    ).argv()
    assert argv[argv.index("--tools") + 1] == "Bash,Read"
    assert argv[argv.index("--allowedTools") + 1] == "Bash(git status*),Read"


def test_add_dirs_are_repeated_flags():
    argv = BrainConfig(add_dirs=("C:/a", "C:/b")).argv()
    assert argv.count("--add-dir") == 2
