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

        def argv(resume_session="", grants=()):
            args = original_argv(resume_session, grants)
            return [args[0], str(FAKE)] + args[1:]

        config.argv = argv
        original_status = config.status_argv

        def status_argv():
            args = original_status()
            return [args[0], str(FAKE)] + args[1:]

        config.status_argv = status_argv
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


# --- permission grants ------------------------------------------------------


def test_a_grant_respawns_the_process_and_keeps_the_conversation(brain_factory):
    """The CLI fixes its allowlist at spawn, so a spoken yes costs a respawn.
    The session has to survive it or approving something would also forget what
    was being discussed."""
    brain = brain_factory(replies=["First.", "Second."])
    collect(brain, "hello")
    session = brain.session_id
    assert session

    brain.grant(("Bash(git commit:*)",))
    assert brain.grants == ("Bash(git commit:*)",)
    assert brain.alive, "the brain must come back up after a grant"

    events = collect(brain, "now do it")
    assert of(events, TurnComplete), "the conversation continues after a grant"
    assert brain.session_id == session


def test_an_empty_grant_changes_nothing(brain_factory):
    """A refusal Vesper could not describe must not silently widen anything."""
    brain = brain_factory(replies=["Fine."])
    collect(brain, "hello")
    brain.grant(())
    assert brain.grants == ()


def test_revoking_puts_the_permission_back(brain_factory):
    brain = brain_factory(replies=["Fine.", "Fine."])
    collect(brain, "hello")
    brain.grant(("Write",))
    brain.revoke()
    assert brain.grants == ()
    assert "Write" not in " ".join(brain.config.argv("session", brain.grants))


def test_a_standing_grant_survives_a_one_shot_revoke(brain_factory):
    """The hands stand for the session; a yes to one file write in the middle
    of it is handed back at the end of its turn without taking the hands with
    it. Two slots, one allowlist."""
    brain = brain_factory(replies=["Fine.", "Fine.", "Fine.", "Fine."])
    collect(brain, "hello")
    brain.grant(("Bash(vasper click:*)",), standing=True)
    brain.grant(("Write",))
    argv = " ".join(brain.config.argv("session", brain.grants))
    assert "Bash(vasper click:*)" in argv and "Write" in argv

    brain.revoke()
    assert brain.grants == ("Bash(vasper click:*)",)
    assert brain.standing == ("Bash(vasper click:*)",)
    assert "Write" not in " ".join(brain.config.argv("session", brain.grants))

    brain.revoke_standing()
    assert brain.grants == () and brain.standing == ()
    assert brain.alive, "the brain must come back up after the hands are taken back"


def test_a_background_revoke_finishes_before_the_next_question(brain_factory):
    """The next turn must not race the respawn: if it won, it would run with a
    permission the user granted for something else.

    The earlier version joined the thread before asserting, which removed the
    very race it was written to catch. Deleting the lock from revoke_soon would
    not have failed it."""
    brain = brain_factory(replies=["Fine.", "Fine."])
    collect(brain, "hello")
    brain.grant(("Write",))

    brain.revoke_soon()  # deliberately not joined
    events = collect(brain, "and again")

    assert of(events, TurnComplete), "the turn raced the respawn and died"
    assert brain.grants == (), "the turn ran while a grant was still live"


def test_a_grant_waits_for_a_turn_that_is_already_running(brain_factory):
    """A yes can arrive while the ambient loop is mid-turn, since both share one
    brain. Without the lock, granting tore that turn's process down underneath
    it and the dead process's reader threads went on feeding the new queue.

    The observable invariant is the lock itself, so that is what is asserted.
    Tempting alternatives do not work: driving a real turn and granting halfway
    through passes either way, because `stop()` closes stdin and waits, the fake
    finishes flushing its reply, and the turn completes cleanly. A test that
    cannot fail is worse than no test, so this holds the lock directly and
    checks that `grant()` waits for it.
    """
    brain = brain_factory(replies=["fine"])
    collect(brain, "hello")

    granted = threading.Event()
    holding = threading.Event()

    def hold_the_turn():
        with brain._busy:          # what ask() holds for the length of a turn
            holding.set()
            time.sleep(0.4)

    holder = threading.Thread(target=hold_the_turn)
    holder.start()
    assert holding.wait(5), "never took the lock"

    def do_grant():
        brain.grant(("Write",))
        granted.set()

    granter = threading.Thread(target=do_grant)
    granter.start()

    # Still mid-turn, so the grant must not have gone through yet.
    assert not granted.wait(0.2), "granted while a turn was still running"

    holder.join(timeout=5)
    granter.join(timeout=10)
    assert granted.is_set(), "the grant never completed"
    assert brain.grants == ("Write",)


# --- a login that has died ---------------------------------------------------

AUTH_ERROR = "Failed to authenticate: OAuth session expired and could not be refreshed"


def test_a_dead_login_arrives_as_an_error_result_not_an_answer(brain_factory):
    brain = brain_factory(["Fine now."], FAKE_CLAUDE_AUTH_FAIL=1)
    events = collect(brain, "hello")

    assert of(events, TextDelta) == [], "the CLI sends no deltas for a failed turn"
    done = of(events, TurnComplete)[0]
    assert done.is_error is True
    assert done.text == AUTH_ERROR
    assert done.api_error == "authentication_failed"
    assert done.error_subtype == "error_during_execution"
    assert brain.alive, "a failed turn is not a dead process"


def test_a_restart_picks_up_a_login_renewed_elsewhere(brain_factory):
    brain = brain_factory(["Fine now."], FAKE_CLAUDE_AUTH_FAIL=1)
    assert of(collect(brain, "hello"), TurnComplete)[0].is_error
    first_pid = brain._process.pid

    # The user logged in again in some other terminal. Only a fresh child
    # sees it, which is the whole reason the recovery is a respawn.
    os.environ["FAKE_CLAUDE_AUTH_FAIL"] = "0"
    brain.restart()

    done = of(collect(brain, "hello again"), TurnComplete)[0]
    assert (done.is_error, done.text) == (False, "Fine now.")
    assert brain.alive and brain._process.pid != first_pid


def test_auth_status_reads_the_exit_code(brain_factory):
    assert brain_factory(FAKE_CLAUDE_LOGGED_IN=1).auth_status() is True
    assert brain_factory(FAKE_CLAUDE_LOGGED_IN=0).auth_status() is False


def test_auth_status_is_none_when_the_question_cannot_be_asked():
    brain = ClaudeBrain(BrainConfig(executable="C:/no/such/dir/claude-that-is-not-there.exe"))
    assert brain.auth_status() is None


# --- a brain that is being replaced ------------------------------------------


def test_a_pump_writes_only_to_the_queue_it_was_handed(brain_factory):
    """The pumps used to write into whichever queue existed when they wrote.

    `start()` replaces `self._events` and `self._parser`. The old pump was
    still draining the dying child, and when it reached EOF its `finally` put a
    sentinel into the *new* queue. The next `ask()` read it and answered "the
    brain closed its output stream". Every grant and every revoke respawns, so
    this landed on the turn immediately after a spoken yes.

    The window is real but narrow, so this pins the invariant that closes it
    rather than trying to lose the race on demand: a pump writes to what it was
    handed and never to whatever the brain is holding by the time it writes.
    The queue swap below is exactly what `grant()` does.
    """
    import io

    from vesper.brain.claude import _SENTINEL

    brain = brain_factory(replies=["Hello."])
    collect(brain, "hi")
    dying_events, dying_parser = brain._events, brain._parser

    class DeadChild:
        stdout = io.StringIO("")  # the child has gone; the pipe is at EOF
        stderr = None

    brain.stop()
    brain.start()
    live_events = brain._events
    assert live_events is not dying_events, "start() should hand out a fresh queue"

    # And only now does the pump left over from the old child get scheduled.
    brain._pump_stdout(DeadChild(), dying_parser, dying_events)

    assert dying_events.get_nowait() is _SENTINEL, "the sentinel went nowhere"
    assert live_events.empty(), "the replaced brain wrote into the live brain's queue"


def test_the_turn_after_a_grant_still_completes(brain_factory):
    """The end to end version of the above, over several respawns."""
    brain = brain_factory(replies=["One.", "Two.", "Three.", "Four.", "Five.", "Six."])
    collect(brain, "hello")

    for attempt in range(5):
        brain.grant(("Write",))
        events = collect(brain, "and now")
        assert not of(events, BrainError), (
            f"attempt {attempt}: the replaced brain failed the live one's turn, "
            f"{[e.message for e in of(events, BrainError)]}"
        )
        assert of(events, TurnComplete), f"attempt {attempt}: no turn completed"
        brain.revoke()


def test_a_replaced_brain_cannot_end_the_next_turn_with_its_own_answer(brain_factory):
    """The same race, with the other half of the damage.

    A stale TurnComplete reaching the new queue ends the next turn early and is
    counted again, so one question costs two turns and two lots of cost.

    Counted rather than compared: the fake is a fresh process after the
    respawn, so it starts its reply list over and the text alone cannot tell a
    stale completion from an honest one.
    """
    brain = brain_factory(replies=["First answer.", "Second answer."])
    collect(brain, "hello")
    assert brain.turn_count == 1
    brain.grant(("Write",))

    done = of(collect(brain, "and now"), TurnComplete)
    assert len(done) == 1, f"one question, {len(done)} completions"
    assert brain.turn_count == 2, "a stale completion was counted as a turn of its own"


def test_a_turn_that_times_out_takes_the_child_down_with_it(brain_factory):
    """Giving up on an answer is not the same as the answer stopping.

    Returning on the deadline left the child still generating into the queue
    the next question would read, so the abandoned turn's backlog ended the
    next turn the moment it was asked.
    """
    brain = brain_factory(replies=["Slow one.", "Quick one."], FAKE_CLAUDE_DELAY_MS=400)
    brain.start()
    first_pid = brain._process.pid
    brain.config.turn_timeout_s = 0.25

    timed_out = of(collect(brain, "take your time"), BrainError)
    assert timed_out and "too long" in timed_out[0].message
    assert not brain.alive or brain._process.pid != first_pid, (
        "the child that timed out was left running, and still writing"
    )

    brain.config.turn_timeout_s = 30.0
    events = collect(brain, "now be quick")
    assert len(of(events, TurnComplete)) == 1, "the backlog answered this question"
    assert not of(events, BrainError), "the next turn inherited the abandoned one"


def test_a_timed_out_turn_says_something_a_person_would_say():
    """`ask` produces plumbing. Only one of its messages is fit to be heard."""
    from vesper.brain import failures

    assert failures.spoken_break("that took too long, so I stopped waiting") == (
        "that took too long, so I stopped waiting"
    )
    for internal in (
        "could not reach the brain: [WinError 232] The pipe is being closed",
        "the brain closed its output stream",
        "stdout reader failed: C:/Users/somebody/Vesper/vesper/brain/claude.py",
        "",
    ):
        spoken = failures.spoken_break(internal)
        assert spoken == "I lost my connection to Claude. Give me a moment and ask me again."
        assert "\\" not in spoken and "/" not in spoken, "a path was about to be read aloud"
