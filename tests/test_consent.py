"""Asking before doing something big, and doing exactly what was agreed to.

The stakes here are different from the rest of the suite. Everywhere else a bug
means Vesper says something silly; here a bug means it changes a file nobody
told it to change. So these tests are mostly about what must *not* happen:
silence must not approve, an unrelated sentence must not approve, a yes must not
grant more than it was asked about, and no grant may outlive the turn it was
given for.
"""

import time

import pytest

from vesper import audit, config as config_module
from vesper.brain.claude import BrainConfig
from vesper.brain.consent import (
    NO, QUALIFIED, UNCLEAR, YES, ActionRequest, _verbs, hear_answer,
)
from vesper.brain.persona import is_refusal_noise
from vesper.brain.protocol import (
    PermissionNeeded,
    StreamParser,
    ToolStarted,
    TurnComplete,
)
from vesper.conversation import Conversation, ConversationConfig
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI
from vesper.audio.speaker import Speaker

import numpy as np


# --- hearing a yes ----------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    ["yes", "Yes.", "yeah", "yep", "sure", "go ahead", "do it", "okay",
     "Yes please.", "um, yes", "alright", "please do", "go for it"],
)
def test_a_yes_is_heard_as_a_yes(said):
    assert hear_answer(said) == YES


@pytest.mark.parametrize(
    "said",
    ["no", "No.", "nope", "don't", "do not", "stop", "cancel", "leave it",
     "not now", "never mind", "no thanks", "well, no"],
)
def test_a_no_is_heard_as_a_no(said):
    assert hear_answer(said) == NO


@pytest.mark.parametrize(
    "said",
    ["", "   ", "what time is it", "how much disk do I have left",
     "the weather is nice", "open the file"],
)
def test_anything_that_is_not_an_answer_is_unclear(said):
    """The important case. An unclear reply must never read as consent."""
    assert hear_answer(said) == UNCLEAR


@pytest.mark.parametrize(
    "said",
    ["yes but not the config file", "yeah but wait",
     "Sure, do it, but in front of me.", "okay but only this once"],
)
def test_a_qualified_yes_is_neither_a_yes_nor_a_no(said):
    """"Yes but" is a conversation, not permission, so it must not act. It
    used to be treated as a refusal, and on 2026-09-05 the third example was:
    the refusal note then made the thing untouchable for the evening. Now it
    is its own answer, and the conversation asks for a plain one."""
    assert hear_answer(said) == QUALIFIED


def test_a_but_with_a_refusal_in_it_still_refuses():
    assert hear_answer("yes but don't touch the config") == NO
    assert hear_answer("sure but no") == NO


def test_a_but_that_is_not_preceded_by_agreement_is_unclear():
    """"The weather is nice but cold" is not an answer at all."""
    assert hear_answer("the weather is nice but cold") == UNCLEAR


# --- what a yes buys --------------------------------------------------------


def test_a_file_write_is_described_by_its_name_not_its_path():
    request = ActionRequest("Write", {"file_path": "C:/Users/johns/notes.txt"})
    assert request.spoken().startswith("create notes dot txt")
    assert "C:/Users/johns/notes.txt" in request.written()


def test_a_write_says_out_loud_that_the_grant_is_wider_than_the_file():
    """The invariant `approvable` exists for: the question has to cover the grant.

    `grants()` returns bare `Write`, because measured against claude 2.1.259 a
    path scoped `Write(...)` rule is refused even for the path it names while a
    bare `Write` allows writing to a file that was never mentioned. Since the
    grant cannot be narrowed, the question is what has to widen.
    """
    request = ActionRequest("Write", {"file_path": "C:/Users/johns/notes.txt"})
    assert request.whole_tool is True
    assert request.grants() == ("Write",)
    assert "other files" in request.spoken()


def test_a_shell_verb_is_scoped_so_its_question_stays_narrow():
    """The counterpart: Bash *is* scopeable, so it must not gain the warning."""
    request = ActionRequest("Bash", {"command": "git commit -m x"})
    assert request.whole_tool is False
    assert request.grants() == ("Bash(git commit:*)",)
    assert "other files" not in request.spoken()


def test_a_web_fetch_names_the_host_it_is_reaching():
    """The one action that sends something off the machine used to say only
    "a page from the web", which named nothing at all."""
    request = ActionRequest("WebFetch", {"url": "https://example.com/a?token=secret"})
    assert "example dot com" in request.spoken()
    # The audit log keeps every character, including the part not said aloud.
    assert "token=secret" in request.written()


def test_a_shell_command_is_described_by_its_verb():
    request = ActionRequest("Bash", {"command": "git commit -m 'wip'"})
    assert request.spoken() == "run git commit"
    assert request.grants() == ("Bash(git commit:*)",)


def test_a_chained_command_grants_every_verb_in_it_and_no_more():
    """`cd x && git push` is a push. Granting only the first verb would either
    fail immediately or, worse, grant the wrong thing."""
    request = ActionRequest("Bash", {"command": "cd C:/repo && git push origin main"})
    assert request.grants() == ("Bash(cd:*)", "Bash(git push:*)")


def test_a_grant_never_widens_to_bare_bash_for_a_real_command():
    """The whole gate collapses if a yes ever returns bare Bash: with Write
    refused, Claude reaches for `printf > file` and writes anyway."""
    for command in ("rm -rf C:/data", "npm install left-pad", "python -c 'x'"):
        grants = ActionRequest("Bash", {"command": command}).grants()
        assert "Bash" not in grants, command
        assert all(g.startswith("Bash(") for g in grants), command


def test_approving_one_verb_does_not_approve_another():
    commit = ActionRequest("Bash", {"command": "git commit -m x"}).grants()
    assert "Bash(rm:*)" not in commit
    assert "Bash(git push:*)" not in commit


# --- the protocol -----------------------------------------------------------


def _frame(**kwargs):
    import json

    return json.dumps(kwargs)


def test_a_denial_carries_what_was_actually_attempted():
    """The denial frame names the tool but not its input, so the parser has to
    remember the tool call to make the question answerable."""
    parser = StreamParser()
    list(parser.feed(_frame(
        type="assistant",
        message={"content": [{
            "type": "tool_use", "id": "toolu_1", "name": "Write",
            "input": {"file_path": "C:/notes.txt", "content": "hi"},
        }]},
    )))
    events = list(parser.feed(_frame(
        type="system", subtype="permission_denied",
        tool_name="Write", tool_use_id="toolu_1",
        message="Claude requested permissions to write to C:/notes.txt",
    )))
    assert len(events) == 1
    assert isinstance(events[0], PermissionNeeded)
    assert events[0].tool_input["file_path"] == "C:/notes.txt"


def test_the_same_refusal_is_only_reported_once():
    """The result frame repeats every denial. Asking twice about one action
    would train the user to say yes without listening."""
    parser = StreamParser()
    list(parser.feed(_frame(
        type="assistant",
        message={"content": [{
            "type": "tool_use", "id": "toolu_1", "name": "Write",
            "input": {"file_path": "C:/notes.txt"},
        }]},
    )))
    list(parser.feed(_frame(
        type="system", subtype="permission_denied",
        tool_name="Write", tool_use_id="toolu_1",
    )))
    events = list(parser.feed(_frame(
        type="result", is_error=False, result="I could not write that.",
        permission_denials=[{"tool_name": "Write", "tool_use_id": "toolu_1"}],
    )))
    assert [type(e) for e in events] == [TurnComplete]


def test_a_recorded_refusal_from_the_real_cli_still_parses():
    """Recorded from claude 2.1.250 rather than written from memory, so a change
    in the CLI's frame shape fails here instead of quietly leaving Vesper unable
    to ask, and therefore unable to ever act."""
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "turn_with_denial.jsonl"
    parser = StreamParser()
    events = []
    for line in fixture.read_text(encoding="utf-8").splitlines():
        events.extend(parser.feed(line))

    denials = [e for e in events if isinstance(e, PermissionNeeded)]
    assert len(denials) == 1, "one attempt, one question"
    request = ActionRequest(denials[0].tool, denials[0].tool_input)
    assert request.spoken().startswith("create ")
    assert request.grants() == ("Write",)

    # The recorded turn must also show Claude giving up rather than reaching for
    # the shell, which is the behaviour the persona is written to produce.
    tools = [e.name for e in events if type(e).__name__ == "ToolStarted"]
    assert tools == ["Write"], f"tried to route around the refusal: {tools}"


# --- the flags that make the gate real --------------------------------------


def test_the_brain_runs_in_manual_permission_mode():
    """Without this flag the CLI runs in `auto`, where a file write under the
    working directory happens with no announcement at all. Measured against the
    real CLI, not assumed."""
    argv = BrainConfig().argv()
    assert argv[argv.index("--permission-mode") + 1] == "manual"


def test_a_grant_is_passed_to_the_child_and_is_absent_by_default():
    base = BrainConfig(allowed_tools=("Read",))
    assert base.argv()[base.argv().index("--allowedTools") + 1] == "Read"
    granted = base.argv("session-1", ("Bash(git commit:*)",))
    assert granted[granted.index("--allowedTools") + 1] == "Read,Bash(git commit:*)"


def test_the_default_allowlist_cannot_change_anything():
    """A regression guard with teeth. `Bash(python*)` was on this list once, and
    `python -c` writes files, so the gate was open the whole time."""
    allowed = config_module.BrainSettings().allowed_tools
    assert "Bash" not in allowed, "bare Bash allows every command there is"
    forbidden = ("python", "node", "pip", "npm", "powershell -Command", "sh", "bash",
                 "rm", "del", "curl", "wget", "git commit", "git push")
    for spec in allowed:
        for danger in forbidden:
            assert f"Bash({danger}" not in spec, spec


def test_write_tools_are_available_but_not_pre_approved():
    """Vesper must be able to act after a yes, and unable to before one."""
    settings = config_module.BrainSettings()
    assert "Write" in settings.tools and "Edit" in settings.tools
    assert "Write" not in settings.allowed_tools
    assert "Edit" not in settings.allowed_tools


# --- not talking over the question ------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        # All four observed from the real model, in real runs.
        "Waiting on you.",
        "Waiting on you for that one.",
        "That was blocked, not by the approval layer, but by the sandbox's "
        "allowed directories for this session.",
        "I don't have permission to write to that file.",
        "I need your permission first.",
        "The command was refused.",
        "Permission denied.",
    ],
)
def test_a_sentence_that_only_restates_the_refusal_is_dropped(said):
    """The prompt forbids these and the model says them anyway. Told explicitly
    never to say "waiting on you", the very next run said "Waiting on you for
    that one." Hence a local filter rather than a stern instruction."""
    assert is_refusal_noise(said)


@pytest.mark.parametrize(
    "said",
    [
        "I was going to add a line to the todo file.",
        "The disk is at ninety two percent.",
        "Committed, one file changed.",
        "That test was already failing before your change.",
        "I read the config and the port is eight thousand.",
    ],
)
def test_a_sentence_worth_hearing_survives(said):
    assert not is_refusal_noise(said)


def test_only_the_refusal_sentence_is_dropped_not_the_useful_one():
    class ChattyBrain(FakeBrain):
        def ask(self, text):
            self.asked.append(text)
            yield PermissionNeeded(
                "Write", "d", tool_input={"file_path": "C:/notes.txt"},
                tool_use_id="toolu_1",
            )
            yield TurnComplete(
                text="I wanted to add your meeting note to that file. "
                     "I don't have permission though, waiting on you.",
                turns=1,
            )

    conv, speaker, ui, voice = _build(ChattyBrain(), ["Vesper save that"])
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()
    assert voice.lines == [
        "I wanted to add your meeting note to that file.",
        "I want to create notes dot txt, which lets me write to other files too. Do I do this for you?",
    ]


def test_a_turn_with_no_refusal_is_never_filtered():
    """Asking "why was that blocked?" in a later turn must still get an answer.
    The filter only applies to a turn that was itself refused."""
    conv, speaker, ui, voice = _build(
        FakeBrain(["It was blocked because that folder is read only."]),
        ["Vesper why was that blocked"],
    )
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()
    assert voice.lines == ["It was blocked because that folder is read only."]


# --- the conversation -------------------------------------------------------


def _audio(seconds=1.0, level=0.2):
    return np.full(int(16000 * seconds), level, dtype=np.float32)


def _settle(speaker, timeout=5.0):
    assert speaker.wait_until_idle(timeout), "speaker never drained"


class DenyingBrain(FakeBrain):
    """Refuses the first turn's action, then answers normally."""

    def __init__(self, request_tool="Write", request_input=None, replies=None):
        super().__init__(replies or ["I could not do that.", "Done."])
        self.request_tool = request_tool
        self.request_input = request_input or {"file_path": "C:/notes.txt"}
        self.turns_asked = 0

    def ask(self, text):
        self.turns_asked += 1
        if self.turns_asked == 1:
            self.asked.append(text)
            yield PermissionNeeded(
                self.request_tool,
                "detail",
                tool_input=self.request_input,
                tool_use_id="toolu_1",
            )
            yield TurnComplete(text=self.replies.pop(0), turns=1)
            return
        yield from super().ask(text)


def _build(brain, transcripts, *, audit_log=None, window_s=45.0):
    voice = FakeVoice(duration=0.0)
    speaker = Speaker(voice)
    ui = RecordingUI()
    conversation = Conversation(
        brain=brain,
        stt=FakeSTT(transcripts),
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(
            greet_on_start=False, audit_log=audit_log, consent_window_s=window_s
        ),
        ui=ui,
    )
    return conversation, speaker, ui, voice


def test_a_refused_action_becomes_a_spoken_question():
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(brain, ["Vesper save that note"])
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()
    assert conv._pending is not None
    assert voice.lines == [
        "I could not do that.",
        "I want to create notes dot txt, which lets me write to other files too. Do I do this for you?",
    ]
    assert ui.permissions == [("Write", "Write: C:/notes.txt")]


def test_a_spoken_yes_grants_exactly_that_action_and_hands_it_back():
    brain = DenyingBrain(
        request_tool="Bash", request_input={"command": "git commit -m wip"}
    )
    conv, speaker, ui, voice = _build(brain, ["Vesper commit that", "Vesper yes"])
    conv._on_utterance(_audio())   # refused, Vesper asks
    conv._on_utterance(_audio())   # "Vesper, yes"
    _settle(speaker)
    speaker.close()

    assert brain.grant_history == [("Bash(git commit:*)",)]
    assert brain.revoked, "the grant outlived the action it was given for"
    assert brain.grants == ()
    assert conv._pending is None
    assert conv.approvals == 1
    assert ("approved", "Bash: git commit -m wip") in ui.decisions


def test_a_yes_with_a_condition_keeps_the_question_open_and_carries_the_condition():
    """Verbatim from 2026-09-05: the answer to "do I search the web for you"
    was "Sure, do it, but in front of me." It was declined, Claude was told
    never to try again, and four more "do it"s were refused. Now it approves
    nothing, refuses nothing, asks for a plain answer, and the condition
    reaches Claude with the plain yes."""
    brain = DenyingBrain(
        request_tool="Bash", request_input={"command": "git commit -m wip"}
    )
    conv, speaker, ui, voice = _build(
        brain, ["Vesper commit that", "Vesper, sure, do it, but in front of me.", "Vesper yes"]
    )
    conv._on_utterance(_audio())   # refused, Vesper asks
    conv._on_utterance(_audio())   # a yes with a condition
    _settle(speaker)

    assert conv._pending is not None, "the question must stay open"
    assert conv.approvals == 0 and conv.refusals == 0
    assert brain.grant_history == []
    assert not any("Permission refused" in note for note in brain.notes)
    re_ask = voice.lines[-1].lower()
    assert "condition" in re_ask
    # The re-ask must not contain the very words it is asking for, or the echo
    # filter throws the answer away as Vesper hearing itself.
    assert not any(word in re_ask.split() for word in ("yes", "no", "it", "vesper"))

    conv._on_utterance(_audio())   # "Vesper, yes"
    _settle(speaker)
    speaker.close()

    assert conv.approvals == 1
    assert brain.grant_history == [("Bash(git commit:*)",)]
    approved_note = brain.asked[-1]
    assert "Permission granted" in approved_note
    assert "in front of me" in approved_note
    assert conv._pending_condition == ""


def test_a_spoken_no_grants_nothing():
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(brain, ["Vesper save that", "no"])
    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert brain.grant_history == []
    assert conv.refusals == 1
    assert conv._pending is None
    assert any("Permission refused" in note for note in brain.notes)


def test_an_unrelated_reply_lapses_the_request_rather_than_approving_it():
    """Someone who ignores the question and asks something else has not agreed
    to anything, and the request must not sit around waiting for a later yes."""
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(brain, ["Vesper save that", "Vesper what time is it"])
    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert brain.grant_history == []
    assert conv._pending is None
    assert ("ignored", "Write: C:/notes.txt") in ui.decisions
    assert brain.turns_asked == 2, "the new question should still be answered"


def test_a_yes_said_to_someone_else_in_the_room_approves_nothing():
    """The dangerous half of ignoring the wake gate. Someone saying "yes" to a
    colleague, with no wake word and outside the follow-up window, must not
    approve a file write."""
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(
        brain, ["Vesper save that", "yes that's what I told him"]
    )
    conv._on_utterance(_audio())
    conv.wake.disengage()  # the follow-up window has closed
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert brain.grant_history == [], "unaddressed speech approved an action"
    assert conv.approvals == 0
    assert conv._pending is not None, "and the request should still be open"


def test_talking_past_the_question_leaves_it_open():
    """The other half. Speech never addressed to Vesper must not throw away the
    request you were two seconds from approving."""
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(
        brain, ["Vesper save that", "did you see the game last night", "Vesper yes"]
    )
    conv._on_utterance(_audio())
    conv.wake.disengage()
    conv._on_utterance(_audio())   # said to a colleague
    assert conv._pending is not None, "the question was thrown away by room noise"

    conv.wake.engage(time.monotonic())  # addressed again
    conv._on_utterance(_audio())   # "yes"
    _settle(speaker)
    speaker.close()
    assert conv.approvals == 1


def test_a_yes_arriving_too_late_approves_nothing():
    brain = DenyingBrain()
    conv, speaker, ui, voice = _build(brain, ["Vesper save that", "yes"], window_s=0.01)
    conv._on_utterance(_audio())
    time.sleep(0.05)
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert brain.grant_history == [], "a stale yes must not land on an old request"
    assert conv.approvals == 0


def test_typed_input_answers_the_question_too():
    """Text mode used to call respond() directly, which meant typing "yes" was
    sent to Claude as a new question while the request quietly lapsed."""
    brain = DenyingBrain(
        request_tool="Bash", request_input={"command": "git commit -m wip"}
    )
    conv, speaker, ui, voice = _build(brain, [])
    conv.hear("commit that")
    conv.hear("yes")
    _settle(speaker)
    speaker.close()
    assert brain.grant_history == [("Bash(git commit:*)",)]
    assert conv.approvals == 1


def test_every_decision_is_written_down(tmp_path):
    log = tmp_path / "actions.log"
    brain = DenyingBrain()
    conv, speaker, _, _voice = _build(
        brain, ["Vesper save that", "Vesper yes"], audit_log=log
    )
    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    written = log.read_text(encoding="utf-8")
    assert "approved" in written
    assert "C:/notes.txt" in written


def test_the_audit_log_survives_an_unwritable_path(tmp_path):
    """An unwritable log is not a reason to refuse something already approved
    out loud, and definitely not a reason to crash the conversation.

    The earlier version had no assertions at all, so it could not tell a silent
    failure from a silent success, and never checked that logging still worked
    afterwards."""
    bad = tmp_path / "nope" / "\0bad"
    audit.record(bad, audit.APPROVED, "Write: x")

    good = tmp_path / "actions.log"
    audit.record(good, audit.APPROVED, "Write: y")
    assert "Write: y" in good.read_text(encoding="utf-8"), "logging died after a failure"
    assert not list((tmp_path / "nope").glob("*")), "left a half written log behind"


# --- what the reviews found -------------------------------------------------
#
# Every test below started as a real defect found by an adversarial review of
# this file, reproduced against the running code before being fixed. They are
# grouped together because they share one property: each one was a way for the
# question Vesper asks out loud to differ from what would actually happen.


def test_a_command_that_cannot_be_read_is_not_approvable_at_all():
    """`VERSION=$(./build.sh)` produced no verbs, and the empty case fell back
    to a bare `Bash` grant. So the vaguest possible question, "run a command",
    handed over the widest possible permission."""
    for command in ("VERSION=$(./build.sh)", "$(cat /tmp/x)", ""):
        request = ActionRequest("Bash", {"command": command})
        assert "Bash" not in request.grants(), command
        if not request.grants():
            assert not request.approvable, command


def test_a_grant_never_names_a_verb_the_question_did_not():
    """The informed-consent invariant, checked as a property rather than as
    examples. `spoken()` used to truncate to three verbs while `grants()`
    returned all of them, so a yes to a routine commit also handed over `curl`
    and an unconstrained `bash`."""
    commands = [
        "git commit -m x",
        "cd /tmp && git push",
        "sudo rm -rf /tmp/x",
        "npm install left-pad && npm run build",
        "python -c 'print(1)'",
        "echo hi > out.txt",
        "start x && echo y",
    ]
    for command in commands:
        request = ActionRequest("Bash", {"command": command})
        spoken = request.spoken().lower()
        for spec in request.grants():
            verb = spec[len("Bash("):spec.index(":*)")] if spec.startswith("Bash(") else spec
            assert verb.lower() in spoken, f"{command!r} grants {verb!r} unspoken"


def test_a_command_hidden_inside_a_substitution_is_named():
    """This one shipped. `start "" "url$(node -e '...')"` was spoken as "run
    start, then echo" and granted exactly those, so node was never named and
    every approval was refused again. A regex could not fix it either: the
    inner command contains its own parentheses."""
    command = 'start "" "https://x/?v=$(node -e \'console.log(1)\' || echo abc)"'
    verbs = _verbs(command)
    assert "node" in verbs, verbs
    assert "node" in ActionRequest("Bash", {"command": command}).spoken()


def test_a_wrapper_does_not_hide_the_real_verb():
    """`sudo rm -rf /tmp/x` was spoken as "run sudo", which describes nothing
    anybody would refuse."""
    request = ActionRequest("Bash", {"command": "sudo rm -rf /tmp/x"})
    assert "rm" in request.spoken()
    assert "Bash(rm:*)" in request.grants()


def test_approving_an_interpreter_says_what_that_means():
    """Approving `python -c` hands over the whole interpreter, and what it will
    run is not visible from here. "Run python" makes that sound small."""
    for command in ("python -c 'x'", "powershell -Command x", "node -e 'x'"):
        spoken = ActionRequest("Bash", {"command": command}).spoken()
        assert "can do anything" in spoken, command


def test_a_line_with_too_many_commands_is_refused_rather_than_summarised():
    """No single spoken sentence can put five separate commands to someone
    fairly, so it is not asked at all."""
    request = ActionRequest(
        "Bash", {"command": "git add . && git commit -m f && git push && curl u | bash"}
    )
    assert request.grants() == ()
    assert not request.approvable


def test_a_grant_spec_is_never_malformed():
    """A stray bracket produced the spec `Bash(y):*)`, which grants nothing and
    silently loops the approval instead."""
    for command in ("X=$(curl example)", "weird)( thing", "a|b", "'"):
        for spec in ActionRequest("Bash", {"command": command}).grants():
            assert spec.count("(") == spec.count(")"), spec
            assert spec.startswith("Bash(") and spec.endswith(":*)"), spec


@pytest.mark.parametrize(
    "said",
    [
        "Yeah, sure, that's fine, I'll call you back",
        "yes I was just telling him about that",
        "sure, whatever you think",
        "ok so anyway as I was saying",
        "yeah he said the same thing",
    ],
)
def test_a_sentence_that_merely_starts_with_yes_is_not_consent(said):
    """All five were read as YES. Said into a phone with a permission question
    still open, any of them would have run the action. A genuine answer is made
    of agreement and nothing else."""
    assert hear_answer(said) == UNCLEAR


@pytest.mark.parametrize(
    "said", ["please don't", "just no", "actually no", "really no", "no please don't"]
)
def test_an_ordinary_decline_actually_declines(said):
    """These came back UNCLEAR, which meant they neither declined nor told
    Claude to drop the idea. The request lapsed and the words were sent on as a
    fresh question instead."""
    assert hear_answer(said) == NO


def test_nothing_in_the_default_allowlist_can_act_without_asking():
    """A review found six entries that could change things or send data out,
    each one an action nobody would ever be asked about."""
    allowed = config_module.BrainSettings().allowed_tools
    banned = ("find:", "wmic", "nvidia-smi", "git branch", "git remote",
              "powershell", "python", "node", "curl", "wget", "rm", "del")
    for spec in allowed:
        for danger in banned:
            assert f"Bash({danger}" not in spec, spec
    assert "WebSearch" not in allowed, "sends text off the machine with no consent"


# --- what a widened grant actually gets spent on ----------------------------
#
# The CLI will not scope a Write or an Edit to one path. Measured against
# claude 2.1.259: a bare `Write` grant wrote a file that had never been
# mentioned to the user, and every path scoped form tried, through
# `--allowedTools` and through `--settings`, refused even the file it named.
# So a yes to "edit notes dot txt" is a yes to `Edit` until it is taken back.
# The question now says so. These pin the other half: Vesper notices.


def test_a_tool_call_matching_the_approved_one_is_not_flagged():
    from vesper.conversation import Conversation

    request = ActionRequest("Write", {"file_path": "C:/notes.txt"})
    event = ToolStarted("Write", "C:/notes.txt")
    assert Conversation._unasked_use(request, event) is None


def test_the_same_tool_on_a_different_file_is_flagged():
    """The whole point. `Write` was approved for notes.txt and spent on another."""
    from vesper.conversation import Conversation

    request = ActionRequest("Write", {"file_path": "C:/notes.txt"})
    event = ToolStarted("Write", "C:/Users/johns/.ssh/authorized_keys")
    flagged = Conversation._unasked_use(request, event)
    assert flagged == "Write: C:/Users/johns/.ssh/authorized_keys"


def test_reading_is_never_flagged_because_reading_was_never_granted():
    """Read, Grep and Glob run all day without asking. They are not overreach."""
    from vesper.conversation import Conversation

    request = ActionRequest("Write", {"file_path": "C:/notes.txt"})
    for tool in ("Read", "Grep", "Glob"):
        event = ToolStarted(tool, "C:/anything.txt")
        assert Conversation._unasked_use(request, event) is None


def test_nothing_is_flagged_when_no_grant_is_open():
    from vesper.conversation import Conversation

    event = ToolStarted("Write", "C:/anything.txt")
    assert Conversation._unasked_use(None, event) is None


def test_a_blank_permission_mode_falls_back_to_manual_not_to_nothing():
    """A typo in config.yaml must never be able to open the gate.

    `if self.permission_mode:` dropped the flag entirely on a blank value, and
    the session then ran in `auto`, where a Write under cwd goes through with
    nobody asked. Failing toward `manual` is the only safe direction.
    """
    from vesper.brain.claude import BrainConfig

    for blank in ("", "   ", None):
        argv = BrainConfig(permission_mode=blank).argv()
        assert "--permission-mode" in argv
        assert argv[argv.index("--permission-mode") + 1] == "manual"


def test_a_second_shell_command_under_the_same_verb_is_not_flagged():
    """`Bash(git commit:*)` is exactly what the question said out loud.

    A second `git commit` is inside the approval as spoken, so flagging it
    would be noise. The file tools are flagged because their grant is wider
    than their sentence, which is the whole distinction `whole_tool` draws.
    """
    from vesper.conversation import Conversation

    request = ActionRequest("Bash", {"command": "git commit -m first"})
    event = ToolStarted("Bash", "git commit -m second")
    assert Conversation._unasked_use(request, event) is None


def test_a_grant_spent_on_another_file_is_logged_and_said_out_loud(tmp_path):
    """End to end, through a real approved turn.

    The user is asked about `notes.txt` and says yes. The grant is bare `Write`,
    because the CLI will not take a narrower one, so on the approved turn Claude
    also writes `secrets.env`. Nobody was asked about that file, and before this
    nothing anywhere would have said so.
    """

    class WidenedBrain(DenyingBrain):
        def ask(self, text):
            self.turns_asked += 1
            if self.turns_asked == 1:
                yield PermissionNeeded(
                    "Write", "detail",
                    tool_input={"file_path": "C:/notes.txt"},
                    tool_use_id="toolu_1",
                )
                yield TurnComplete(text="I could not do that.", turns=1)
                return
            # The approved turn. The first call is the one that was approved;
            # the second is the one the widened grant also permits.
            yield ToolStarted("Write", "C:/notes.txt")
            yield ToolStarted("Write", "C:/secrets.env")
            yield TurnComplete(text="Done.", turns=2)

    log = tmp_path / "actions.log"
    brain = WidenedBrain()
    conv, speaker, ui, voice = _build(
        brain, ["Vesper save that", "Vesper yes"], audit_log=log
    )
    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert conv.approvals == 1
    assert conv.unasked == 1, "the second file was never put to the user"
    assert ("unasked", "Write: C:/secrets.env") in ui.decisions
    written = log.read_text(encoding="utf-8")
    assert "unasked\tWrite: C:/secrets.env" in written
    assert "secrets.env" not in " ".join(voice.lines), (
        "the spoken line reports that it happened, without reading a path aloud"
    )
    assert any("didn't approve" in line for line in voice.lines)


def test_the_approved_file_itself_is_not_reported_as_unasked(tmp_path):
    """The obvious false positive: the action the user actually said yes to."""

    class ObedientBrain(DenyingBrain):
        def ask(self, text):
            self.turns_asked += 1
            if self.turns_asked == 1:
                yield PermissionNeeded(
                    "Write", "detail",
                    tool_input={"file_path": "C:/notes.txt"},
                    tool_use_id="toolu_1",
                )
                yield TurnComplete(text="I could not do that.", turns=1)
                return
            yield ToolStarted("Write", "C:/notes.txt")
            yield TurnComplete(text="Done.", turns=2)

    brain = ObedientBrain()
    conv, speaker, ui, voice = _build(brain, ["Vesper save that", "Vesper yes"])
    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    _settle(speaker)
    speaker.close()

    assert conv.approvals == 1
    assert conv.unasked == 0
    assert not any("didn't approve" in line for line in voice.lines)
