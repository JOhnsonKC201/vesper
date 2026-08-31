"""What Vesper keeps, and the two ways it could go wrong.

Learning has an asymmetric cost. A missed lesson means saying it again, which
is mildly annoying. A wrong lesson goes into the system prompt on every turn
forever and quietly changes his behaviour in a way that is very hard to trace
back to the sentence that caused it. So most of these tests are about what must
*not* be learned.

The other half is the negation trap. "Don't read me file paths" captures as
"read me file paths" if you strip the prefix and stop thinking, and the stored
lesson then means the exact opposite of what was said. That one has its own
test because it is the kind of bug that reads as correct.
"""

import json
import threading

import pytest

from vesper import learning
from vesper.learning import CORRECTION, EXPLICIT, Lessons, extract, wants_forgetting


# --- what counts as an instruction ------------------------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("Remember that I prefer short answers", "I prefer short answers"),
        ("remember I prefer short answers", "I prefer short answers"),
        ("From now on, keep replies under two sentences",
         "keep replies under two sentences"),
        ("Always confirm before you touch my repos",
         "Always confirm before you touch my repos"),
        ("Never read file paths out loud", "Never read file paths out loud"),
        ("Make sure to check the tests first", "check the tests first"),
        ("I want you to use British spelling", "use British spelling"),
    ],
)
def test_an_explicit_instruction_is_kept_without_its_framing(said, expected):
    """"Remember that X" is a way of addressing him. The rule is X."""
    found = extract(said)
    assert found is not None, f"{said!r} was not recognised as an instruction"
    assert found[0] == expected
    assert found[1] == EXPLICIT


@pytest.mark.parametrize(
    "said",
    [
        "Don't read file paths out loud",
        "do not read file paths out loud",
        "Stop reading file paths out loud",
    ],
)
def test_a_negative_instruction_keeps_its_negation(said):
    """The trap. Strip "don't" and the stored rule means the opposite.

    A lesson that says "read file paths out loud" would then sit in the system
    prompt on every turn, causing the exact behaviour that was complained
    about, and nothing about the stored line would look wrong.
    """
    found = extract(said)
    assert found is not None
    lesson = found[0].lower()
    # "do not read" or "stop reading". Which one depends on how it was said,
    # and the grammar has to survive: "do not reading" is not English, and a
    # system prompt full of broken grammar is one that gets ignored.
    assert lesson.startswith(("do not ", "stop ")), (
        f"the negation was lost: {found[0]!r}"
    )
    assert "file paths" in lesson


def test_the_wake_word_is_not_part_of_the_rule():
    assert extract("Vesper, remember that I prefer short answers")[0] == (
        "I prefer short answers"
    )
    assert extract("hey Vesper always use British spelling")[0] == (
        "always use British spelling"
    )


@pytest.mark.parametrize(
    "said",
    [
        "what is the weather",
        "open a new terminal",
        "yes",
        "no",
        "run the tests",
        "how much disk have I got left",
        "remember",                       # nothing after the trigger
        "remember it",                    # too short to be a rule
        "always?",                        # a question is a request, not a rule
        "",
        "   ",
    ],
)
def test_ordinary_speech_does_not_become_a_permanent_rule(said):
    """The expensive failure. A wrong lesson is invisible and forever."""
    assert extract(said) is None, f"{said!r} would have been learned"


def test_something_too_long_to_be_a_rule_is_not_one():
    """A paragraph in a system prompt is where instructions go to be ignored."""
    assert extract("remember that " + "x" * 400) is None


def test_a_correction_is_recognised_but_marked_weaker():
    found = extract("No, I meant the other repository")
    assert found is not None
    assert found[1] == CORRECTION


# --- the store --------------------------------------------------------------


def test_an_explicit_lesson_counts_immediately(tmp_path):
    store = Lessons(tmp_path / "l.json")
    stored = store.learn("keep replies short", EXPLICIT)

    assert stored is not None
    assert stored.active
    assert "keep replies short" in store.prompt_block()


def test_a_correction_waits_to_be_repeated(tmp_path):
    """A single "no" in a noisy transcript must not become a standing rule."""
    store = Lessons(tmp_path / "l.json")

    assert store.learn("the other repository", CORRECTION) is None
    assert store.prompt_block() == ""

    second = store.learn("the other repository", CORRECTION)
    assert second is not None and second.active
    assert "the other repository" in store.prompt_block()


def test_saying_it_outright_promotes_a_correction(tmp_path):
    store = Lessons(tmp_path / "l.json")
    store.learn("use British spelling", CORRECTION)

    stored = store.learn("use British spelling", EXPLICIT)

    assert stored is not None and stored.kind == EXPLICIT


def test_near_identical_transcriptions_reinforce_rather_than_duplicate(tmp_path):
    """Speech recognition will not produce the same string twice."""
    store = Lessons(tmp_path / "l.json")
    store.learn("please keep the replies short", EXPLICIT)
    store.learn("keep replies short", EXPLICIT)

    assert len(store.items) == 1, [item.text for item in store.items]
    assert store.items[0].times == 2


def test_the_most_repeated_lesson_goes_first(tmp_path):
    """A lesson repeated is a lesson that did not take, so it goes where it is
    least likely to be lost in a long prompt."""
    store = Lessons(tmp_path / "l.json")
    store.learn("first rule", EXPLICIT)
    for _ in range(3):
        store.learn("second rule", EXPLICIT)

    assert store.active()[0].text == "second rule"


def test_only_so_many_reach_the_prompt(tmp_path):
    """The system prompt rides on every single turn, so this is a cost paid
    continuously rather than once."""
    store = Lessons(tmp_path / "l.json", max_in_prompt=3)
    for index in range(10):
        store.learn(f"rule number {index}", EXPLICIT)

    assert len(store.active()) == 3
    assert store.prompt_block().count("\n- ") == 3


def test_lessons_survive_a_restart(tmp_path):
    path = tmp_path / "l.json"
    Lessons(path).learn("use British spelling", EXPLICIT)

    assert "use British spelling" in Lessons(path).prompt_block()


def test_a_broken_file_means_nothing_learned_rather_than_a_crash(tmp_path):
    path = tmp_path / "l.json"
    path.write_text("{ not a list")

    store = Lessons(path)
    assert store.items == []
    assert store.prompt_block() == ""
    # And it recovers: the next lesson writes a valid file.
    store.learn("use British spelling", EXPLICIT)
    assert json.loads(path.read_text())[0]["text"] == "use British spelling"


def test_no_path_means_learning_is_simply_off(tmp_path):
    store = Lessons(None)
    store.learn("use British spelling", EXPLICIT)
    assert store.prompt_block() != "", "in-memory learning should still work"
    # But nothing was written anywhere.
    assert list(tmp_path.iterdir()) == []


# --- forgetting -------------------------------------------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("forget that", "last"),
        ("Vesper, forget that", "last"),
        ("forget what I just said", "last"),
        ("forget everything", "all"),
        ("Vesper forget all of it", "all"),
        ("what did you forget", ""),
        ("open a terminal", ""),
    ],
)
def test_forget_is_recognised(said, expected):
    assert wants_forgetting(said) == expected


def test_forgetting_the_last_one_removes_it_from_the_prompt(tmp_path):
    store = Lessons(tmp_path / "l.json")
    store.learn("keep replies short", EXPLICIT)
    store.learn("use British spelling", EXPLICIT)

    dropped = store.forget_last()

    assert dropped.text == "use British spelling"
    assert "use British spelling" not in store.prompt_block()
    assert "keep replies short" in store.prompt_block()


def test_forgetting_everything_clears_the_file(tmp_path):
    path = tmp_path / "l.json"
    store = Lessons(path)
    store.learn("keep replies short", EXPLICIT)

    assert store.forget_all() == 1
    assert store.prompt_block() == ""
    assert Lessons(path).items == []


# --- how it reaches Claude --------------------------------------------------


def test_the_lessons_go_last_in_the_system_prompt(tmp_path):
    """An instruction at the end of a long prompt survives better than the same
    instruction buried in the middle of a character description."""
    from vesper.brain.persona import build_system_prompt

    store = Lessons(tmp_path / "l.json")
    store.learn("use British spelling", EXPLICIT)

    prompt = build_system_prompt("Johnson", "Be terse.", store.prompt_block())

    assert prompt.rstrip().endswith("- use British spelling")
    assert "Be terse." in prompt


def test_an_empty_prompt_block_changes_nothing(tmp_path):
    from vesper.brain.persona import build_system_prompt

    assert build_system_prompt("Johnson", "Be terse.") == build_system_prompt(
        "Johnson", "Be terse.", ""
    )


# --- in the loop ------------------------------------------------------------


def _conversation(tmp_path, replies=("Right.",)):
    from vesper.audio.speaker import Speaker
    from vesper.conversation import Conversation, ConversationConfig
    from vesper.wake import WakeConfig, WakeGate

    from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI

    voice = FakeVoice()
    speaker = Speaker(voice)
    conversation = Conversation(
        brain=FakeBrain(list(replies)), stt=FakeSTT([]), speaker=speaker,
        mic=FakeMic(), wake=WakeGate(WakeConfig()),
        config=ConversationConfig(greet_on_start=False), ui=RecordingUI(),
    )
    conversation.lessons = Lessons(tmp_path / "l.json")
    return conversation, speaker, voice


def test_an_instruction_is_learned_and_still_answered(tmp_path):
    """"From now on keep answers short" is a rule for later *and* a request
    for right now. Swallowing the turn would be wrong."""
    conversation, speaker, voice = _conversation(tmp_path)
    try:
        conversation.hear("Vesper, from now on keep replies under two sentences")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert "keep replies under two sentences" in conversation.lessons.prompt_block()
    assert "I'll remember that." in voice.lines, "it did not acknowledge"
    assert "Right." in voice.lines, "the turn was swallowed"


def test_a_correction_is_learned_without_announcing_itself(tmp_path):
    """Claiming to have learned something that is not yet in the prompt would
    be a lie that is impossible to notice."""
    conversation, speaker, voice = _conversation(tmp_path)
    try:
        conversation.hear("Vesper, no I meant the other repository")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert "I'll remember that." not in voice.lines
    assert conversation.lessons.items, "nothing was recorded at all"


def test_forgetting_never_reaches_claude(tmp_path):
    """Same reason as mute and undo: the moment you want something forgotten
    is not the moment to depend on a network call."""
    conversation, speaker, voice = _conversation(tmp_path)
    conversation.lessons.learn("use British spelling", EXPLICIT)
    try:
        conversation.hear("Vesper, forget everything")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert conversation.brain.asked == [], "a forget command was sent to Claude"
    assert conversation.lessons.items == []
    assert any("Forgotten" in line for line in voice.lines)


def test_a_correction_stays_silent_even_once_it_takes_effect(tmp_path):
    """The guard that was never actually reached.

    `test_a_correction_is_learned_without_announcing_itself` sends one
    correction, and `learn()` returns None for that, so `_maybe_learn` returns
    at the `stored is None` check before the EXPLICIT comparison is ever
    evaluated. Deleting that comparison left every test green. This says it
    twice, so the lesson becomes active and the guard is the only thing
    keeping him quiet.
    """
    conversation, speaker, voice = _conversation(tmp_path, replies=("Right.", "Right."))
    try:
        conversation.hear("Vesper, no I meant the other repository")
        conversation.hear("Vesper, no I meant the other repository")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert "the other repository" in conversation.lessons.prompt_block(), (
        "saying it twice should have made it stick"
    )
    assert "I'll remember that." not in voice.lines, (
        "a correction announced itself; only instructions given outright do"
    )


def test_a_failing_store_never_costs_a_reply(tmp_path):
    """Learning is a nicety. It must never be the reason a question goes
    unanswered."""
    conversation, speaker, voice = _conversation(tmp_path)

    class Broken(Lessons):
        def learn(self, text, kind=EXPLICIT):
            raise RuntimeError("disk on fire")

    conversation.lessons = Broken(tmp_path / "l.json")
    try:
        conversation.hear("Vesper, always use British spelling")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert "Right." in voice.lines


def test_learning_off_leaves_the_loop_exactly_as_it_was(tmp_path):
    conversation, speaker, voice = _conversation(tmp_path)
    conversation.lessons = None
    try:
        conversation.hear("Vesper, always use British spelling")
        speaker.wait_until_idle(timeout=5)
    finally:
        speaker.close()

    assert "I'll remember that." not in voice.lines
    assert "Right." in voice.lines


def test_the_learned_count_is_a_scalar_in_the_snapshot(tmp_path):
    conversation, speaker, _ = _conversation(tmp_path)
    conversation.lessons.learn("use British spelling", EXPLICIT)
    try:
        status = conversation.status()
    finally:
        speaker.close()

    assert status["learned"] == 1
    assert isinstance(status["learned"], int)


def test_the_acknowledgement_is_one_of_the_prewarmed_phrases():
    """It is said often, so it should be free and instant like the fillers."""
    from vesper.tts.eleven import STOCK_PHRASES

    assert "I'll remember that." in STOCK_PHRASES
