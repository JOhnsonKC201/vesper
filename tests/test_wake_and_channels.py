"""Wake word gating and the streaming screen/speech splitter."""

import pytest

from vesper.brain.channels import ChannelRouter
from vesper.brain.persona import clean_for_speech, frame_turn, split_channels
from vesper.config import Identity
from vesper.wake import WakeConfig, WakeGate, strip_wake_word

# --- wake word --------------------------------------------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("Vesper what time is it", "what time is it"),
        ("vesper, what time is it", "what time is it"),
        ("Hey Vesper, what time is it", "what time is it"),
        ("Okay Vesper what time is it", "what time is it"),
        ("What time is it, Vesper", "What time is it"),
        ("Jarvis, run the tests", "run the tests"),
    ],
)
def test_wake_word_is_stripped_from_the_request(said, expected):
    remainder, matched = strip_wake_word(said, ("vesper", "jarvis"))
    assert matched
    assert remainder == expected


@pytest.mark.parametrize("misheard", ["Vespa", "Vester", "vespr", "Vesber"])
def test_common_mishearings_still_wake_it(misheard):
    """Whisper produces all of these for a real utterance of the name.

    Missing a wake is recoverable, you just say it again. So the matching is
    generous, but only over strings that are not ordinary English: see the
    note in wake.py about why "whisper" and "jasper" were removed.
    """
    result = WakeGate().check(f"{misheard} what is my battery", now=0.0)
    assert result.triggered is True
    assert result.text == "what is my battery"


def test_speech_without_the_name_is_ignored():
    result = WakeGate().check("I was telling Mark about the deploy", now=0.0)
    assert result.triggered is False
    assert result.reason == "not-addressed"


def test_wake_word_deep_in_a_sentence_does_not_trigger():
    """Otherwise every mention of the word in conversation wakes it."""
    result = WakeGate().check("I think the whole vesper thing is interesting", now=0.0)
    assert result.triggered is False


def test_follow_up_needs_no_wake_word():
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.engage(now=100.0)
    result = gate.check("and what about memory", now=105.0)
    assert result.triggered is True
    assert result.reason == "follow-up"


def test_follow_up_window_expires():
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.engage(now=100.0)
    assert gate.check("and what about memory", now=140.0).triggered is False


def test_disengage_closes_the_window_immediately():
    gate = WakeGate()
    gate.engage(now=100.0)
    gate.disengage()
    assert gate.check("still there", now=101.0).triggered is False


def test_a_hold_cannot_be_shortened_by_a_later_engage():
    """A question holds the window open for as long as the question is live.
    The re-engage that fires when his voice stops must not pull it back in.

    Mutation that fails this: make `engage` assign instead of taking the max.
    """
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.hold_open(until=145.0)
    gate.engage(now=100.0)
    assert gate.engaged(140.0), "the hold was cut short by an ordinary engage"

    gate.disengage()
    assert not gate.engaged(101.0), "disengage must still close everything"


def test_engage_never_moves_the_window_backwards():
    gate = WakeGate(WakeConfig(follow_up_window_s=25))
    gate.engage(now=100.0)
    gate.engage(now=90.0)
    assert gate.engaged(124.0)


def test_open_mic_mode_accepts_everything():
    gate = WakeGate(WakeConfig(require_wake_word=False))
    assert gate.check("no name needed", now=0.0).triggered is True


def test_bare_name_yields_empty_request():
    result = WakeGate().check("Vesper", now=0.0)
    assert result.triggered is True
    assert result.text.strip().lower() in ("", "vesper")


def test_empty_transcript_never_triggers():
    assert WakeGate().check("", now=0.0).triggered is False


# --- streaming channel router -----------------------------------------------


def route(deltas):
    router = ChannelRouter()
    spoken, screen = [], []
    for delta in deltas:
        s, c = router.feed(delta)
        spoken.append(s)
        screen.append(c)
    s, c = router.flush()
    spoken.append(s)
    screen.append(c)
    return "".join(spoken), "".join(screen)


def test_plain_text_is_all_spoken():
    spoken, screen = route(["Hello ", "there."])
    assert spoken == "Hello there."
    assert screen == ""


def test_screen_block_is_removed_from_speech():
    spoken, screen = route(["Line forty. <screen>main.py:40</screen> Want it open?"])
    assert "main.py" not in spoken
    assert screen == "main.py:40"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8, 13, 40])
def test_tags_split_across_deltas_still_route_correctly(chunk_size):
    """The real condition: tags arrive one character at a time."""
    text = "Found it. <screen>path/to/file.py:12</screen> Shall I open it?"
    deltas = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    spoken, screen = route(deltas)
    assert screen == "path/to/file.py:12"
    assert "<screen>" not in spoken
    assert "path/to" not in spoken
    assert "Found it." in spoken and "Shall I open it?" in spoken


def test_text_that_merely_looks_like_a_tag_is_still_spoken():
    spoken, screen = route(["I looked at the ", "<scr", "ipt> tag"])
    assert spoken == "I looked at the <script> tag"
    assert screen == ""


def test_unclosed_screen_block_flushes_to_screen_not_speech():
    """A truncated reply must not dump a file path into the speakers."""
    spoken, screen = route(["Here it is. <screen>secret/path"])
    assert "secret/path" not in spoken
    assert screen == "secret/path"


def test_multiple_screen_blocks():
    spoken, screen = route(["One <screen>A</screen> two <screen>B</screen> three"])
    assert screen == "AB"
    assert spoken.split() == ["One", "two", "three"]


# --- persona helpers --------------------------------------------------------


def test_split_channels_on_a_finished_reply():
    spoken, screen = split_channels("It fails. <screen>main.py:40</screen> Open it?")
    assert spoken == "It fails. Open it?"
    assert screen == "main.py:40"


def test_clean_for_speech_strips_markup():
    messy = "**Bold** and `code` and\n- a bullet\n1. a number\n## heading"
    cleaned = clean_for_speech(messy)
    for symbol in ("**", "`", "- ", "1.", "#"):
        assert symbol not in cleaned


def test_clean_for_speech_removes_dashes_the_user_dislikes():
    cleaned = clean_for_speech("This happened, then that")
    assert "—" not in cleaned
    assert "–" not in clean_for_speech("a – b")


def test_clean_for_speech_drops_code_fences_entirely():
    cleaned = clean_for_speech("Try this:\n```python\nprint('hi')\n```\nThat works.")
    assert "print" not in cleaned
    assert "That works." in cleaned


def test_frame_turn_labels_context_as_not_from_the_user():
    framed = frame_turn("what am I looking at", "focused window: Code.exe")
    assert "machine context" in framed
    assert framed.strip().endswith("what am I looking at")


def test_frame_turn_without_context_is_just_the_utterance():
    assert frame_turn("hello", "") == "hello"


# --- fuzzy wake matching ----------------------------------------------------
#
# Added after the voice smoke test caught Whisper rendering "Vesper," as "But"
# and as "best but". The homophone list cannot enumerate a name split across two
# tokens, so the head of the utterance is also compared fuzzily.


@pytest.mark.parametrize(
    "misheard",
    ["bestper", "vesber", "vespur", "wesper", "vesperr", "vespera"],
)
def test_fuzzy_matching_catches_mishearings_not_in_the_list(misheard):
    result = WakeGate().check(f"{misheard} check the disk", now=0.0)
    assert result.triggered is True
    assert result.text == "check the disk"


@pytest.mark.parametrize(
    "innocent",
    [
        "the best butter is expensive",
        "what is the weather today",
        "I told him no and left",
        "testing the microphone now",
        "can you check that for me",
        "vesting schedules are confusing",
        "whisper it to me quietly",
        "let us prosper this year",
        "the desperate need for coffee",
    ],
)
def test_fuzzy_matching_does_not_wake_on_ordinary_speech(innocent):
    """A false wake is annoying; a chain of them is why people uninstall."""
    assert WakeGate().check(innocent, now=0.0).triggered is False


def test_fuzzy_match_reports_what_it_matched():
    result = WakeGate().check("vesber what time is it", now=0.0)
    assert result.matched
    assert result.reason == "wake-word"


# --- whisper biasing prompt -------------------------------------------------


def test_wake_word_prompt_names_every_wake_word():
    from vesper.stt.whisper import wake_word_prompt

    prompt = wake_word_prompt(("vesper", "jarvis"), "Vesper")
    assert "Vesper" in prompt and "Jarvis" in prompt


def test_wake_word_prompt_handles_a_single_word():
    from vesper.stt.whisper import wake_word_prompt

    prompt = wake_word_prompt(("vesper",), "Vesper")
    assert prompt == "Talking to an assistant named Vesper."


def test_wake_word_prompt_does_not_repeat_the_name():
    from vesper.stt.whisper import wake_word_prompt

    assert wake_word_prompt(("vesper", "jarvis"), "Vesper").count("Vesper") == 1


def test_wake_word_prompt_is_empty_without_words():
    from vesper.stt.whisper import wake_word_prompt

    assert wake_word_prompt((), "") == ""


# --- the name said out loud -------------------------------------------------


def test_the_shipped_wake_words_are_the_ones_that_are_actually_said():
    """Pinned so the tests below cannot drift from what a new install gets."""
    assert Identity().wake_words == ("vasper", "vesper", "jarvis")
    assert Identity().name == "Vasper"


def test_hey_vasper_wakes_it_wherever_it_falls_in_the_sentence():
    """The name is Vasper, and it was deaf to it at the end of an utterance.

    The fuzzy pass scores "vasper" against "vesper" at 0.83 and so caught it at
    the head, which hid the problem. The tail is matched against the homophone
    list only, and "vasper" was not on it, so "what time is it, Vasper" was
    deaf while "what time is it, Vespa" woke.
    """
    words = Identity().wake_words
    for said in (
        "hey Vasper what is my battery",
        "Vasper what time is it",
        "what time is it Vasper",
        "are you there Vasper",
    ):
        gate = WakeGate(WakeConfig(words=words))
        assert gate.check(said, 0.0).triggered, said


def test_the_old_spelling_still_answers():
    """Whisper was biased toward "Vesper" for months and still produces it.

    Dropping it would mean the assistant going deaf on its own transcriptions
    during the changeover, for no gain: neither spelling is ordinary English.
    It is carried as its own wake word rather than as a vasper homophone,
    because unlike vasper it is safe to match approximately.
    """
    words = Identity().wake_words
    for said in ("Vesper are you there", "hey Vesper what is my battery", "Vespa what time is it"):
        gate = WakeGate(WakeConfig(words=words))
        assert gate.check(said, 0.0).triggered, said


def test_the_wider_list_still_refuses_ordinary_english():
    """The reason "whisper" and "jasper" came off the list in the first place."""
    words = Identity().wake_words
    for said in (
        "whisper it to me quietly",
        "best for now",
        "that was a vast improvement",
        "I will pass for now",
        "the vase broke",
    ):
        gate = WakeGate(WakeConfig(words=words))
        assert not gate.check(said, 0.0).triggered, said


def test_whisper_is_biased_toward_the_name_that_is_actually_said():
    from vesper.stt.whisper import wake_word_prompt

    assert wake_word_prompt(("vasper", "jarvis"), "Vasper") == (
        "Talking to an assistant named Vasper, also called Jarvis."
    )


def test_it_introduces_itself_by_the_name_it_answers_to():
    """It said "Vesper here" while answering to Vasper, which is just wrong."""
    from vesper.conversation import ConversationConfig

    assert ConversationConfig().name == "Vasper"
