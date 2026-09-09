"""Rejecting what Whisper invented rather than heard.

Every string in this file is verbatim from a real session log. That matters
more than usual here: hallucinations are not random noise, they are fluent,
confident English, and guessing at what they might look like produces filters
that catch nothing. These are what the model actually produced when handed a
quiet room.

The worst of them woke Vesper and spent a turn, because Whisper echoed its own
initial_prompt back as speech and that text contains the wake word.
"""

import pytest

from vesper.stt.whisper import (
    Listener,
    Transcript,
    WhisperConfig,
    _is_repetitive,
    wake_word_prompt,
)


def listener() -> Listener:
    prompt = wake_word_prompt(("vesper", "jarvis"), "Vesper")
    return Listener(WhisperConfig(initial_prompt=prompt))


def spoken(text: str, **kwargs) -> Transcript:
    base = {"duration_s": 3.0, "avg_logprob": -0.4, "no_speech_prob": 0.2}
    base.update(kwargs)
    return Transcript(text=text, **base)


# --- what it invented -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Talking to an assistant named Vesper, also called Jarvis.",
        "Talking to an assistant named Vesper.",
        "talking to an assistant named vesper, also called jarvis",
    ],
)
def test_the_prompt_coming_back_is_not_speech(text):
    """The one that did real damage. Handed room tone, Whisper returns its own
    initial_prompt, and that text contains the wake word, so Vesper woke up and
    spent a turn on a sentence nobody said."""
    assert listener()._post_gate(spoken(text)) == "prompt-echo"


@pytest.mark.parametrize(
    "text",
    [
        "Choo, choo, choo, choo, choo.",
        "One more, one more, one more, one more, one more.",
        "I don't know if you can hear me, but I don't know if you can hear me.",
        "I'm not a fan of the old laptop, but I'm a fan of the old laptop.",
    ],
)
def test_a_decoder_stuck_in_a_loop_is_not_speech(text):
    assert listener()._post_gate(spoken(text)) == "looping"


def test_the_models_own_doubt_is_actually_used():
    """`no_speech_prob` and `avg_logprob` were being collected on every
    transcription and then never read. The model saying "this probably is not
    speech" is the cheapest signal available and it was going in the bin."""
    assert listener()._post_gate(spoken("some words here", no_speech_prob=0.9)) == "silence"
    assert listener()._post_gate(spoken("some words here", avg_logprob=-2.0)) == "low-confidence"


# --- what it actually heard -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hey Vesper, open new terminal",
        "What's my full name, Vesper?",
        "How much computation do you use while answering me?",
        "How can I make you faster?",
        "yes",
        "sure",
        "Vesper, open the terminal and then run the tests for me please",
    ],
)
def test_real_speech_still_gets_through(text):
    """The filter is worth nothing if it also eats the user. These are all
    things actually said in the same session."""
    assert listener()._post_gate(spoken(text)) == ""


@pytest.mark.parametrize(
    "text",
    ["no no no stop that", "yes yes go ahead", "wait wait", "very very quickly"],
)
def test_ordinary_emphasis_is_not_mistaken_for_a_loop(text):
    """People do repeat themselves. The bar is four of the same word in a row,
    or a whole clause said twice, not any repetition at all."""
    assert not _is_repetitive(text)


def test_a_long_answer_that_reuses_words_is_not_a_loop():
    assert not _is_repetitive(
        "open the terminal and then open the log file and tell me what the "
        "last error in it says"
    )


# --- the end of a video -----------------------------------------------------


def test_the_video_outro_family_is_rejected_at_any_length():
    """Straight out of var/vesper.log, and it was accepted as speech.

        HEARD Until next time, I'll talk to you again soon with Naoli Online.

    Trained on a great deal of video, Whisper answers noise with the end of
    one. Those are whole sentences, so they sail past the `duration_s < 2.0`
    rule that catches "thank you." and "thanks for watching!".
    """
    stt = Listener()
    for invented in (
        "Until next time, I'll talk to you again soon with Naoli Online.",
        "Thanks for watching, and I'll see you in the next video.",
        "Thank you all so much for watching!",
        "Don't forget to like and subscribe.",
        "Please subscribe to my channel.",
        "Subtitles by the Amara.org community",
        "See you next time.",
        "And thanks for watching everyone.",
    ):
        result = Transcript(text=invented, duration_s=6.0, avg_logprob=-0.2)
        assert stt._post_gate(result) == "hallucination", f"let through: {invented!r}"


def test_ordinary_speech_that_shares_words_with_an_outro_gets_through():
    """The list below is the review that nearly did not happen.

    A first version of this filter matched any outro phrase anywhere in the
    utterance. Eleven of these thirteen would have been discarded in silence,
    and a discarded utterance is Vesper ignoring you for no stated reason,
    which this repo has repeatedly decided is the worse failure. The fix was
    to require a sign-off to open the utterance: "Until next time, ..." is one,
    "save that until next time" is a request.
    """
    stt = Listener()
    for genuine in (
        # Every one of these was silently dropped by the first version.
        "Please subscribe me to the newsletter on that page.",
        "Save that until next time.",
        "Leave it until next time then.",
        "Thanks for listening to me ramble.",
        "See you soon.",
        "I will see you soon at the meeting.",
        "Hit the button on the toolbar.",
        "Hit the like button on that post for me.",
        "Remember to subscribe to the mailing list.",
        "Be sure to subscribe me to updates.",
        "Read me the captions from that video.",
        "Transcription from the meeting please.",
        # And the ones that were always fine, kept as a floor.
        "Thanks, that is exactly what I needed.",
        "Subscribe me to the newsletter on that page.",
        "What am I watching on Tuesday?",
        "Thank you for checking, and can you also look at the log?",
        "See you later, I am off to lunch.",
        "Next time remind me to run the tests first.",
        "Can you like the top comment for me?",
        "I will talk to you next time.",
    ):
        result = Transcript(text=genuine, duration_s=3.0, avg_logprob=-0.2)
        assert stt._post_gate(result) == "", f"wrongly dropped: {genuine!r}"


def test_the_outro_filter_cannot_be_made_slow_by_a_long_utterance():
    """It runs on every transcription, so it may not be a backtracking trap."""
    import time as _time

    stt = Listener()
    for hostile in ("thanks " * 4000, "word " * 20000, "a" * 60000):
        started = _time.monotonic()
        stt._post_gate(Transcript(text=hostile, duration_s=9.0, avg_logprob=-0.2))
        assert _time.monotonic() - started < 0.5, "the outro filter backtracked"
