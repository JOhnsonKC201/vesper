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
