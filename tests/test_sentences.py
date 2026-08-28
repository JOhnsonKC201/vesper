"""The sentence assembler decides when Vesper opens its mouth. Get it wrong and
either it stutters mid-number or it waits for the whole turn before speaking."""

import pytest

from vesper.brain.sentences import SentenceAssembler


def collect(deltas, **kw):
    """Feed deltas one character at a time, the way the CLI actually streams."""
    asm = SentenceAssembler(**kw)
    out = []
    for delta in deltas:
        out.extend(asm.feed(delta))
    tail = asm.flush()
    if tail:
        out.append(tail)
    return out


def stream(text, **kw):
    return collect(list(text), **kw)


def test_splits_two_plain_sentences():
    assert stream("The build passed. Nothing else broke.") == [
        "The build passed.",
        "Nothing else broke.",
    ]


def test_does_not_split_a_decimal_number():
    # The real regression: "You have 598.1 GB free" must stay one utterance.
    assert stream("You have 598.1 GB free on your C drive.") == [
        "You have 598.1 GB free on your C drive."
    ]


def test_does_not_split_on_abbreviation():
    assert stream("Ask Dr. Lee about it. He knows.") == [
        "Ask Dr. Lee about it.",
        "He knows.",
    ]


def test_does_not_split_on_initials():
    assert stream("That commit is from J. Smith. Check it.") == [
        "That commit is from J. Smith.",
        "Check it.",
    ]


def test_does_not_split_on_dotted_acronym():
    assert stream("It runs in the U.S.A. mostly. Fine.") == [
        "It runs in the U.S.A. mostly.",
        "Fine.",
    ]


def test_question_and_exclamation_are_boundaries():
    assert stream("Did it work? It did! Good.") == ["Did it work?", "It did!", "Good."]


def test_closing_quote_stays_with_sentence():
    assert stream('He said "it is done." Then he left.') == [
        'He said "it is done."',
        "Then he left.",
    ]


def test_paragraph_break_is_a_boundary():
    assert stream("First thought\n\nSecond thought") == ["First thought", "Second thought"]


def test_incomplete_sentence_waits_for_flush():
    asm = SentenceAssembler()
    assert list(asm.feed("Still typing")) == []
    assert asm.flush() == "Still typing"


def test_long_run_on_is_force_flushed():
    text = "word " * 200
    chunks = stream(text, max_chars=60)
    assert len(chunks) > 1
    assert all(len(c) <= 70 for c in chunks)
    # Force-flushing must not lose or invent words.
    assert " ".join(chunks).split() == text.split()


def test_never_splits_a_word_when_force_flushing():
    chunks = stream("supercalifragilistic " * 20, max_chars=40)
    for chunk in chunks:
        for word in chunk.split():
            assert word == "supercalifragilistic"


def test_min_chars_prevents_tiny_utterances():
    # "Ok." alone is too short to be worth its own audio clip and its own pause.
    chunks = stream("Ok. That is now fixed.", min_chars=12)
    assert chunks[0].startswith("Ok. That is")


def test_reset_discards_buffer():
    asm = SentenceAssembler()
    list(asm.feed("half a thou"))
    asm.reset()
    assert asm.flush() == ""


def test_no_text_is_lost_across_a_realistic_turn():
    text = (
        "I checked the drive. You have 598.1 GB free, which is fine. "
        "The last build failed on test_auth.py, so I would start there. "
        "Want me to open it?"
    )
    assert " ".join(stream(text)) == text.replace("  ", " ").strip()


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_result_is_independent_of_delta_boundaries(chunk_size):
    """The CLI chunks deltas arbitrarily; output must not depend on that."""
    text = "First one. Second one costs 3.5 GB. Third one?"
    deltas = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    assert collect(deltas) == ["First one.", "Second one costs 3.5 GB.", "Third one?"]
