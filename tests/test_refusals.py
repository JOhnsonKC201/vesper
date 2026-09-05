"""What Vesper is told after a refusal, and what it claims to be able to do.

Both pinned because of one evening. On 2026-09-05 the refusal note said "do not
attempt it again", Claude read that as forever, and the user asking four more
times for the thing they had just agreed to got "I don't relitigate a refusal"
and "get some sleep". The same prompt told Claude it had a mouse and a keyboard
it does not have, so it promised a kind of help it could not give.
"""

from vesper.brain.persona import DECLINED_NOTE, build_system_prompt


def test_a_refusal_is_not_forever():
    note = DECLINED_NOTE.format(action="WebSearch")
    assert "on your own" in note
    assert "asks for the same thing again" in note
    assert "attempt it again" in note
    # The half that must stay: no going around the decision unasked.
    assert "another way around it" in note


def test_the_prompt_says_a_refusal_can_be_asked_again():
    prompt = build_system_prompt("Johnson", "Be terse.")
    assert "A refusal is not permanent" in prompt
    assert "relitigate" in prompt


def test_the_prompt_does_not_claim_a_mouse_or_keyboard():
    prompt = build_system_prompt("Johnson", "Be terse.")
    assert "do NOT have a mouse or a keyboard yet" in prompt
    assert "You have a mouse and a keyboard" not in prompt
    # What it can still do is stated, so the honest answer has content.
    assert "vasper focus" in prompt and "vasper screenshot" in prompt


def test_no_dashes_in_the_spoken_wording():
    prompt = build_system_prompt("Johnson", "Be terse.")
    for text in (prompt, DECLINED_NOTE):
        assert "—" not in text and "–" not in text
