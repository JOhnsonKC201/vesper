"""What Vesper is told after a refusal, and what it claims to be able to do.

Both pinned because of one evening. On 2026-09-05 the refusal note said "do not
attempt it again", Claude read that as forever, and the user asking four more
times for the thing they had just agreed to got "I don't relitigate a refusal"
and "get some sleep". The same prompt told Claude it had a mouse and a keyboard
it does not have, so it promised a kind of help it could not give.
"""

from vesper.brain.persona import (
    DECLINED_NOTE, HANDS_APPROVED_NOTE, HANDS_ASKED_NOTE, build_system_prompt,
)


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
    # The prompt describes what exists. The hands exist now, and the prompt
    # must also say how they ask: refused first, then one spoken yes per turn.
    assert "vasper look" in prompt and "vasper click" in prompt
    assert "mouse and keyboard" in prompt and "in front of" in prompt
    assert "vasper focus" in prompt and "vasper screenshot" in prompt
    # Browsing is the hands in Chrome, read off the page, never a background fetch.
    assert "BROWSING" in prompt and "vasper read" in prompt
    assert "never through WebSearch or WebFetch when they can watch" in prompt


def test_no_dashes_in_the_spoken_wording():
    prompt = build_system_prompt("Johnson", "Be terse.")
    for text in (prompt, DECLINED_NOTE, HANDS_APPROVED_NOTE, HANDS_ASKED_NOTE):
        assert "—" not in text and "–" not in text


def test_the_prompt_says_to_click_by_name_and_that_asked_for_clicks_need_no_question():
    prompt = build_system_prompt("Johnson", "Be terse.")
    assert "Click by name whenever the control has one" in prompt
    assert "already granted" in prompt
    assert "already granted for this turn" in HANDS_ASKED_NOTE
