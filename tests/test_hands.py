"""The hands, tested without a desktop.

Everything here is pure: key spelling, text escaping, the cursor path, the
element list and how a name is resolved against it. The parts that touch the
screen are exercised live, with a person watching, which is the only honest
test for them.
"""

import pytest

from vesper import hands
from vesper.hands import Element, escape_text, find_element, glide_path, to_send_keys


# --- keys ---------------------------------------------------------------------


def test_literal_text_cannot_turn_into_a_menu_chord():
    """send_keys reads % as alt. Typing "50% off" unescaped opens a menu."""
    assert escape_text("50% off") == "50{%} off"
    assert escape_text("a^b+c") == "a{^}b{+}c"
    assert escape_text("plain words") == "plain words"


@pytest.mark.parametrize("combo, sent", [
    ("ctrl+s", "^s"),
    ("ctrl+shift+n", "^+n"),
    ("alt+f4", "%{F4}"),
    ("enter", "{ENTER}"),
    ("ctrl+l", "^l"),
    ("ctrl l", "^l"),
    ("Ctrl+Enter", "^{ENTER}"),
    ("ctrl+t", "^t"),
    # The Windows key is held, not tapped. `{VK_LWIN}d` taps Win, which opens
    # Start, then types d into its search box; that is what "extend my screen"
    # produced on 2026-09-07 (`vasper key win+p`).
    ("win+d", "{VK_LWIN down}d{VK_LWIN up}"),
    ("win+p", "{VK_LWIN down}p{VK_LWIN up}"),
    ("win+shift+s", "{VK_LWIN down}+s{VK_LWIN up}"),
    ("windows+e", "{VK_LWIN down}e{VK_LWIN up}"),
    # On its own it is a tap, and a tap of Win is how you open Start.
    ("win", "{VK_LWIN}"),
])
def test_hotkeys_become_send_keys_syntax(combo, sent):
    assert to_send_keys(combo) == sent


def test_an_unknown_key_raises_rather_than_guessing():
    """A mistyped hotkey lands somewhere in a window nobody is looking at."""
    with pytest.raises(ValueError):
        to_send_keys("ctrl+frobnicate")
    with pytest.raises(ValueError):
        to_send_keys("hyper+s")
    with pytest.raises(ValueError):
        to_send_keys("")


# --- the cursor path -----------------------------------------------------------


def test_the_glide_ends_exactly_on_the_target():
    path = glide_path((0, 0), (1003, 517))
    assert path[-1] == (1003, 517)
    assert len(path) == round(hands.GLIDE_MS / hands.GLIDE_STEP_MS)


def test_the_glide_moves_monotonically_and_visibly():
    """Eased at both ends, never doubling back: a hand, not a machine."""
    path = glide_path((100, 100), (700, 400))
    xs = [x for x, _ in path]
    ys = [y for _, y in path]
    assert xs == sorted(xs) and ys == sorted(ys)
    assert len(set(path)) > 10, "a glide with a handful of points is a jump"
    # Slow out of the start: the first step is much shorter than a middle one.
    first = hands.distance((100, 100), path[0])
    middle = hands.distance(path[len(path) // 2 - 1], path[len(path) // 2])
    assert first < middle


def test_a_zero_length_glide_is_one_point():
    assert glide_path((5, 5), (5, 5)) == [(5, 5)]


# --- the element list ------------------------------------------------------------


def element(eid=0, name="Save", control_type="Button", box=(0, 0, 10, 10), **kw):
    left, top, right, bottom = box
    return Element(eid=eid, name=name, control_type=control_type,
                   left=left, top=top, right=right, bottom=bottom, **kw)


def test_an_element_describes_itself_with_its_centre():
    line = element(3, "Save", "Button", (800, 630, 824, 650)).describe()
    assert line == "[3] Button 'Save' at 812,640"


def test_a_disabled_control_and_a_value_are_named():
    line = element(1, "Search", "Edit", (0, 0, 10, 10), enabled=False, value="cats").describe()
    assert "(disabled)" in line and "value='cats'" in line


def test_an_exact_name_wins():
    found, others = find_element([element(0, "Save"), element(1, "Save As")], "save")
    assert found.eid == 0 and others == []


def test_two_exact_matches_is_an_ambiguity_not_a_coin_toss():
    found, others = find_element([element(0, "Delete"), element(1, "Delete")], "Delete")
    assert found is None
    assert [e.eid for e in others] == [0, 1]


def test_an_ambiguous_name_is_reported_in_words_claude_can_relay():
    """The old line told Claude to "click one by its X Y", which is how a tie
    gets broken by a guess. The line now says to ask, and names the choices in
    a form that can be read out."""
    others = [
        element(3, "Delete", "Button", (800, 630, 824, 650)),
        element(7, "Delete", "MenuItem", (30, 110, 50, 130)),
    ]
    assert hands.ambiguity_line("Delete", "Explorer", others) == (
        "'Delete' matches 2 controls in 'Explorer': [3] Button 'Delete' at 812,640; "
        "[7] MenuItem 'Delete' at 40,120. Ask which one."
    )


def test_a_substring_is_used_when_it_is_the_only_candidate():
    found, _ = find_element([element(0, "Don't save"), element(1, "Cancel")], "don't")
    assert found.eid == 0


def test_a_clearly_closer_substring_wins_over_a_longer_one():
    found, others = find_element(
        [element(0, "Save As Template"), element(1, "Save As")], "as"
    )
    assert found.eid == 1 and [e.eid for e in others] == [0]


def test_an_unknown_name_finds_nothing():
    assert find_element([element(0, "Save")], "quit") == (None, [])
    assert find_element([element(0, "Save")], "   ") == (None, [])


class _Node:
    def __init__(self, text):
        self._text = text

    def window_text(self):
        return self._text


class _Window:
    def __init__(self, texts):
        self.texts = texts

    def descendants(self, control_type=None):
        assert control_type == "Text"
        return [_Node(t) for t in self.texts]


def test_page_text_reads_the_text_controls_in_order_and_tidies_whitespace():
    window = _Window(["  Weather  in\n Baltimore ", "", "  Tomorrow: 71 F, sunny  "])
    assert hands.page_text(window) == "Weather in Baltimore\nTomorrow: 71 F, sunny"


def test_page_text_is_capped_and_says_so():
    window = _Window(["x" * 50] * 100)
    text = hands.page_text(window, max_chars=200)
    assert text.endswith("...")
    assert len(text) < 400


def test_page_text_survives_a_window_that_cannot_be_read():
    class Broken:
        def descendants(self, control_type=None):
            raise RuntimeError("gone")

    assert hands.page_text(Broken()) == ""


def test_the_walk_caps_are_real_numbers_not_infinity():
    assert 0 < hands.MAX_ELEMENTS <= 200
    assert 0 < hands.MAX_DEPTH <= 20
