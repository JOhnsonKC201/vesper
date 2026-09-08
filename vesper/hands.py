"""The parts of Vasper's hands that can be reasoned about without a screen.

Vesper is the user's own voice assistant on the user's own machine. Every action
in this module runs only after the user has said yes out loud to "use the mouse
and keyboard", once per session or once per turn, and every action is performed
in front of them:
the cursor travels visibly to where it will click, and text is typed one
character at a time rather than pasted. The point is that a person watching can
see what is about to happen and say stop.

`tool.py` is the command line the brain calls. This module holds what those
commands are made of: how a hotkey is spelled for the keyboard driver, how
literal text is kept from turning into a menu chord, the path the cursor takes,
and the walk over a window's accessibility tree that turns a screen into a
numbered list a model can read.

Pure functions first, so they are tested without a desktop. The functions that
touch UI Automation import pywinauto only when called, because this module is
imported by the assistant itself and the assistant never looks at a window.

Ported from `research/hands`, where the same verbs solved two of two live
Notepad tasks. Left behind on purpose: the destructive-word guard and the JSONL
trace, because in Vesper the guard is the spoken consent gate and the record is
the actions log.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

# pywinauto's send_keys treats these as syntax. Typing a literal one means
# wrapping it in braces, and forgetting to is how "50% off" becomes an alt
# chord that opens a menu.
_SEND_KEYS_SPECIAL = "^%+~(){}[]"

# Modifiers the way a person or a model spells them, mapped to send_keys. These
# three are prefixes: send_keys holds them for the key that follows.
_MODIFIERS = {"ctrl": "^", "control": "^", "alt": "%", "shift": "+"}

# The Windows key has no prefix form. `{VK_LWIN}` is a tap, and a tap of Win
# opens Start, so `{VK_LWIN}p` opened Start and typed p into its search box:
# that is what "extend my screen" would have done on 2026-09-07 had the click
# been approved. A held key is spelled `{VK_LWIN down}...{VK_LWIN up}`.
_HELD = {"win": "VK_LWIN", "windows": "VK_LWIN", "super": "VK_LWIN"}

# Keys that need braces. Single printable characters do not.
_NAMED_KEYS = {
    "enter": "{ENTER}", "return": "{ENTER}", "tab": "{TAB}", "esc": "{ESC}",
    "escape": "{ESC}", "space": "{SPACE}", "backspace": "{BACKSPACE}",
    "delete": "{DELETE}", "del": "{DELETE}", "home": "{HOME}", "end": "{END}",
    "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
    "pgup": "{PGUP}", "pgdn": "{PGDN}", "pageup": "{PGUP}", "pagedown": "{PGDN}",
    "insert": "{INSERT}", "win": "{VK_LWIN}", "windows": "{VK_LWIN}",
    **{f"f{n}": "{F%d}" % n for n in range(1, 13)},
}

# Control types worth a line in the list even when they carry no name. A
# nameless Edit box is where you type; a nameless Pane is scenery.
INTERACTIVE = frozenset({
    "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "ListItem",
    "MenuItem", "RadioButton", "Slider", "SplitButton", "TabItem", "Text",
    "TreeItem", "Document", "DataItem", "Spinner", "ToolBar", "Image",
})

# Caps on the tree walk. A full Chrome tree is tens of thousands of nodes and
# walking it takes minutes; a look that takes longer than the task is a hang.
# Breadth first, so what survives the cap is the top of the tree, where the
# toolbar and the address bar live, not the deepest corner of the first panel.
MAX_ELEMENTS = 120
MAX_DEPTH = 12

# How the cursor travels. Long enough to be seen, short enough not to feel
# like waiting. The person watching sees where the click is going before it
# lands, which is the whole reason the cursor does not simply jump.
GLIDE_MS = 350
GLIDE_STEP_MS = 16
# Seconds between typed characters. Visible typing, not a paste.
TYPE_PAUSE_S = 0.03


def escape_text(text: str) -> str:
    """Make arbitrary text safe to hand to send_keys."""
    return "".join("{%s}" % ch if ch in _SEND_KEYS_SPECIAL else ch for ch in text)


def to_send_keys(combo: str) -> str:
    """"ctrl+shift+n" into send_keys syntax.

    Raises on an unrecognised key rather than guessing. A silently mistyped
    hotkey lands somewhere unpredictable, in a window the person may not be
    looking at.
    """
    parts = [part.strip().lower() for part in combo.replace(" ", "+").split("+") if part.strip()]
    if not parts:
        raise ValueError("empty key")
    prefix = ""
    held: list[str] = []
    for part in parts[:-1]:
        if part in _MODIFIERS:
            prefix += _MODIFIERS[part]
        elif part in _HELD:
            held.append(_HELD[part])
        else:
            raise ValueError(f"{part!r} is not a modifier in {combo!r}")
    final = parts[-1]
    if final in _NAMED_KEYS:
        inner = prefix + _NAMED_KEYS[final]
    elif len(final) == 1:
        inner = prefix + escape_text(final)
    else:
        raise ValueError(f"{final!r} is not a key this understands, in {combo!r}")
    for name in reversed(held):
        inner = "{%s down}%s{%s up}" % (name, inner, name)
    return inner


def ambiguity_line(name: str, title: str, others: list["Element"]) -> str:
    """What the brain reads when a name matches more than one control.

    It used to say "click one by its X Y", which is how a tie gets broken by a
    guess. Now it names the choices in a form that can be read out, and says
    to ask. The persona's WHEN YOU ARE NOT SURE section is the other half.
    """
    listing = "; ".join(e.describe() for e in others)
    return (
        f"{name!r} matches {len(others)} controls in {title!r}: {listing}. "
        "Ask which one."
    )


def glide_path(start: tuple[int, int], end: tuple[int, int],
               duration_ms: int = GLIDE_MS, step_ms: int = GLIDE_STEP_MS) -> list[tuple[int, int]]:
    """The cursor positions between here and there, eased at both ends.

    Always ends exactly on `end`, whatever the rounding did on the way, because
    the click that follows uses the final position and a pixel short of a small
    button is a miss. A zero-length move is a single point.
    """
    steps = max(1, int(round(duration_ms / max(1, step_ms))))
    (x0, y0), (x1, y1) = start, end
    if (x0, y0) == (x1, y1):
        return [end]
    points: list[tuple[int, int]] = []
    for index in range(1, steps + 1):
        t = index / steps
        # Smoothstep: slow out of the start, slow into the target, the way a
        # hand moves. A linear glide reads as a machine.
        eased = t * t * (3 - 2 * t)
        points.append((int(round(x0 + (x1 - x0) * eased)), int(round(y0 + (y1 - y0) * eased))))
    points[-1] = end
    return points


@dataclass(frozen=True)
class Element:
    """One thing on screen the brain could act on, numbered for this look only.

    The number is per look, not a handle. Handles are unstable across looks and
    would let a plan reference something that has since been repainted; a
    number that only means something for one list makes a stale reference an
    obvious error rather than a silent misclick.
    """

    eid: int
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int
    enabled: bool | None = None
    value: str | None = None

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def describe(self) -> str:
        """The single line the brain reads: `[3] Button 'Save' at 812,640`."""
        bits = [f"[{self.eid}]", self.control_type]
        if self.name:
            bits.append(repr(self.name[:60]))
        if self.value:
            bits.append(f"value={self.value[:40]!r}")
        if self.enabled is False:
            bits.append("(disabled)")
        x, y = self.center
        bits.append(f"at {x},{y}")
        return " ".join(bits)


def find_element(elements: list[Element], name: str) -> tuple[Element | None, list[Element]]:
    """The element a spoken or typed name means, plus the runners up.

    Exact name first, then a name that contains it, interactive types before
    plain text. Two exact matches is an ambiguity, not a coin toss: the caller
    gets None and the candidates, and says so, because clicking the wrong
    "Delete" is the failure this whole file is careful about.
    """
    needle = name.strip().lower()
    if not needle:
        return None, []
    exact = [e for e in elements if e.name.lower() == needle]
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, exact[:6]
    partial = [e for e in elements if needle in e.name.lower()]
    partial.sort(key=lambda e: (e.control_type == "Text", len(e.name)))
    if len(partial) == 1:
        return partial[0], []
    if len(partial) > 1 and len(partial[0].name) < len(partial[1].name):
        # One clearly closer match: "Save" against "Save" and "Save As".
        return partial[0], partial[1:6]
    return None, partial[:6]


# --- the accessibility tree -------------------------------------------------


def _rect_of(control) -> tuple[int, int, int, int] | None:
    try:
        box = control.rectangle()
    except Exception:
        return None
    left, top, right, bottom = int(box.left), int(box.top), int(box.right), int(box.bottom)
    # Zero-area and off-screen controls are in the tree constantly: collapsed
    # panels, virtualised list rows, the undrawn parts of a scrolled document.
    if right - left <= 0 or bottom - top <= 0 or right < 0 or bottom < 0:
        return None
    return left, top, right, bottom


def _text_of(control) -> str:
    try:
        text = control.window_text()
        if text:
            return str(text).strip()
    except Exception:
        pass
    try:
        return str(getattr(control.element_info, "name", "") or "").strip()
    except Exception:
        return ""


def _type_of(control) -> str:
    try:
        return str(control.element_info.control_type or "Unknown")
    except Exception:
        return "Unknown"


def _enabled(control) -> bool | None:
    try:
        return bool(control.is_enabled())
    except Exception:
        return None


def _value_of(control) -> str | None:
    """The contents of an edit box, which no screenshot can give reliably."""
    try:
        if hasattr(control, "get_value"):
            text = control.get_value()
            return str(text) if text else None
    except Exception:
        pass
    return None


def walk(window, *, max_elements: int = MAX_ELEMENTS, max_depth: int = MAX_DEPTH) -> list[Element]:
    """Breadth first over a window's control tree, under both caps."""
    elements: list[Element] = []
    queue: deque = deque([(window, 0)])
    while queue and len(elements) < max_elements:
        control, depth = queue.popleft()
        if depth >= max_depth:
            continue
        try:
            children = control.children()
        except Exception:
            children = []
        for child in children:
            control_type = _type_of(child)
            rect = _rect_of(child)
            if rect is None:
                # Still descend: a zero-area Pane routinely holds real controls.
                queue.append((child, depth + 1))
                continue
            name = _text_of(child)
            if control_type in INTERACTIVE and (name or control_type in ("Edit", "Document")):
                elements.append(Element(
                    eid=len(elements), name=name, control_type=control_type,
                    left=rect[0], top=rect[1], right=rect[2], bottom=rect[3],
                    enabled=_enabled(child), value=_value_of(child),
                ))
                if len(elements) >= max_elements:
                    break
            queue.append((child, depth + 1))
    return elements


def page_text(window, *, max_chars: int = 6000, max_nodes: int = 800) -> str:
    """The readable text of a window, in tree order.

    Measured on a Chrome page 2026-09-05: the document's UIA text pattern
    returned 152 characters, the cookie banner and nothing else, while the
    147 `Text` controls under it held the whole page, 6,965 characters, in
    0.08 s. So this reads the text controls and ignores the text pattern.
    Capped in both directions, because a long page is thousands of nodes and
    the brain does not need all of them to answer "what does it say".
    """
    try:
        nodes = window.descendants(control_type="Text")
    except Exception:
        return ""
    pieces: list[str] = []
    total = 0
    for node in nodes[:max_nodes]:
        try:
            text = " ".join(str(node.window_text() or "").split())
        except Exception:
            continue
        if not text:
            continue
        pieces.append(text)
        total += len(text) + 1
        if total >= max_chars:
            pieces.append("...")
            break
    return "\n".join(pieces)


def attach(handle: int):
    """UI Automation attached to one window by handle.

    Never `Desktop().windows()`: measured here, enumerating the desktop through
    UIA takes two minutes, and attaching to a single handle takes 21 ms.
    """
    from pywinauto import Desktop

    return Desktop(backend="uia").window(handle=handle)


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
