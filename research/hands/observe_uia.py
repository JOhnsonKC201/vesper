"""Observation (a): the UI Automation tree, which is what UFO reads first.

UIA is the cheap and precise half of the comparison. It gives real names, real
control types and real enabled/disabled state, with no model in the loop, in
single-digit milliseconds for a small window. Where it fails is where UFO2 says
it fails: custom-drawn surfaces (canvas apps, Electron with no accessibility
tree, game UI) that report one giant pane and nothing inside it.

Two caps that are not tuning knobs but correctness requirements:

`max_depth` and `max_elements` exist because a full Chrome tree is tens of
thousands of nodes and walking it can take minutes. An observation that takes
longer than the task is not an observation, it is a hang, and an experiment
measuring latency per mode must not have one mode able to stall indefinitely.

Breadth first rather than depth first, so that when the cap bites, what
survives is the top of the tree, which is where the toolbar and the menu bar
live. Depth first under a cap returns the deepest corner of the first panel and
nothing a planner can use.
"""

from __future__ import annotations

import time
from collections import deque

from .elements import Element, Observation, Rect

# Control types worth handing to a planner even when they carry no name. A
# nameless Edit box is actionable; a nameless Pane is scenery.
INTERACTIVE = frozenset({
    "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "ListItem",
    "MenuItem", "RadioButton", "Slider", "SplitButton", "TabItem", "Text",
    "TreeItem", "Document", "DataItem", "Spinner", "ToolBar",
})

# Never worth a slot in the element budget.
SKIP = frozenset({"Pane", "Group", "Custom", "Thumb", "Separator", "TitleBar"})


def _rect_of(control) -> Rect | None:
    try:
        box = control.rectangle()
    except Exception:
        return None
    rect = Rect(int(box.left), int(box.top), int(box.right), int(box.bottom))
    # Zero-area and off-screen controls are present in the tree constantly:
    # collapsed panels, virtualised list rows, the parts of a scrolled document
    # that are not drawn. None of them can be clicked.
    if rect.width <= 0 or rect.height <= 0:
        return None
    if rect.right < 0 or rect.bottom < 0:
        return None
    return rect


def _text_of(control) -> str:
    for getter in ("window_text", "element_info"):
        try:
            if getter == "window_text":
                text = control.window_text()
            else:
                text = getattr(control.element_info, "name", "")
            if text:
                return str(text).strip()
        except Exception:
            continue
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


def _value(control) -> str | None:
    """The contents of an edit box, which is the one piece of state a planner
    cannot get from a screenshot at all."""
    try:
        if hasattr(control, "get_value"):
            text = control.get_value()
            return str(text) if text else None
    except Exception:
        pass
    return None


def foreground_window():
    """The window the user is actually looking at.

    Raises rather than returning None: every caller here treats "no window" as
    a failed observation, and a None that flows onward becomes an empty element
    list, which is indistinguishable from a real window with nothing in it.
    """
    import win32gui
    from pywinauto import Desktop

    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("no foreground window")
    return Desktop(backend="uia").window(handle=hwnd)


def walk(window, *, max_elements: int = 200, max_depth: int = 12,
         start_eid: int = 0) -> tuple[list[Element], list[str]]:
    """Breadth first over the control tree, under both caps.

    Returns the elements and any notes worth putting in the trace, because
    "the cap bit" is a fact about the run that changes how its numbers read.
    """
    elements: list[Element] = []
    notes: list[str] = []
    eid = start_eid
    seen = 0

    queue: deque = deque([(window, 0)])
    while queue:
        control, depth = queue.popleft()
        seen += 1
        if len(elements) >= max_elements:
            notes.append(f"element cap {max_elements} reached after {seen} nodes")
            break
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
                # Still descend. A zero-area Pane routinely holds visible
                # children, so skipping the subtree loses real controls.
                queue.append((child, depth + 1))
                continue

            name = _text_of(child)
            useful = control_type in INTERACTIVE and (name or control_type == "Edit")
            if useful and control_type not in SKIP:
                elements.append(Element(
                    eid=eid,
                    name=name,
                    control_type=control_type,
                    rect=rect,
                    source="uia",
                    enabled=_enabled(child),
                    value=_value(child),
                ))
                eid += 1
                if len(elements) >= max_elements:
                    notes.append(f"element cap {max_elements} reached")
                    break
            queue.append((child, depth + 1))

    return elements, notes


def observe(*, max_elements: int = 200, max_depth: int = 12,
            window=None) -> Observation:
    """One UIA look at the foreground window."""
    started = time.perf_counter()
    notes: list[str] = []
    title = ""
    elements: list[Element] = []

    try:
        window = window if window is not None else foreground_window()
        try:
            title = _text_of(window)
        except Exception:
            title = ""
        elements, notes = walk(window, max_elements=max_elements,
                               max_depth=max_depth)
    except Exception as exc:
        # Reported, never raised. A backend that cannot see is a data point in
        # this experiment, not a crash: "UIA returned nothing on Spotify" is
        # exactly the finding UFO2's hybrid detection exists to address.
        notes.append(f"uia failed: {exc}")

    return Observation(
        elements=elements,
        mode="uia",
        latency_s=time.perf_counter() - started,
        window_title=title,
        notes=notes,
    )
