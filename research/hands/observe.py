"""The three observation modes Experiment 1 compares, behind one call.

(a) uia     the control tree, cheap and precise, blind to custom-drawn surfaces
(b) vision  a parsed screenshot, sees anything drawn, knows no names or state
(c) both    UFO2's hybrid: UIA first, vision only where UIA saw nothing

The fusion rule in (c) is the interesting part and it is deliberately
asymmetric. UIA wins every overlap, because a box labelled "icon" at 0.31
confidence is strictly worse than a control that says it is the Save button and
reports whether it is enabled. Vision only contributes where UIA has nothing,
which is exactly the gap UFO2 introduced hybrid detection to close. Merging the
other way round, or averaging them, would let the weaker signal overwrite the
stronger one and make (c) score worse than (a) for a reason that is an artifact
of the merge rather than a fact about the world.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from . import observe_uia, observe_vision
from .elements import Observation

MODES = ("uia", "vision", "both")

# Above this, a detected box and a UIA control are treated as the same thing
# and the UIA one is kept. Low rather than high on purpose: a near miss is far
# more likely to be the same button seen twice than two stacked controls, and
# the cost of wrongly merging is one lost duplicate, while the cost of wrongly
# keeping both is a planner choosing between two ids for one button.
SAME_THING_IOU = 0.30


def fuse(uia: Observation, vision: Observation) -> Observation:
    """UIA elements, plus the vision boxes that cover something UIA missed."""
    kept = list(uia.elements)
    next_eid = max((e.eid for e in kept), default=-1) + 1
    added = 0

    for candidate in vision.elements:
        if any(candidate.rect.iou(existing.rect) >= SAME_THING_IOU
               for existing in uia.elements):
            continue
        kept.append(replace(candidate, eid=next_eid))
        next_eid += 1
        added += 1

    notes = list(uia.notes) + list(vision.notes)
    notes.append(
        f"fused: {len(uia.elements)} from uia, {added} of "
        f"{len(vision.elements)} vision boxes added, "
        f"{len(vision.elements) - added} were duplicates"
    )
    return Observation(
        elements=kept,
        mode="both",
        # Summed, not maxed. Mode (c) really does pay for both passes, and
        # reporting the larger of the two would hide the cost of the hybrid.
        latency_s=uia.latency_s + vision.latency_s,
        screenshot_path=vision.screenshot_path,
        window_title=uia.window_title,
        notes=notes,
    )


def look(mode: str, *, detector=None, shots_dir: Path,
         max_elements: int = 200, max_depth: int = 12) -> Observation:
    """One observation in the named mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    if mode == "uia":
        return observe_uia.observe(max_elements=max_elements,
                                   max_depth=max_depth)
    if mode == "vision":
        return observe_vision.observe(detector, shots_dir=shots_dir,
                                      max_elements=max_elements)

    started = time.perf_counter()
    uia = observe_uia.observe(max_elements=max_elements, max_depth=max_depth)
    vision = observe_vision.observe(detector, shots_dir=shots_dir,
                                    max_elements=max_elements)
    fused = fuse(uia, vision)
    # The measured wall clock beats the sum of the parts if anything between
    # them cost time, and the honest number is the one the clock says.
    fused.latency_s = max(fused.latency_s, time.perf_counter() - started)
    return fused
