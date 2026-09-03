"""Observation (b): a screenshot, parsed into clickable boxes.

This is the OmniParser half of the comparison. It sees what is drawn rather
than what is declared, which is the whole point: the Windows Terminal window
this was first tested against reports nine UIA elements, none of them the text
you are reading, because it paints its own surface.

The detector is behind a protocol on purpose. OmniParser is a YOLO icon
detector plus a captioner, roughly two gigabytes of weights, and hard-wiring it
would mean this module cannot be imported, read or tested on a machine that has
not downloaded them yet. More importantly it would make the honest answer
impossible: when no detector is present this returns an empty observation and
says so in the notes, rather than quietly falling back to something cheaper and
letting Experiment 1 report a number for a mode that never ran.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from .elements import Element, Observation, Rect


class Detector(Protocol):
    """Turns an image into candidate boxes.

    Returning (rect, label, confidence) rather than an Element keeps the
    detector ignorant of the element id scheme, which is assigned per
    observation and is not the detector's business.
    """

    def detect(self, image_path: str) -> list[tuple[Rect, str, float]]:
        ...


def grab(path: Path, monitor: int = 1) -> tuple[Path, tuple[int, int]]:
    """Screenshot the given monitor. Returns the path and its pixel size.

    `mss` writes the png itself, which is why Pillow is not a dependency here.
    """
    import mss
    import mss.tools

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[monitor])
        mss.tools.to_png(shot.rgb, shot.size, output=str(path))
    return path, (shot.size.width, shot.size.height)


class OmniParser:
    """Adapter for OmniParser weights, if they are on this machine.

    Deliberately lazy and deliberately quiet about failing: `available()` is the
    question every caller actually wants answered, and a constructor that raises
    on a missing model would make "is this mode runnable" impossible to ask
    without a try block at every call site.
    """

    def __init__(self, weights: str | None = None, box_threshold: float = 0.05):
        self.weights = weights
        self.box_threshold = box_threshold
        self._model = None
        self._error = ""

    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._model is not None or self._error:
            return self._model
        if not self.weights or not Path(self.weights).exists():
            self._error = f"weights not found at {self.weights!r}"
            return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        except ImportError:
            self._error = "ultralytics is not installed"
        except Exception as exc:
            self._error = f"could not load weights: {exc}"
        return self._model

    @property
    def why_not(self) -> str:
        self._load()
        return self._error

    def detect(self, image_path: str) -> list[tuple[Rect, str, float]]:
        model = self._load()
        if model is None:
            return []
        found: list[tuple[Rect, str, float]] = []
        results = model.predict(source=image_path, conf=self.box_threshold,
                                verbose=False)
        for result in results:
            for box in getattr(result, "boxes", []):
                try:
                    left, top, right, bottom = (
                        int(v) for v in box.xyxy[0].tolist()
                    )
                    confidence = float(box.conf[0])
                except Exception:
                    continue
                # No caption model wired yet, so the label is honest about
                # being a box and nothing more. A planner told "icon" behaves
                # differently than one told "the Save button", and pretending
                # otherwise would flatter this mode.
                found.append((Rect(left, top, right, bottom), "icon", confidence))
        return found


def observe(detector: Detector | None = None, *, shots_dir: Path,
            max_elements: int = 200, monitor: int = 1) -> Observation:
    """One visual look at the screen."""
    started = time.perf_counter()
    notes: list[str] = []
    elements: list[Element] = []
    shot_path: str | None = None

    try:
        path, _size = grab(Path(shots_dir) / f"shot-{int(time.time() * 1000)}.png",
                           monitor=monitor)
        shot_path = str(path)
    except Exception as exc:
        notes.append(f"screenshot failed: {exc}")
        return Observation(mode="vision", latency_s=time.perf_counter() - started,
                           notes=notes)

    if detector is None:
        notes.append("no detector wired, so this mode saw nothing")
    else:
        try:
            boxes = detector.detect(shot_path)
            if not boxes and hasattr(detector, "why_not") and detector.why_not:
                notes.append(f"detector produced nothing: {detector.why_not}")
            # Largest first. When the element cap bites, the boxes most likely
            # to be a real control beat a stray four-pixel detection.
            boxes.sort(key=lambda item: item[0].area, reverse=True)
            for eid, (rect, label, confidence) in enumerate(boxes[:max_elements]):
                elements.append(Element(
                    eid=eid,
                    name=label,
                    control_type="Detected",
                    rect=rect,
                    source="vision",
                    confidence=confidence,
                ))
        except Exception as exc:
            notes.append(f"detection failed: {exc}")

    return Observation(
        elements=elements,
        mode="vision",
        latency_s=time.perf_counter() - started,
        screenshot_path=shot_path,
        notes=notes,
    )
