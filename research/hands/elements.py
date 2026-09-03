"""What the agent is allowed to see, in one shape.

The whole point of Experiment 1 is comparing three ways of finding out what is
on screen. That comparison is only meaningful if all three hand back the same
kind of thing, so every observation backend produces `Element`s and nothing
else. A planner reading a UIA tree and a planner reading OmniParser boxes see
the same fields in the same order, and the only difference between runs is
where those fields came from.

`source` exists so the fused mode can be audited after the fact: when
(c) beats (a) and (b), the trace should say which backend actually supplied the
element that got clicked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Rect:
    """Screen coordinates, left/top/right/bottom, in physical pixels."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def overlaps(self, other: "Rect") -> int:
        """Intersection area. Used to decide whether a UIA control and a
        detected box are the same thing."""
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.bottom, other.bottom) - max(self.top, other.top)
        return dx * dy if dx > 0 and dy > 0 else 0

    def iou(self, other: "Rect") -> float:
        """Intersection over union, the standard way to ask "same box?"."""
        overlap = self.overlaps(other)
        if overlap == 0:
            return 0.0
        union = self.area + other.area - overlap
        return overlap / union if union else 0.0


@dataclass(frozen=True)
class Element:
    """One thing on screen the agent could act on.

    `eid` is a small integer the planner refers to, not a handle. Handles are
    unstable across observations and would let a plan reference something that
    has since been repainted; a per-observation id makes a stale reference an
    obvious error rather than a silent misclick.
    """

    eid: int
    name: str
    control_type: str
    rect: Rect
    source: str = "uia"
    # UIA gives these; vision does not. Kept optional rather than faked, so a
    # comparison between backends cannot quietly credit vision with knowledge
    # it never had.
    enabled: bool | None = None
    value: str | None = None
    confidence: float | None = None

    def describe(self) -> str:
        """The single line the planner actually reads."""
        bits = [f"[{self.eid}]", self.control_type]
        if self.name:
            bits.append(repr(self.name))
        if self.value:
            bits.append(f"value={self.value!r}")
        if self.enabled is False:
            bits.append("(disabled)")
        return " ".join(bits)

    def to_json(self) -> dict:
        out = asdict(self)
        out["rect"] = [self.rect.left, self.rect.top, self.rect.right, self.rect.bottom]
        return out


@dataclass
class Observation:
    """One look at the screen, plus what it cost to take."""

    elements: list[Element] = field(default_factory=list)
    mode: str = "uia"
    # Wall clock for the observation itself. Experiment 1 reports latency per
    # mode, and a vision pass is orders of magnitude slower than a UIA walk, so
    # this is the number that makes the comparison honest.
    latency_s: float = 0.0
    screenshot_path: str | None = None
    window_title: str = ""
    notes: list[str] = field(default_factory=list)

    def by_id(self, eid: int) -> Element | None:
        for element in self.elements:
            if element.eid == eid:
                return element
        return None

    def render(self, limit: int = 120) -> str:
        """What the planner is shown. Truncated, because a full Chrome tree is
        thousands of nodes and blowing the context window is itself a result
        worth not having."""
        lines = [element.describe() for element in self.elements[:limit]]
        if len(self.elements) > limit:
            lines.append(f"... {len(self.elements) - limit} more not shown")
        return "\n".join(lines)
