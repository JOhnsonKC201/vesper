"""A Voyager-style skill library, for a desktop instead of Minecraft.

Voyager stores executable code, retrieves it by similarity, and only admits a
skill once self-verification says the task worked. Two of those three carry
over unchanged. The third is replaced with something stricter: a skill is only
admitted when the task's `check` passed against the world. Voyager had to ask
the model whether it had succeeded because Minecraft gave it no better oracle.
Here there is one, and letting a model's own verdict decide what enters a
library that then teaches future runs is how a library fills up with confident
nonsense.

The representation is the part that needed the most thought. The obvious move
is to record the successful actions verbatim and replay them, and it does not
work: a click is stored as a screen coordinate and an element id, and both are
meaningless the next time round. Element ids are assigned per observation, and
coordinates move when a window opens 40 pixels lower. So a stored click carries
a *description* of what it pressed, and is re-resolved against a live
observation at replay time. That is the difference between a skill and a macro.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Words that appear in nearly every desktop goal and carry no signal about
# which skill is relevant.
STOPWORDS = frozenset("""
a an and are as at be by for from in into is it of on or so that the then to
using with open opens opened make makes made do does done use used
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")

# Substituted into a skill's arguments at replay. Deliberately not an f-string
# or anything eval-shaped: a skill is data replayed through the same guarded
# action layer as everything else, never code this process executes.
_PARAM = re.compile(r"\{([a-z0-9_]+)\}")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower())
            if t not in STOPWORDS and len(t) > 1]


@dataclass
class Skill:
    """One thing the agent worked out how to do, and can do again."""

    name: str
    description: str
    goal: str
    steps: list[dict] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    source_task: str = ""
    created: str = ""
    # Kept so a skill that keeps being retrieved and keeps failing is visible
    # rather than quietly dragging every future run down.
    uses: int = 0
    wins: int = 0

    @property
    def reliability(self) -> float | None:
        return self.wins / self.uses if self.uses else None

    def document(self) -> str:
        """What retrieval matches against."""
        return " ".join([self.name, self.description, self.goal])

    def render(self) -> str:
        """The one block the planner is shown."""
        head = f"- {self.name}({', '.join(self.params)}): {self.description}"
        if self.uses:
            head += f"  [used {self.uses}x, worked {self.wins}x]"
        return head

    def bind(self, values: dict) -> list[dict]:
        """Fill the parameters in, returning steps ready to replay.

        Raises on a missing parameter rather than replaying a step with a
        literal "{path}" in it, which would type the placeholder into whatever
        had focus and look like a model error rather than a binding error.
        """
        bound = []
        for step in self.steps:
            filled = dict(step)
            for key, value in step.items():
                if not isinstance(value, str):
                    continue

                def swap(match):
                    name = match.group(1)
                    if name not in values:
                        raise KeyError(
                            f"skill {self.name!r} needs parameter {name!r}")
                    return str(values[name])

                filled[key] = _PARAM.sub(swap, value)
            bound.append(filled)
        return bound


def resolve_click(match: dict, observation) -> object | None:
    """Find the element a stored click meant, in a fresh observation.

    Exact name beats prefix beats substring, and a matching control type breaks
    ties. Ranked rather than first-hit because "Save" and "Save As..." are both
    on the same toolbar and picking the wrong one is a different task.
    """
    want_name = str(match.get("name", "")).strip().lower()
    want_type = str(match.get("control_type", "")).strip()
    if not want_name:
        return None

    best = None
    best_score = 0.0
    for element in observation.elements:
        name = (element.name or "").strip().lower()
        if not name:
            continue
        if name == want_name:
            score = 3.0
        elif name.startswith(want_name):
            score = 2.0
        elif want_name in name:
            score = 1.0
        else:
            continue
        if want_type and element.control_type == want_type:
            score += 0.5
        if score > best_score:
            best, best_score = element, score
    return best


@dataclass
class SkillLibrary:
    """Load, retrieve, and grow. One json file, because a library you cannot
    open in an editor is a library you cannot audit."""

    path: Path
    skills: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> "SkillLibrary":
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.skills = {item["name"]: Skill(**item) for item in raw.get("skills", [])}
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [asdict(s) for s in self.skills.values()]}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def __len__(self) -> int:
        return len(self.skills)

    def add(self, skill: Skill) -> bool:
        """Admit a skill. Returns whether it was new.

        A repeat name replaces the old one only if the new one is shorter. The
        second successful run of a task is often the tidier one, having wasted
        fewer steps finding the window, and keeping the shorter path is the
        whole point of a library that compounds.
        """
        existing = self.skills.get(skill.name)
        if existing is not None:
            if len(skill.steps) < len(existing.steps):
                skill.uses, skill.wins = existing.uses, existing.wins
                self.skills[skill.name] = skill
            return False
        self.skills[skill.name] = skill
        return True

    def record_use(self, name: str, won: bool) -> None:
        skill = self.skills.get(name)
        if skill is None:
            return
        skill.uses += 1
        skill.wins += 1 if won else 0

    def _idf(self) -> dict:
        """Rarity weight per token, over the library as it stands."""
        total = len(self.skills) or 1
        seen: dict[str, int] = {}
        for skill in self.skills.values():
            for token in set(tokenize(skill.document())):
                seen[token] = seen.get(token, 0) + 1
        return {token: math.log(1 + total / count) for token, count in seen.items()}

    def find(self, goal: str, limit: int = 3, floor: float = 0.15) -> list[Skill]:
        """The skills worth showing the planner for this goal.

        Lexical, not embeddings. An embedding model would retrieve better, and
        it would also add a dependency and a source of run-to-run variance to
        an experiment whose entire question is whether the library helps. A
        deterministic retriever means a difference between the with and without
        arms is the library, not the retriever having a different day.
        """
        query = set(tokenize(goal))
        if not query:
            return []
        idf = self._idf()
        scale = math.sqrt(sum(idf.get(t, 1.0) for t in query)) or 1.0

        scored = []
        for skill in self.skills.values():
            shared = query & set(tokenize(skill.document()))
            if not shared:
                continue
            score = sum(idf.get(token, 1.0) for token in shared) / scale
            # A skill that has been tried and mostly failed is demoted rather
            # than dropped, so a bad skill can still be chosen and seen to fail
            # instead of vanishing and leaving the failure unexplained.
            if skill.uses >= 2 and (skill.reliability or 0) < 0.5:
                score *= 0.5
            if score >= floor:
                scored.append((score, skill))

        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [skill for _score, skill in scored[:limit]]

    def render(self, goal: str, limit: int = 3) -> str:
        found = self.find(goal, limit=limit)
        if not found:
            return ""
        lines = [skill.render() for skill in found]
        return "\n".join(lines)


def new_library(path: Path) -> SkillLibrary:
    return SkillLibrary(Path(path)).load()


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
