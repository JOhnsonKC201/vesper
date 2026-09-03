"""Turning a run that worked into something reusable.

Called only on a verified win. The gate is not negotiable and lives in the
caller, but it is worth restating here: a library grown from tasks the planner
merely believed it had finished would teach every future run the same mistake,
with the added authority of having been written down.

The description is generated deterministically rather than by asking a model,
which is a real tradeoff against Voyager. Voyager has the LLM write each
skill's description and retrieves on that, and a model-written description
would almost certainly retrieve better than a bag of verbs. It would also make
retrieval vary run to run, and the entire question this experiment asks is
whether the library changes success and step count. A retriever that had a
different day would sit directly on top of the measurement.
"""

from __future__ import annotations

import re

from .skills import Skill, stamp, tokenize

# Below this, a shared string between goal and action is more likely to be a
# coincidence ("in", "the", "txt") than a real parameter.
MIN_PARAM_LEN = 4

# Verbs worth keeping in a skill. `wait` survives because letting a window
# appear is genuinely part of the recipe.
KEEP = frozenset({"click", "type", "hotkey", "launch", "wait"})


def name_for(task: dict, steps: list[dict]) -> str:
    """A short, readable, stable name.

    From the goal rather than the task id, because a library keyed on `np-01`
    tells a future planner nothing, and retrieval matches on the name too.
    """
    words = tokenize(str(task.get("goal", "")))[:4]
    app = task.get("app")
    if isinstance(app, list):
        app = "_".join(app[:2])
    parts = [str(app)] if app else []
    parts += words
    slug = "_".join(parts) or "skill"
    slug = re.sub(r"[^a-z0-9_]+", "", slug.lower()).strip("_")
    return slug[:48] or "skill"


def describe(steps: list[dict], task: dict) -> str:
    """One sentence, built from what the steps actually do."""
    bits = []
    for step in steps:
        verb = step["verb"]
        if verb == "launch":
            bits.append(f"open {step.get('target', '')}")
        elif verb == "click":
            bits.append(f"click {step.get('match', {}).get('name', '?')!r}")
        elif verb == "type":
            text = step.get("text", "")
            bits.append(f"type {text[:24]!r}" if text else "type")
        elif verb == "hotkey":
            bits.append(step.get("keys", ""))
    # Deduplicated in order: a recipe that clicks the same button four times
    # reads better as one mention than four.
    seen = []
    for bit in bits:
        if bit and bit not in seen:
            seen.append(bit)
    goal = str(task.get("goal", "")).strip().split("\n")[0][:80]
    return f"{goal} :: {', '.join(seen[:8])}"


def parameterise(steps: list[dict], goal: str) -> tuple[list[dict], list[str]]:
    """Replace literals that came from the goal with named parameters.

    Deliberately conservative and deliberately limited. It only lifts a value
    that appears verbatim in the goal, which catches the case that matters (a
    path or a phrase the task specified) and misses everything a model would
    have generalised properly. That limit is the honest headline for this part:
    these are near-literal recipes with the obvious arguments pulled out, not
    the general programs Voyager writes.
    """
    lowered = goal.lower()
    params: list[str] = []
    used: dict[str, str] = {}
    out: list[dict] = []

    for step in steps:
        new_step = dict(step)
        for key in ("text", "target"):
            value = step.get(key)
            if not isinstance(value, str) or len(value) < MIN_PARAM_LEN:
                continue
            if value.lower() not in lowered:
                continue
            if value in used:
                new_step[key] = used[value]
                continue
            base = key
            name = base
            suffix = 2
            while name in params:
                name = f"{base}{suffix}"
                suffix += 1
            params.append(name)
            used[value] = "{%s}" % name
            new_step[key] = used[value]
        out.append(new_step)
    return out, params


def from_actions(task: dict, performed: list[dict]) -> Skill | None:
    """Build a skill from the actions a winning run actually performed.

    Returns None when there is nothing worth storing. A task solved in one
    launch is not a skill, it is a shortcut the planner will find again for
    free, and filling the library with those makes retrieval worse for the
    recipes that took real work.
    """
    steps: list[dict] = []
    for record in performed:
        if not record.get("ok"):
            continue
        verb = record.get("verb", "")
        if verb not in KEEP:
            continue
        args = record.get("args", {}) or {}

        if verb == "click":
            element = record.get("element") or {}
            name = element.get("name", "")
            if not name:
                # A click on something with no name cannot be re-resolved
                # later, so the recipe would be broken from the moment it was
                # written. Better to store no skill than a poisoned one.
                return None
            steps.append({"verb": "click",
                          "match": {"name": name,
                                    "control_type": element.get("control_type", "")}})
        elif verb == "type":
            steps.append({"verb": "type", "text": str(args.get("text", ""))})
        elif verb == "hotkey":
            steps.append({"verb": "hotkey", "keys": str(args.get("keys", ""))})
        elif verb == "launch":
            steps.append({"verb": "launch", "target": str(args.get("target", ""))})
        elif verb == "wait":
            steps.append({"verb": "wait", "seconds": float(args.get("seconds", 0))})

    if len(steps) < 2:
        return None

    goal = str(task.get("goal", ""))
    steps, params = parameterise(steps, goal)
    return Skill(
        name=name_for(task, steps),
        description=describe(steps, task),
        goal=goal,
        steps=steps,
        params=params,
        source_task=str(task.get("id", "")),
        created=stamp(),
    )
