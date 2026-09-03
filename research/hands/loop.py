"""The UFO-style control loop: look, decide, act, repeat.

One decision here is worth more than the rest of the file. Success is checked
against the world, not against what the planner claims. A loop that ends when
the model says "done, success true" measures the model's confidence, and models
are confidently wrong in exactly the multi-application cases WindowsWorld shows
they fail at, which would turn a 21% success rate into a reported 80% one.

So a task carries a `check`, and the check runs after the loop stops. The
planner's own verdict is recorded next to it, because the gap between the two
is itself a finding: an agent that thinks it succeeded and did not is a
different failure than one that gave up.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import observe as observe_module
from .actions import Actions


@dataclass
class Outcome:
    task_id: str
    mode: str
    ok: bool
    steps: int
    wall_s: float
    detail: str = ""
    claimed_ok: bool = False
    blocked: int = 0
    observation_s: float = 0.0
    planner_s: float = 0.0
    action_s: float = 0.0
    # Two counts, not one, and Experiment 2 is unreadable without both.
    # `steps` is planner decisions; `actions` is primitive actions performed.
    # A skill collapses six actions into one decision, so reporting only
    # `steps` would show the library winning by a factor of six purely because
    # the word changed meaning. The honest claim needs both to move.
    actions: int = 0
    skills_used: list = field(default_factory=list)
    skill_actions: int = 0
    learned: str = ""


def reset(spec: dict | None, scratch_root: Path) -> str:
    """Put the world back before a task runs.

    Experiment 2 measures the same tasks twice, and without this the second
    pass is meaningless: pass one creates np-01.txt, so pass two's check passes
    the instant it starts whether or not the agent does anything at all. Every
    repeat measurement needs the world returned to where it started, or the
    library appears to make everything succeed.

    Confined to the scratch directory, and refuses anything outside it. The
    reset is the one part of the harness that deletes, it runs before the guard
    has any say, and a mistyped path in a task file must not be able to take
    something real with it.
    """
    if not spec:
        return "no reset"

    kind = str(spec.get("kind", "")).lower()
    if kind != "delete_path":
        return f"unknown reset kind {kind!r}"

    target = Path(str(spec.get("path", ""))).resolve()
    root = Path(scratch_root).resolve()
    if root not in target.parents:
        # Refused, not clamped. Silently redirecting a delete somewhere else is
        # a worse surprise than refusing to do it.
        return f"REFUSED: {target} is outside the scratch root {root}"

    try:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    except OSError as exc:
        return f"could not reset {target}: {exc}"
    return f"removed {target}"


def verify(check: dict | None) -> tuple[bool, str]:
    """Ask the world whether the task actually got done.

    Deliberately few kinds of check, all of them cheap and objective. A check
    that needs its own agent to evaluate would put the thing being measured
    inside the measurement.
    """
    if not check:
        return False, "no check defined, so success cannot be claimed"

    kind = str(check.get("kind", "")).lower()

    if kind == "file_exists":
        path = Path(str(check.get("path", "")))
        return path.exists(), f"{path} {'exists' if path.exists() else 'is missing'}"

    if kind == "file_contains":
        path = Path(str(check.get("path", "")))
        needle = str(check.get("text", ""))
        if not path.exists():
            return False, f"{path} is missing"
        body = path.read_text(encoding="utf-8", errors="replace")
        return needle in body, f"{needle!r} {'found' if needle in body else 'absent'}"

    if kind == "window_contains":
        needle = str(check.get("text", "")).lower()
        try:
            from . import observe_uia

            title = observe_uia.observe(max_elements=1, max_depth=1).window_title
        except Exception as exc:
            return False, f"could not read the foreground window: {exc}"
        return needle in title.lower(), f"foreground window is {title!r}"

    if kind == "element_named":
        needle = str(check.get("text", "")).lower()
        try:
            from . import observe_uia

            found = observe_uia.observe()
        except Exception as exc:
            return False, f"could not observe: {exc}"
        hit = any(needle in element.name.lower() for element in found.elements)
        return hit, f"{needle!r} {'is' if hit else 'is not'} on screen"

    if kind == "manual":
        # For tasks whose outcome a person has to eyeball. Never counted as a
        # pass automatically, so it cannot inflate a mode's score.
        return False, "manual check, score this one by hand"

    return False, f"unknown check kind {kind!r}"


def replay_skill(skill, values: dict, hands: Actions, seen, step: int,
                 *, trace, task_id: str, mode: str, detector, shots_dir: Path,
                 max_elements: int, max_depth: int, settle_s: float) -> tuple[int, str]:
    """Perform a stored recipe. Returns (actions performed, failure or "").

    Re-observes before every click rather than resolving them all against the
    observation the skill was chosen on. That costs an observation per click
    and it is the only thing that makes a skill different from a macro: the
    Save button is in a dialog that did not exist when the recipe started.

    Stops at the first failure. Half a recipe is not a partial success, it is a
    machine in a state nobody planned for, and pressing on would take actions
    against a screen the recipe was never written for.
    """
    from .skills import resolve_click

    try:
        steps = skill.bind(values)
    except KeyError as exc:
        return 0, str(exc)

    performed = 0
    for index, stored in enumerate(steps, start=1):
        verb = stored["verb"]
        if verb == "click":
            fresh = observe_module.look(mode, detector=detector,
                                        shots_dir=shots_dir,
                                        max_elements=max_elements,
                                        max_depth=max_depth)
            element = resolve_click(stored.get("match", {}), fresh)
            if element is None:
                wanted = stored.get("match", {}).get("name", "?")
                trace.write("skill_miss", task=task_id, step=step,
                            skill=skill.name, at=index, wanted=wanted,
                            available=len(fresh.elements))
                return performed, f"{skill.name} step {index}: no {wanted!r} on screen"
            result = hands.click(element, step=step)
        elif verb == "type":
            result = hands.type_text(stored.get("text", ""), step=step)
        elif verb == "hotkey":
            result = hands.hotkey(stored.get("keys", ""), step=step)
        elif verb == "launch":
            result = hands.launch(stored.get("target", ""), step=step)
        elif verb == "wait":
            result = hands.wait(float(stored.get("seconds", 0)), step=step)
        else:
            return performed, f"{skill.name} step {index}: unknown verb {verb!r}"

        performed += 1
        if not result.ok:
            return performed, f"{skill.name} step {index}: {result.detail}"
        time.sleep(settle_s)
    return performed, ""


def run_task(task: dict, mode: str, planner, *, guard, trace,
             detector=None, shots_dir: Path, max_steps: int = 20,
             max_elements: int = 200, max_depth: int = 12,
             dry_run: bool = False, settle_s: float = 0.4,
             library=None, learn: bool = True,
             scratch_root: Path | None = None) -> Outcome:
    """One task, in one observation mode, start to finish.

    `library` is None for the without-library arm of Experiment 2. When it is
    None the planner is never handed a skills block, so its prompt is byte for
    byte the one it saw before the library existed.
    """
    task_id = str(task.get("id", "unnamed"))
    goal = str(task.get("goal", ""))

    trace.task_start(task_id, mode, goal)
    if scratch_root is not None and not dry_run:
        detail = reset(task.get("reset"), scratch_root)
        trace.write("reset", task=task_id, mode=mode, detail=detail)
    hands = Actions(guard=guard, trace=trace, task_id=task_id, dry_run=dry_run)

    history: list[str] = []
    started = time.perf_counter()
    observation_s = planner_s = action_s = 0.0
    claimed_ok = False
    stop_detail = f"ran out of steps at {max_steps}"
    steps = 0
    skills_used: list[str] = []
    skill_actions = 0

    offered = library.render(goal) if library is not None else ""
    if offered:
        trace.write("skills_offered", task=task_id, mode=mode,
                    skills=[line.strip() for line in offered.splitlines()])

    for step in range(1, max_steps + 1):
        steps = step

        seen = observe_module.look(mode, detector=detector, shots_dir=shots_dir,
                                   max_elements=max_elements, max_depth=max_depth)
        observation_s += seen.latency_s
        trace.observation(task_id, step, seen)

        thought_at = time.perf_counter()
        action = planner.next_action(goal, seen, history, offered)
        planner_s += time.perf_counter() - thought_at

        if action.verb == "use_skill":
            skill = library.skills.get(action.skill) if library is not None else None
            if skill is None:
                trace.write("bad_skill", task=task_id, step=step,
                            skill=action.skill)
                history.append(f"{step}. use_skill {action.skill} failed, no such skill")
                continue
            acted_at = time.perf_counter()
            performed, failure = replay_skill(
                skill, action.skill_args, hands, seen, step, trace=trace,
                task_id=task_id, mode=mode, detector=detector,
                shots_dir=shots_dir, max_elements=max_elements,
                max_depth=max_depth, settle_s=settle_s)
            action_s += time.perf_counter() - acted_at
            skills_used.append(skill.name)
            skill_actions += performed
            trace.write("skill_used", task=task_id, step=step, skill=skill.name,
                        actions=performed, failure=failure)
            history.append(
                f"{step}. use_skill {skill.name} performed {performed} actions"
                + (f", then FAILED: {failure}" if failure else "")
            )
            time.sleep(settle_s)
            continue

        if action.is_done:
            claimed_ok = action.success
            stop_detail = action.why or "planner stopped"
            break

        element = seen.by_id(action.eid) if action.verb == "click" else None
        if action.verb == "click" and element is None:
            # A hallucinated id. Recorded rather than retried silently: a
            # planner that keeps inventing element ids is a real finding about
            # a mode that shows it too few, and hiding it behind a retry would
            # make (b) look merely slow instead of unusable.
            trace.write("bad_reference", task=task_id, step=step, eid=action.eid,
                        available=len(seen.elements))
            history.append(f"{step}. click {action.eid} failed, no such element")
            continue

        acted_at = time.perf_counter()
        if action.verb == "click":
            result = hands.click(element, step=step)
        elif action.verb == "type":
            result = hands.type_text(action.text, step=step)
        elif action.verb == "hotkey":
            result = hands.hotkey(action.keys, step=step)
        elif action.verb == "launch":
            result = hands.launch(action.target, step=step)
        else:
            result = hands.wait(action.seconds, step=step)
        action_s += time.perf_counter() - acted_at

        history.append(
            f"{step}. {action.verb} {action.args()} "
            f"{'ok' if result.ok else 'FAILED: ' + result.detail}"
        )
        # The screen needs a moment before the next observation, or the agent
        # plans against the frame before its own click landed.
        time.sleep(settle_s)

    really_ok, check_detail = verify(task.get("check"))
    wall = time.perf_counter() - started
    performed_ok = sum(1 for record in hands.performed if record.get("ok"))

    # Credit or blame every skill this run leaned on, before anything is
    # learned from it. A skill that keeps being retrieved and keeps losing has
    # to become visible, or the library quietly gets worse while growing.
    if library is not None:
        for name in set(skills_used):
            library.record_use(name, really_ok)

    learned = ""
    if library is not None and learn and really_ok and not skills_used:
        # Only from a verified win, and only when the run did not already lean
        # on a skill. Distilling a run that replayed a recipe would store the
        # recipe again under a new name and let the library grow by echo.
        from . import distill

        candidate = distill.from_actions(task, hands.performed)
        if candidate is not None:
            fresh = library.add(candidate)
            learned = candidate.name
            trace.write("skill_learned", task=task_id, mode=mode,
                        skill=candidate.name, steps=len(candidate.steps),
                        params=candidate.params, new=fresh)

    outcome = Outcome(
        task_id=task_id, mode=mode, ok=really_ok, steps=steps, wall_s=wall,
        detail=f"{stop_detail}; check: {check_detail}",
        claimed_ok=claimed_ok, blocked=len(guard.blocked),
        observation_s=observation_s, planner_s=planner_s, action_s=action_s,
        actions=performed_ok, skills_used=skills_used,
        skill_actions=skill_actions, learned=learned,
    )
    trace.task_end(task_id, mode, really_ok, steps, wall, outcome.detail)
    trace.write("timing", task=task_id, mode=mode,
                observation_s=round(observation_s, 3),
                planner_s=round(planner_s, 3), action_s=round(action_s, 3),
                claimed_ok=claimed_ok, verified_ok=really_ok,
                decisions=steps, actions=performed_ok,
                skills_used=skills_used, learned=learned)
    return outcome
