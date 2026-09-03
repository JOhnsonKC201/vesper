"""The whole Experiment 2 cycle, end to end, without touching a screen.

Observation is stubbed with a canned screen and actions run in dry mode, so
what is exercised here is the part that decides things: what gets learned, what
does not, and what happens on the second encounter with the same task. The
checks are real file checks against a tmp directory, which is what makes
"verified win" mean something in a test that never opens a window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hands import loop as loop_module
from hands import trace as trace_module
from hands.elements import Element, Observation, Rect
from hands.guard import Guard
from hands.planner import Action, ScriptedPlanner
from hands.skills import SkillLibrary


SCREEN = Observation(
    elements=[
        Element(0, "File", "MenuItem", Rect(0, 0, 40, 20)),
        Element(1, "Save", "Button", Rect(50, 0, 90, 20)),
        Element(2, "Cancel", "Button", Rect(100, 0, 140, 20)),
    ],
    mode="uia",
    window_title="Notepad",
)


@pytest.fixture(autouse=True)
def canned_screen(monkeypatch):
    monkeypatch.setattr(loop_module.observe_module, "look",
                        lambda *a, **k: SCREEN)


def run(task, planner, library, tmp_path, learn=True):
    with trace_module.Trace(tmp_path / "t.jsonl") as trace:
        return loop_module.run_task(
            task, "uia", planner, guard=Guard(), trace=trace,
            shots_dir=tmp_path, dry_run=True, settle_s=0.0,
            library=library, learn=learn, max_steps=8,
        )


def winning_task(tmp_path):
    """A task whose check passes, because the test made it pass. The point is
    the loop's reaction to a verified win, not who created the file."""
    target = tmp_path / "out.txt"
    target.write_text("done", encoding="utf-8")
    return {
        "id": "np-01",
        "app": "notepad",
        "goal": "Open Notepad, type report and press Save",
        "check": {"kind": "file_exists", "path": str(target)},
    }


def losing_task(tmp_path):
    return {
        "id": "np-99",
        "app": "notepad",
        "goal": "Open Notepad, type report and press Save",
        "check": {"kind": "file_exists", "path": str(tmp_path / "never.txt")},
    }


# The distiller lifts literals that appear in the goal into parameters, so a
# replay has to supply them. "notepad" and "report" are both in the goal.
ARGS = {"target": "notepad", "text": "report"}


def script():
    return ScriptedPlanner([
        Action("launch", target="notepad"),
        Action("type", text="report"),
        Action("click", eid=1),
        Action("done", success=True),
    ])


def test_a_verified_win_becomes_a_skill(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    outcome = run(winning_task(tmp_path), script(), library, tmp_path)

    assert outcome.ok
    assert len(library) == 1
    skill = next(iter(library.skills.values()))
    assert [s["verb"] for s in skill.steps] == ["launch", "type", "click"]
    # Stored by what it pressed, so it can be found again on a screen that has
    # moved.
    assert skill.steps[-1]["match"]["name"] == "Save"


def test_a_run_that_failed_its_check_teaches_nothing(tmp_path):
    """The whole gate. The planner said success true and the world disagreed,
    and it is the world that decides what enters the library."""
    library = SkillLibrary(tmp_path / "skills.json")
    outcome = run(losing_task(tmp_path), script(), library, tmp_path)

    assert not outcome.ok
    assert outcome.claimed_ok is True
    assert len(library) == 0


def test_the_second_encounter_replays_the_recipe(tmp_path):
    """One planner decision instead of four, and the stored actions really
    happen: the click is re-resolved against a live observation."""
    library = SkillLibrary(tmp_path / "skills.json")
    task = winning_task(tmp_path)
    first = run(task, script(), library, tmp_path)
    assert len(library) == 1
    learned = next(iter(library.skills.values())).name

    second = run(task, ScriptedPlanner([
        Action("use_skill", skill=learned, skill_args=ARGS),
        Action("done", success=True),
    ]), library, tmp_path, learn=False)

    assert second.ok
    assert second.skills_used == [learned]
    # Decisions fall from 4 to 2, but the actions still happen. Both numbers
    # are reported precisely so this cannot be mistaken for the agent doing
    # less work than it did.
    assert second.steps < first.steps
    assert second.skill_actions == 3


def test_a_skill_that_worked_is_credited(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    task = winning_task(tmp_path)
    run(task, script(), library, tmp_path)
    learned = next(iter(library.skills.values())).name

    run(task, ScriptedPlanner([Action("use_skill", skill=learned, skill_args=ARGS),
                               Action("done", success=True)]),
        library, tmp_path, learn=False)

    assert library.skills[learned].uses == 1
    assert library.skills[learned].wins == 1


def test_a_run_that_leaned_on_a_skill_does_not_learn_it_again(tmp_path):
    """Otherwise the library grows by echo: the same recipe stored under a new
    name every time it is used."""
    library = SkillLibrary(tmp_path / "skills.json")
    task = winning_task(tmp_path)
    run(task, script(), library, tmp_path)
    learned = next(iter(library.skills.values())).name

    run(task, ScriptedPlanner([Action("use_skill", skill=learned, skill_args=ARGS),
                               Action("done", success=True)]),
        library, tmp_path, learn=True)

    assert len(library) == 1


def test_asking_for_a_skill_that_does_not_exist_is_survivable(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    outcome = run(winning_task(tmp_path), ScriptedPlanner([
        Action("use_skill", skill="no_such_recipe"),
        Action("done", success=True),
    ]), library, tmp_path, learn=False)

    assert outcome.skills_used == []
    assert outcome.ok


def test_a_recipe_whose_button_is_gone_fails_loudly_and_stops(tmp_path):
    """Half a replayed recipe is a machine in a state nobody planned for, so it
    must stop at the miss rather than carry on against the wrong screen."""
    library = SkillLibrary(tmp_path / "skills.json")
    task = winning_task(tmp_path)
    run(task, script(), library, tmp_path)
    skill = next(iter(library.skills.values()))
    skill.steps[-1]["match"]["name"] = "Publish"

    outcome = run(task, ScriptedPlanner([
        Action("use_skill", skill=skill.name, skill_args=ARGS),
        Action("done", success=True),
    ]), library, tmp_path, learn=False)

    # launch and type ran, the click did not.
    assert outcome.skill_actions == 2


def test_the_without_arm_never_sees_a_skills_block(tmp_path):
    """Experiment 2's control. With no library the prompt must be byte for byte
    the one the planner saw before the library existed."""
    seen = {}

    class Spy(ScriptedPlanner):
        def next_action(self, goal, observation, history, skills=""):
            seen["skills"] = skills
            return Action("done", success=True)

    run(winning_task(tmp_path), Spy([]), None, tmp_path)
    assert seen["skills"] == ""


def test_a_parameterised_recipe_invoked_with_no_arguments_does_nothing(tmp_path):
    """It must not replay with a literal "{text}" in it, which would type the
    placeholder into whatever had focus and read as a model error rather than a
    binding error. Nothing happens, and the trace says why."""
    library = SkillLibrary(tmp_path / "skills.json")
    task = winning_task(tmp_path)
    run(task, script(), library, tmp_path)
    learned = next(iter(library.skills.values()))
    assert learned.params, "this test is meaningless if nothing was parameterised"

    outcome = run(task, ScriptedPlanner([
        Action("use_skill", skill=learned.name),
        Action("done", success=True),
    ]), library, tmp_path, learn=False)

    assert outcome.skill_actions == 0
