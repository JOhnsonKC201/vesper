"""The skill library, and the several ways a library like this quietly rots.

Most of these are not "does it store a thing". They are about the failure modes
that make a growing library worse than no library: recipes that cannot be
replayed, recipes learned from runs that did not actually work, and retrieval
that keeps handing back something that has never once succeeded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hands import distill
from hands.elements import Element, Observation, Rect
from hands.skills import Skill, SkillLibrary, resolve_click, tokenize


def element(name, control_type="Button", eid=0, box=(0, 0, 10, 10)):
    return Element(eid=eid, name=name, control_type=control_type, rect=Rect(*box))


def performed(*records):
    return [{"verb": v, "args": a, "ok": True, "element": e}
            for v, a, e in records]


# --- retrieval --------------------------------------------------------------


def test_filler_words_do_not_drive_retrieval():
    assert "notepad" in tokenize("Open Notepad and save the file")
    assert "the" not in tokenize("Open Notepad and save the file")
    assert "open" not in tokenize("Open Notepad")


def test_a_relevant_recipe_comes_back_and_an_unrelated_one_does_not(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(Skill(name="notepad_save", description="save a file in notepad",
                      goal="type text into notepad and save it"))
    library.add(Skill(name="spotify_search", description="search in spotify",
                      goal="open the spotify search view"))

    found = library.find("write something in notepad and save it")
    assert [s.name for s in found] == ["notepad_save"]


def test_a_goal_matching_nothing_returns_nothing(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(Skill(name="notepad_save", description="save a file in notepad",
                      goal="type into notepad"))
    assert library.find("defragment the registry") == []


def test_a_recipe_that_keeps_failing_is_demoted_not_hidden(tmp_path):
    """Dropping it would make the failure invisible. Demoting it means it can
    still be chosen, still fail, and still be seen failing in the trace."""
    library = SkillLibrary(tmp_path / "skills.json")
    good = Skill(name="notepad_write", description="notepad write file",
                 goal="notepad write file")
    bad = Skill(name="notepad_write_alt", description="notepad write file",
                goal="notepad write file")
    library.add(good)
    library.add(bad)
    for _ in range(4):
        library.record_use("notepad_write_alt", False)
    library.record_use("notepad_write", True)

    order = [s.name for s in library.find("notepad write file")]
    assert order[0] == "notepad_write"
    assert "notepad_write_alt" in order


# --- storage ----------------------------------------------------------------


def test_a_library_survives_a_round_trip_to_disk(tmp_path):
    path = tmp_path / "skills.json"
    library = SkillLibrary(path)
    library.add(Skill(name="a", description="d", goal="g",
                      steps=[{"verb": "launch", "target": "notepad"}],
                      params=["target"]))
    library.save()

    reloaded = SkillLibrary(path).load()
    assert len(reloaded) == 1
    assert reloaded.skills["a"].steps[0]["target"] == "notepad"
    assert reloaded.skills["a"].params == ["target"]


def test_a_shorter_route_to_the_same_place_replaces_the_longer_one(tmp_path):
    """The second win is often the tidier one, and a library that compounds
    should keep the better path, not the first one it happened to see."""
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(Skill(name="x", description="d", goal="g",
                      steps=[{"verb": "wait"}] * 5))
    library.record_use("x", True)
    library.add(Skill(name="x", description="d", goal="g",
                      steps=[{"verb": "wait"}] * 2))

    assert len(library.skills["x"].steps) == 2
    # The record of how it has done must survive the swap.
    assert library.skills["x"].uses == 1


# --- replay -----------------------------------------------------------------


def test_a_stored_click_finds_its_button_again():
    seen = Observation(elements=[element("Cancel", eid=0), element("Save", eid=1)])
    found = resolve_click({"name": "Save", "control_type": "Button"}, seen)
    assert found is not None and found.eid == 1


def test_save_is_preferred_over_save_as():
    """Both match by substring. Picking the wrong one is a different task."""
    seen = Observation(elements=[element("Save As...", eid=0), element("Save", eid=1)])
    assert resolve_click({"name": "Save"}, seen).eid == 1


def test_a_button_that_is_gone_resolves_to_nothing_rather_than_anything():
    seen = Observation(elements=[element("Cancel")])
    assert resolve_click({"name": "Save"}, seen) is None


def test_parameters_are_filled_in_before_replay():
    skill = Skill(name="s", description="d", goal="g", params=["text"],
                  steps=[{"verb": "type", "text": "{text}"}])
    assert skill.bind({"text": "hello"})[0]["text"] == "hello"


def test_a_missing_parameter_raises_rather_than_typing_the_placeholder():
    """Otherwise it types a literal "{path}" into whatever has focus, and the
    trace shows a model error where there was a binding error."""
    skill = Skill(name="s", description="d", goal="g", params=["path"],
                  steps=[{"verb": "type", "text": "{path}"}])
    with pytest.raises(KeyError):
        skill.bind({})


# --- distillation -----------------------------------------------------------


def test_a_winning_run_becomes_a_replayable_recipe():
    task = {"id": "np-01", "app": "notepad",
            "goal": "Open Notepad and save to C:/tmp/np-01.txt"}
    skill = distill.from_actions(task, performed(
        ("launch", {"target": "notepad"}, None),
        ("type", {"text": "hello"}, None),
        ("hotkey", {"keys": "ctrl+s"}, None),
        ("click", {"eid": 3}, {"name": "Save", "control_type": "Button"}),
    ))
    assert skill is not None
    verbs = [step["verb"] for step in skill.steps]
    assert verbs == ["launch", "type", "hotkey", "click"]
    # The click stores what it pressed, never where it was.
    assert skill.steps[-1]["match"]["name"] == "Save"
    assert "eid" not in skill.steps[-1]


def test_a_literal_from_the_goal_is_lifted_into_a_parameter():
    task = {"id": "t", "app": "notepad",
            "goal": "save it to C:/tmp/report.txt please"}
    skill = distill.from_actions(task, performed(
        ("launch", {"target": "notepad"}, None),
        ("type", {"text": "C:/tmp/report.txt"}, None),
    ))
    assert skill.params == ["text"]
    assert skill.steps[1]["text"] == "{text}"


def test_a_one_action_run_is_not_worth_storing():
    """A task solved by one launch is a shortcut the planner finds for free.
    Storing those makes retrieval worse for the recipes that took real work."""
    task = {"id": "t", "goal": "open notepad"}
    assert distill.from_actions(task, performed(
        ("launch", {"target": "notepad"}, None),
    )) is None


def test_a_click_on_something_nameless_poisons_the_recipe_so_none_is_stored():
    """It could never be re-resolved, so the recipe would be broken from the
    moment it was written."""
    task = {"id": "t", "goal": "do a thing"}
    assert distill.from_actions(task, performed(
        ("launch", {"target": "notepad"}, None),
        ("click", {"eid": 1}, {"name": "", "control_type": "Button"}),
    )) is None


def test_failed_actions_are_left_out_of_the_recipe():
    task = {"id": "t", "app": "notepad", "goal": "type and save"}
    records = performed(
        ("launch", {"target": "notepad"}, None),
        ("type", {"text": "hello"}, None),
    )
    records.insert(1, {"verb": "hotkey", "args": {"keys": "ctrl+z"},
                       "ok": False, "element": None})
    skill = distill.from_actions(task, records)
    assert [s["verb"] for s in skill.steps] == ["launch", "type"]
