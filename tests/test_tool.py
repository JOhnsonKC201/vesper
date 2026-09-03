"""Vasper's hands.

The safety tests come first, because this is the file that gained a mouse.
"""

from pathlib import Path

import pytest

from vesper import tool
from vesper.brain.consent import ActionRequest
from vesper.config import BrainSettings


# --- what may happen without asking -----------------------------------------


def test_only_looking_and_visible_changes_are_free():
    """The standing allowlist is the whole safety story for the hands.

    Looking is free. So are the two changes that a person sees happen and can
    undo by hand. Everything else has to be asked about out loud.
    """
    hands = {t for t in BrainSettings().allowed_tools if "vasper" in t}
    assert hands == {
        "Bash(vasper windows:*)",
        "Bash(vasper apps:*)",
        "Bash(vasper screenshot:*)",
        "Bash(vasper focus:*)",
        "Bash(vasper open:*)",
    }


def test_clicking_and_typing_are_never_free():
    """A click can press Send. Typing can write anything into anything.

    If either of these ever appears on the standing allowlist, Vasper can use
    the mouse and keyboard with nobody asked, which is the thing this whole
    design exists to prevent.
    """
    allowed = BrainSettings().allowed_tools
    for forbidden in ("click", "type", "press", "drag", "key"):
        assert not any(f"vasper {forbidden}" in entry for entry in allowed), forbidden
    assert "Bash" not in allowed, "bare Bash would make every entry above pointless"


def test_a_grant_is_scoped_to_one_subcommand():
    """Approving a click must not also buy typing."""
    click = ActionRequest("Bash", {"command": "vasper click --window Chrome --name Send"})
    assert click.grants() == ("Bash(vasper click:*)",)
    typing = ActionRequest("Bash", {"command": "vasper type rm -rf /"})
    assert typing.grants() == ("Bash(vasper type:*)",)
    assert click.grants() != typing.grants()


def test_the_shim_exists_for_both_shells():
    """git-bash will not resolve a bare `vasper` to `vasper.cmd`, and cmd will
    not run the extensionless one. The brain's shell is git-bash here, so the
    POSIX one is the one that actually gets used, and it was missing at first."""
    root = Path(tool.__file__).resolve().parent.parent
    assert (root / "vasper").is_file()
    assert (root / "vasper.cmd").is_file()
    posix = (root / "vasper").read_bytes()
    assert posix.startswith(b"#!"), "needs a shebang to run from bash"
    assert b"\r\n" not in posix, "CRLF in a shell script is 'bad interpreter: /bin/sh^M'"


# --- window titles are as revealing here as anywhere ------------------------


@pytest.mark.parametrize("title", [
    "1Password", "Bitwarden - Vault", "Chase Bank - Online Banking",
    "InPrivate - Microsoft Edge",
])
def test_sensitive_titles_are_hidden_in_the_window_list(title):
    """A window list is read far more often than it is acted on."""
    assert tool._safe_title(title) == "[hidden]"


def test_ordinary_titles_are_kept():
    assert tool._safe_title("main.py - Visual Studio Code") == (
        "main.py - Visual Studio Code"
    )


# --- finding things ---------------------------------------------------------


def _windows(*titles):
    return [tool.Window(i + 1, 100 + i, "app.exe", t) for i, t in enumerate(titles)]


def test_a_substring_beats_a_fuzzy_score(monkeypatch):
    monkeypatch.setattr(tool, "visible_windows", lambda: _windows(
        "Untitled - Notepad", "Chrome", "Settings"))
    assert tool.find_window("notepad").title == "Untitled - Notepad"


def test_the_shortest_matching_title_wins(monkeypatch):
    """"Notepad" should be the editor, not its Save As dialog."""
    monkeypatch.setattr(tool, "visible_windows", lambda: _windows(
        "Untitled - Notepad - Save As", "Notepad"))
    assert tool.find_window("notepad").title == "Notepad"


def test_nothing_close_enough_returns_nothing(monkeypatch):
    """Acting on the alphabetically nearest window is worse than saying no."""
    monkeypatch.setattr(tool, "visible_windows", lambda: _windows("Chrome", "Settings"))
    assert tool.find_window("zzzzqqqq") is None


def test_an_empty_needle_matches_nothing(monkeypatch):
    monkeypatch.setattr(tool, "visible_windows", lambda: _windows("Chrome"))
    assert tool.find_window("   ") is None


def test_a_name_that_starts_the_app_beats_one_that_merely_contains_it(monkeypatch):
    monkeypatch.setattr(tool, "installed_apps", lambda: [
        Path("Visual Studio Code.lnk"), Path("Code.lnk")])
    best, _ = tool.match_app("code")
    assert best.stem == "Code"


def test_a_nonsense_app_name_matches_nothing(monkeypatch):
    monkeypatch.setattr(tool, "installed_apps", lambda: [
        Path("Google Chrome.lnk"), Path("Notepad.lnk")])
    best, others = tool.match_app("qqqzzzxx")
    assert best is None and others == []


def test_the_runners_up_come_back_too(monkeypatch):
    """"open code" is genuinely ambiguous and a wrong launch is worse than a question."""
    monkeypatch.setattr(tool, "installed_apps", lambda: [
        Path("Code.lnk"), Path("Code Insiders.lnk"), Path("Codec Pack.lnk")])
    best, others = tool.match_app("code")
    assert best.stem == "Code"
    assert [o.stem for o in others] == ["Code Insiders", "Codec Pack"]


# --- housekeeping -----------------------------------------------------------


def test_screenshots_do_not_fill_the_disk(tmp_path, monkeypatch):
    """A 2560x1600 png is about 400KB and this runs whenever it is asked to."""
    monkeypatch.setattr(tool, "SHOTS", tmp_path)
    for index in range(30):
        (tmp_path / f"shot-{1000 + index}.png").write_bytes(b"x")
    tool._prune_shots(keep=20)
    assert len(list(tmp_path.glob("shot-*.png"))) == 20
    # The newest survive, not an arbitrary twenty.
    assert (tmp_path / "shot-1029.png").exists()
    assert not (tmp_path / "shot-1000.png").exists()


def test_every_subcommand_is_reachable():
    parser = tool.build_parser()
    for command in ("windows", "apps", "open", "focus", "screenshot"):
        args = parser.parse_args([command] + (["x"] if command in {"open", "focus"} else []))
        assert callable(args.run), command


def test_a_failure_is_a_sentence_rather_than_a_traceback(monkeypatch, capsys):
    """The output is read by a model, which would parse a traceback as a result."""
    def explode():
        raise RuntimeError("the window vanished")

    monkeypatch.setattr(tool, "visible_windows", explode)
    assert tool.main(["windows"]) == 1
    printed = capsys.readouterr().out
    assert "Traceback" not in printed
    assert "windows failed: RuntimeError: the window vanished" in printed
