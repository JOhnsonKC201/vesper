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
        "Bash(vasper look:*)",
        "Bash(vasper focus:*)",
        "Bash(vasper open:*)",
    }


def test_clicking_and_typing_are_never_free():
    """A click can press Send. Typing can write anything into anything.

    If any of these ever appears on the standing allowlist, Vasper can use
    the mouse and keyboard with nobody asked, which is the thing this whole
    design exists to prevent.
    """
    allowed = BrainSettings().allowed_tools
    for forbidden in ("click", "type", "press", "drag", "key", "scroll", "move"):
        assert not any(f"vasper {forbidden}" in entry for entry in allowed), forbidden
    assert "Bash" not in allowed, "bare Bash would make every entry above pointless"


HANDS = (
    "Bash(vasper click:*)", "Bash(vasper type:*)", "Bash(vasper key:*)",
    "Bash(vasper scroll:*)", "Bash(vasper move:*)",
)


def test_the_hands_are_one_grant_and_the_question_says_so():
    """A task is click the address bar, type, press enter. Three questions for
    it made the hands unusable, and the person is watching every action land.
    So one yes buys the five hand verbs for one turn, and the question names
    what it buys: the mouse and keyboard."""
    click = ActionRequest("Bash", {"command": 'vasper click "Send"'})
    assert click.grants() == HANDS
    assert click.spoken() == "use the mouse and keyboard to click Send"
    typing = ActionRequest("Bash", {"command": 'vasper type "weather baltimore" --enter'})
    assert typing.grants() == HANDS
    assert typing.spoken() == "use the mouse and keyboard to type weather baltimore"
    key = ActionRequest("Bash", {"command": "vasper key ctrl+l"})
    assert key.spoken() == "use the mouse and keyboard to press ctrl l"
    point = ActionRequest("Bash", {"command": "vasper click 812 640"})
    assert point.spoken() == "use the mouse and keyboard to click at 812 640"


def test_the_hands_buy_nothing_but_the_hands():
    """`vasper open` stays free and is not part of the grant; a hand action
    chained with anything else is two questions in one line and is refused."""
    assert "Bash(vasper open:*)" not in HANDS
    chained = ActionRequest("Bash", {"command": "vasper type hi && git commit -m x"})
    assert chained.grants() == ()
    assert not chained.approvable
    for spec in HANDS:
        assert spec not in ActionRequest("Bash", {"command": "git commit -m x"}).grants()


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
    for command in ("windows", "apps", "open", "focus", "screenshot", "look"):
        args = parser.parse_args([command] + (["x"] if command in {"open", "focus"} else []))
        assert callable(args.run), command
    for argv in (["click", "Send"], ["click", "812", "640", "--right"], ["move", "1", "2"],
                 ["type", "hello", "--enter"], ["key", "ctrl+l"], ["scroll", "down", "--times", "2"]):
        assert callable(parser.parse_args(argv).run), argv


def test_notepad_does_not_launch_onenote(monkeypatch):
    """Live, 2026-09-05: "notepad" scored 0.57 against "OneNote" under the old
    0.45 floor and OneNote opened in front of the user. Windows 11 Notepad has
    no Start Menu shortcut, so a miss must be a miss, not the nearest thing."""
    monkeypatch.setattr(tool, "installed_apps", lambda: [Path("OneNote.lnk"), Path("Word.lnk")])
    best, others = tool.match_app("notepad")
    assert best is None and others == []


def test_a_typo_still_finds_the_app(monkeypatch):
    monkeypatch.setattr(tool, "installed_apps", lambda: [Path("Google Chrome.lnk"), Path("Word.lnk")])
    assert tool.match_app("chorme")[0].stem == "Google Chrome"


def test_store_apps_are_matched_like_shortcuts_and_launched_by_their_id(monkeypatch):
    """Notepad, Calculator and Terminal live only in the shell's AppsFolder on
    Windows 11. They match by name like a shortcut and launch as shell:AppsFolder\\<id>,
    which is the same double click the Start menu makes."""
    monkeypatch.setattr(tool, "installed_apps", lambda: [
        Path("Word.lnk"),
        tool.App("Notepad", "shell:AppsFolder\\Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"),
    ])
    best, _ = tool.match_app("notepad")
    assert best.stem == "Notepad"
    assert str(best).startswith("shell:AppsFolder\\")


def test_a_shortcut_beats_a_store_app_of_the_same_name(monkeypatch, tmp_path):
    menu = tmp_path / "menu"
    menu.mkdir()
    (menu / "Notepad.lnk").write_bytes(b"")
    monkeypatch.setattr(tool, "START_MENUS", (menu,))
    monkeypatch.setattr(tool, "store_apps", lambda: [tool.App("Notepad", "shell:AppsFolder\\x!App")])
    apps = tool.installed_apps()
    assert [a.stem for a in apps] == ["Notepad"]
    assert isinstance(apps[0], Path)


def test_no_shell_com_means_no_store_apps_and_no_crash(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_com(name, *args, **kwargs):
        if name.startswith("win32com"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_com)
    assert tool.store_apps() == []


def test_a_click_target_is_a_point_or_a_name(monkeypatch, capsys):
    """Two integers are a point; anything else is looked up on the front window
    fresh, because positions from an earlier look go stale when it moves."""
    monkeypatch.setattr(tool, "_front_window", lambda: (42, "Untitled - Notepad"))
    monkeypatch.setattr(tool, "_elements_of", lambda handle, limit=120: [
        tool.hands.Element(0, "Save", "Button", 800, 630, 824, 650),
        tool.hands.Element(1, "Save As", "Button", 900, 630, 960, 650),
    ])
    assert tool._resolve_target(["812", "640"]) == ((812, 640), "812,640")
    point, label = tool._resolve_target(["Save"])
    assert point == (812, 640) and label == "'Save'"
    point, label = tool._resolve_target(["Quit"])
    assert point is None and "nothing called 'Quit'" in label and "vasper look" in label


def test_a_look_at_a_sensitive_window_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(tool, "_front_window", lambda: (42, "1Password"))
    assert tool.main(["look"]) == 1
    out = capsys.readouterr().out
    assert "[hidden]" in out and "1Password" not in out


def _fake_win32(monkeypatch, *, front: int):
    """SetForegroundWindow that Windows quietly ignores, which is the normal
    case from a background process, plus whichever window is really in front."""
    win32gui = pytest.importorskip("win32gui")
    monkeypatch.setattr(win32gui, "IsIconic", lambda handle: False)
    monkeypatch.setattr(win32gui, "SetForegroundWindow", lambda handle: None)
    monkeypatch.setattr(win32gui, "GetForegroundWindow", lambda: front)
    monkeypatch.setattr(win32gui, "GetWindowText", lambda handle: "Terminal" if handle == 7 else "Notepad")
    monkeypatch.setattr(
        tool, "find_window", lambda title: tool.Window(42, 1, "notepad.exe", "Untitled - Notepad")
    )


def test_focus_reports_the_window_that_actually_won(monkeypatch, capsys):
    """On 2026-09-05 focus said "focused LinkedIn" while the terminal stayed on
    top, and the brain went on to describe a screenshot of the wrong window.
    The call rarely raises when Windows ignores it, so the result is checked."""
    _fake_win32(monkeypatch, front=7)
    assert tool.main(["focus", "notepad"]) == 1
    printed = capsys.readouterr().out
    assert "kept 'Terminal' in front" in printed
    assert "focused" not in printed


def test_focus_says_focused_only_when_the_window_is_in_front(monkeypatch, capsys):
    _fake_win32(monkeypatch, front=42)
    assert tool.main(["focus", "notepad"]) == 0
    assert "focused Untitled - Notepad" in capsys.readouterr().out


def test_a_failure_is_a_sentence_rather_than_a_traceback(monkeypatch, capsys):
    """The output is read by a model, which would parse a traceback as a result."""
    def explode():
        raise RuntimeError("the window vanished")

    monkeypatch.setattr(tool, "visible_windows", explode)
    assert tool.main(["windows"]) == 1
    printed = capsys.readouterr().out
    assert "Traceback" not in printed
    assert "windows failed: RuntimeError: the window vanished" in printed
