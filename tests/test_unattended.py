"""Running without a terminal: logging, shutdown, autostart, voice matching.

The theme, and the reason these are one file: every one of them exists because
starting from your login removes the console, and the console was quietly doing
a lot of work. It was where errors went, where ctrl-c came from, and how you
knew the thing was alive at all.

The failure being designed against throughout is an invisible process nobody can
stop, so most of these assert that something remains possible rather than that a
feature works.
"""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from vesper import autostart
from vesper import main as main_module
from vesper import resume
from vesper import single
from vesper.audio.speaker import Speaker
from vesper.conversation import Conversation, ConversationConfig
from vesper.logfile import LogFile, attach
from vesper.stt import voiceprint
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI


# --- somewhere for errors to go ---------------------------------------------


def test_the_log_records_what_the_terminal_would_have_shown(tmp_path):
    log = LogFile(tmp_path / "vesper.log")
    ui = RecordingUI()
    attach(ui, log)

    ui.warn("microphone went away")
    ui.error("the brain died")

    written = (tmp_path / "vesper.log").read_text(encoding="utf-8")
    assert "microphone went away" in written
    assert "the brain died" in written
    # And the terminal still gets everything it did before.
    assert ui.warnings == ["microphone went away"]
    assert ui.errors == ["the brain died"]


def test_logging_survives_a_path_it_cannot_write(tmp_path):
    """A log that cannot be written is not a reason to stop answering. Same
    class of bug as audit.py's, where a null byte raised ValueError rather than
    OSError and slipped past the handler."""
    log = LogFile(tmp_path / "nope" / "\0bad")
    log.warn("still fine")  # must not raise

    good = LogFile(tmp_path / "vesper.log")
    good.warn("and logging still works afterwards")
    assert "afterwards" in (tmp_path / "vesper.log").read_text(encoding="utf-8")


def test_the_log_rotates_rather_than_filling_the_disk(tmp_path):
    """An assistant that fills a disk while trying to be helpful has failed
    twice, and this one runs from login until you stop it."""
    path = tmp_path / "vesper.log"
    log = LogFile(path, max_bytes=2_000)
    for index in range(400):
        log.info(f"line {index} " + "x" * 50)

    assert path.exists()
    assert path.stat().st_size < 10_000
    assert (tmp_path / "vesper.log.1").exists(), "never rotated"


def test_a_log_that_is_already_full_when_we_open_it_still_rotates(tmp_path):
    """The size is carried forward now rather than measured on every line.

    The one case that has to keep working is the first write of a process that
    inherited a file already over the limit, because that is what a restart is.
    """
    path = tmp_path / "vesper.log"
    path.write_text("x" * 5_000, encoding="utf-8")

    log = LogFile(path, max_bytes=2_000)
    log.info("first line after a restart")

    assert (tmp_path / "vesper.log.1").exists(), "the inherited file was never measured"
    assert "first line after a restart" in path.read_text(encoding="utf-8")


def test_a_write_that_failed_does_not_leave_a_stale_size_behind(tmp_path):
    """After a failure the running total can no longer be trusted."""
    path = tmp_path / "vesper.log"
    log = LogFile(path, max_bytes=2_000)
    log.info("one good line")
    assert log._written > 0

    log.path = tmp_path / "gone" / "\0bad"
    log.info("this one cannot be written")
    assert log._written == -1, "the next write must measure rather than assume"


def test_a_disabled_log_is_silent_not_broken(tmp_path):
    log = LogFile(None)
    assert not log.enabled
    log.error("nowhere to go")  # must not raise
    ui = RecordingUI()
    assert attach(ui, log) is ui


# --- being able to stop it --------------------------------------------------


def _conversation(brain=None, transcripts=None):
    speaker = Speaker(FakeVoice(duration=0.0))
    ui = RecordingUI()
    conversation = Conversation(
        brain=brain or FakeBrain([]),
        stt=FakeSTT(transcripts or []),
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(greet_on_start=False),
        ui=ui,
    )
    return conversation, speaker, ui


def test_shutdown_can_be_asked_for_from_another_thread():
    """The tray menu and a signal handler both arrive on a thread that is not
    the loop's. Neither may tear the loop down in place."""
    conversation, speaker, _ = _conversation()
    conversation._running.set()
    assert conversation.running

    threading.Thread(target=conversation.shutdown).start()
    for _ in range(50):
        if not conversation.running:
            break
        import time

        time.sleep(0.02)
    speaker.close()
    assert not conversation.running


@pytest.mark.parametrize(
    "said", ["shut down", "goodbye Vesper", "turn yourself off", "quit"]
)
def test_a_spoken_goodbye_ends_it_without_asking_claude(said):
    """With no console there is no ctrl-c, so this is the only way out that
    does not involve the tray or Task Manager. It must never depend on the
    network: you say it precisely when things have gone wrong."""
    brain = FakeBrain(["should never be asked"])
    conversation, speaker, _ = _conversation(brain=brain)
    conversation._running.set()

    conversation.hear(said)
    speaker.close()

    assert not conversation.running
    assert brain.asked == [], "a shutdown request went to Claude"


def test_go_to_sleep_puts_him_to_sleep_rather_than_ending_him():
    """It used to be a way of saying "shut down", which was reasonable when
    there was no such thing as being asleep. Now that he spends most of his
    time asleep and you can watch him do it, the phrase has to mean the state.

    Mutation that fails this: put "go to sleep" back in _SHUTDOWN_PHRASES.
    """
    import time as _time

    conversation, speaker, _ = _conversation()
    conversation._running.set()
    conversation.wake.engage(_time.monotonic())

    conversation.hear("go to sleep")
    speaker.close()

    assert conversation.running, "it ended the process instead of sleeping"
    assert not conversation.wake.engaged(_time.monotonic())


def test_pausing_stops_acting_on_audio_without_closing_the_microphone():
    """Reopening an audio device is where audio stacks go wrong, and the point
    of pausing is that resuming is instant."""
    conversation, speaker, _ = _conversation()
    conversation.pause(True)

    block = np.full(480, 0.5, dtype=np.float32)
    conversation._handle_block(block)  # must not raise or transcribe

    assert conversation.turns == 0
    conversation.pause(False)
    speaker.close()


# --- starting with windows --------------------------------------------------


def test_autostart_reports_honestly_when_it_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert not autostart.is_installed()
    assert "not installed" in autostart.describe()


def test_autostart_refuses_to_point_at_a_launcher_that_is_missing(monkeypatch, tmp_path):
    """A shortcut to nothing is worse than no shortcut: it fails silently at
    login, which is exactly where nobody is watching."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    ok, detail = autostart.install(tmp_path / "empty")
    assert not ok
    assert "run_silent.vbs" in detail


def test_removing_autostart_that_was_never_there_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    ok, detail = autostart.uninstall()
    assert ok and "not installed" in detail


def test_the_shortcut_lands_in_the_startup_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    expected = (tmp_path / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                / "Startup" / "Vesper.lnk")
    assert autostart.shortcut_path() == expected


# --- coming back on wake ----------------------------------------------------


def test_resume_task_refuses_to_point_at_a_launcher_that_is_missing(tmp_path):
    """Same trap as the startup shortcut, one step further from anyone
    watching: this one fires at unlock."""
    ok, detail = resume.install(tmp_path / "empty")
    assert not ok
    assert "run_silent.vbs" in detail


def test_the_task_runs_on_unlock_and_on_wake(tmp_path):
    """Two triggers, not one. A lid that opens onto a lock screen sends
    SessionUnlock; a resume that never locked sends neither, so the power
    event has to be there too."""
    (tmp_path / "run_silent.vbs").write_text("")
    spec = resume.plan(tmp_path)
    assert spec["triggers"] == ["session-unlock", "resume-from-sleep"]
    assert "Microsoft-Windows-Power-Troubleshooter" in spec["resume_event"]


def test_the_task_passes_if_idle_so_unlocking_cannot_stack_dialogs(tmp_path):
    """Without this flag every unlock while Vesper is running puts a message
    box on screen, because the second copy has no console to print into."""
    (tmp_path / "run_silent.vbs").write_text("")
    assert "--if-idle" in resume.plan(tmp_path)["arguments"]


def test_the_task_still_starts_on_battery(tmp_path):
    """Task Scheduler refuses to start on battery by default, which is exactly
    when a laptop lid opens."""
    (tmp_path / "run_silent.vbs").write_text("")
    settings = resume.plan(tmp_path)["settings"]
    assert settings["DisallowStartIfOnBatteries"] is False
    assert settings["StopIfGoingOnBatteries"] is False


def test_removing_a_resume_task_that_was_never_there_is_not_an_error(monkeypatch):
    monkeypatch.setattr(resume, "is_installed", lambda: False)
    ok, detail = resume.uninstall()
    assert ok and "no resume task" in detail


def test_if_idle_exits_quietly_when_another_vesper_holds_the_microphone(monkeypatch):
    """The whole reason the flag exists. cfg is never reached: the guard
    returns before anything looks at it."""
    monkeypatch.setattr(single, "claim", lambda *a, **k: False)
    shown = []
    monkeypatch.setattr(main_module, "say_on_screen",
                        lambda *a, **k: shown.append(a) or True)
    assert main_module.run_voice(None, if_idle=True) == 0
    assert shown == []


def test_without_if_idle_a_second_copy_still_says_so(monkeypatch):
    """The desktop icon depends on this: a double click that did nothing
    visible would read as a broken program."""
    monkeypatch.setattr(single, "claim", lambda *a, **k: False)
    shown = []
    monkeypatch.setattr(main_module, "say_on_screen",
                        lambda *a, **k: shown.append(a) or True)
    assert main_module.run_voice(None) == 1
    assert shown


# --- an icon on the desktop -------------------------------------------------


def test_the_icon_goes_where_windows_actually_draws_the_desktop(monkeypatch, tmp_path):
    """Not `~/Desktop`. With OneDrive backing the desktop up, which is the
    default on a new install and is the case here, the real one is
    `~/OneDrive/Desktop` while the old folder sits there full of leftovers.
    Writing to it would put the icon somewhere the person who asked for it
    never looks, and nothing anywhere would report a failure.

    Mutation that fails this: return `Path.home() / "Desktop"` unconditionally.
    """
    from vesper import desktop

    redirected = tmp_path / "OneDrive" / "Desktop"
    monkeypatch.setattr(desktop, "shell_folder", lambda name: str(redirected))

    assert desktop.desktop_dir() == redirected
    assert desktop.shortcut_path() == redirected / "Vesper.lnk"


def test_a_machine_that_cannot_be_asked_still_gets_a_guess(monkeypatch):
    from vesper import desktop

    monkeypatch.setattr(desktop, "shell_folder", lambda name: "")
    assert desktop.desktop_dir() == Path.home() / "Desktop"


def test_the_desktop_icon_refuses_to_point_at_a_launcher_that_is_missing(
        monkeypatch, tmp_path):
    from vesper import desktop

    monkeypatch.setattr(desktop, "desktop_dir", lambda: tmp_path)
    ok, detail = desktop.install(tmp_path / "empty")

    assert not ok and "run_silent.vbs" in detail
    assert not (tmp_path / "Vesper.lnk").exists()


def test_removing_a_desktop_icon_that_was_never_there_is_not_an_error(
        monkeypatch, tmp_path):
    from vesper import desktop

    monkeypatch.setattr(desktop, "desktop_dir", lambda: tmp_path)
    ok, detail = desktop.uninstall()

    assert ok and "not installed" in detail
    assert "not installed" in desktop.describe()


def test_the_icon_is_vespers_own_and_starts_it_without_a_console(
        monkeypatch, tmp_path):
    """The shortcut already on this desktop ran `run.bat` directly, so it left
    a console window open for the whole session, and it wore SndVol.exe's icon,
    the Windows volume mixer. Neither of those says "Vesper" to someone looking
    at a desktop full of shortcuts.

    Mutation that fails this: drop `icon=` from `desktop.install`, or point the
    shortcut at run.bat.
    """
    pytest.importorskip("win32com.client")
    from vesper import config as config_module
    from vesper import desktop

    monkeypatch.setattr(desktop, "desktop_dir", lambda: tmp_path)
    ok, detail = desktop.install(config_module.ROOT)
    assert ok, detail

    import win32com.client

    link = win32com.client.Dispatch("WScript.Shell").CreateShortcut(
        str(tmp_path / "Vesper.lnk"))

    assert link.TargetPath.lower().endswith("wscript.exe")
    assert "run_silent.vbs" in link.Arguments
    assert "vesper-desktop.ico" in link.IconLocation
    assert desktop.is_installed()


# --- and a click that lands on a Vesper already running ---------------------


def test_a_second_copy_says_so_where_it_can_actually_be_seen(monkeypatch):
    """Started from the desktop icon there is no console, so the printed line
    goes nowhere and the double click looks ignored. Which is exactly when
    someone clicks it again.

    Mutation that fails this: drop the `say_on_screen` call from `run_voice`.
    """
    from vesper import config as config_module
    from vesper import main as main_module
    from vesper import single

    shown = []
    monkeypatch.setattr(single, "claim", lambda *a, **k: False)
    monkeypatch.setattr(main_module, "say_on_screen",
                        lambda *lines, **kw: shown.append(lines) or True)

    assert main_module.run_voice(config_module.Config()) == 1
    assert shown and "already running" in shown[0][0]


def test_a_terminal_gets_the_printed_line_and_no_dialog(monkeypatch):
    """A dialog nobody asked for, in front of a console that just said the same
    thing, is worse than saying nothing."""
    from vesper import main as main_module

    monkeypatch.setattr(main_module, "console_is_visible", lambda: True)
    assert main_module.say_on_screen("hello") is False


def test_the_dialog_cannot_open_behind_a_full_screen_window(monkeypatch):
    import ctypes

    from vesper import main as main_module

    calls = []
    monkeypatch.setattr(main_module, "console_is_visible", lambda: False)
    monkeypatch.setattr(ctypes.windll.user32, "MessageBoxW",
                        lambda *args: calls.append(args) or 1)

    assert main_module.say_on_screen("one", "", "two") is True
    _, text, title, flags = calls[0]
    assert "one" in text and "two" in text
    assert title == "Vesper"
    assert flags & 0x40000, "not topmost, so it can open behind what is there"


# --- only your voice --------------------------------------------------------


def _fixture(name: str) -> np.ndarray:
    """Real speech, not a sine wave.

    The first version of these tests used stacked sinusoids and passed happily
    while the feature was useless: MFCC statistics separated synthetic tones
    fine and could not tell a man from a woman. Speaker A is Piper's en_GB-alan,
    speaker B is Windows SAPI's David. Two genuinely different voices, saved
    small, for the same reason speech_short.wav exists rather than a tone.
    """
    import wave

    path = Path(__file__).parent / "fixtures" / name
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


needs_model = pytest.mark.skipif(
    not voiceprint.available(),
    reason="the speaker model is not downloaded; run --enroll once",
)


def test_nothing_enrolled_means_every_voice_is_accepted(tmp_path):
    """The state before you ever run --enroll, and the state this must degrade
    to on any failure. An assistant that ignores its owner is the worst outcome
    this feature can produce."""
    store = voiceprint.VoicePrint(tmp_path / "none.json")
    assert not store.enrolled
    verdict, _ = store.compare(_fixture("speaker_a_1.wav"))
    assert verdict == voiceprint.UNSURE


@needs_model
def test_an_enrolled_voice_is_recognised(tmp_path):
    store = voiceprint.VoicePrint(tmp_path / "vp.json")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])
    verdict, score = store.compare(_fixture("speaker_a_3.wav"))
    assert verdict == voiceprint.MATCH, f"did not recognise its owner, {score:.3f}"


@needs_model
def test_a_different_real_voice_is_rejected(tmp_path):
    """The test the MFCC version failed. It scored a stranger at 0.93 against
    an owner at 0.99, and scored a female voice higher than a male one."""
    store = voiceprint.VoicePrint(tmp_path / "vp.json")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2, 3)])
    for index in (1, 2, 3):
        verdict, score = store.compare(_fixture(f"speaker_b_{index}.wav"))
        assert verdict == voiceprint.DIFFERENT, f"accepted a stranger, {score:.3f}"


@needs_model
def test_owner_and_stranger_are_far_enough_apart_to_threshold(tmp_path):
    """The number that decides whether any of this can work. Measured at +0.76
    with this model, against +0.06 for the MFCC version it replaced, where no
    threshold could sit between them."""
    store = voiceprint.VoicePrint(tmp_path / "vp.json")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])

    owner = store.compare(_fixture("speaker_a_3.wav"))[1]
    stranger = max(store.compare(_fixture(f"speaker_b_{i}.wav"))[1] for i in (1, 2, 3))

    assert owner - stranger > 0.3, f"owner {owner:.3f} vs stranger {stranger:.3f}"
    assert stranger < voiceprint.CLEARLY_DIFFERENT < voiceprint.CONFIDENT_MATCH < owner


def test_a_clip_too_short_to_judge_answers_anyway(tmp_path):
    """"Vesper" on its own is under a second. Refusing to wake for the wake
    word would be absurd."""
    store = voiceprint.VoicePrint(tmp_path / "vp.json")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])
    verdict, _ = store.compare(_fixture("speaker_a_1.wav")[:8000])
    assert verdict == voiceprint.UNSURE


def test_a_corrupt_profile_does_not_lock_you_out(tmp_path):
    """Same reasoning as the undo ledger: a bad file on disk must degrade to
    "no opinion", never to a silent assistant."""
    path = tmp_path / "vp.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = voiceprint.VoicePrint(path)
    assert not store.enrolled
    assert store.compare(_fixture("speaker_a_1.wav"))[0] == voiceprint.UNSURE


def test_a_profile_from_the_old_embedding_is_retired_not_compared(tmp_path):
    """The MFCC version wrote 40-dimension profiles. Comparing one against a
    256-dimension embedding scores zero, which would read as "not you" and lock
    the owner out. It has to be treated as not enrolled instead."""
    path = tmp_path / "vp.json"
    path.write_text(json.dumps({"embedding": [0.1] * 40, "samples": 4}), encoding="utf-8")
    store = voiceprint.VoicePrint(path)
    assert not store.enrolled
    assert store.compare(_fixture("speaker_a_1.wav"))[0] == voiceprint.UNSURE


@needs_model
def test_the_profile_records_how_much_your_own_samples_disagreed(tmp_path):
    path = tmp_path / "vp.json"
    store = voiceprint.VoicePrint(path)
    profile = store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2, 3)])

    assert 0.0 < profile.spread <= 1.0
    assert profile.samples == 3
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["samples"] == 3 and len(saved["embedding"]) == voiceprint.EMBEDDING_DIM


@needs_model
def test_the_conversation_only_rejects_a_confident_mismatch(tmp_path):
    """The wiring, not the maths. UNSURE has to reach the same place as MATCH,
    because you chose "answer anyway when unsure"."""
    conversation, speaker, ui = _conversation()
    conversation.voiceprint = voiceprint.VoicePrint(tmp_path / "vp.json")
    conversation.voiceprint.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])

    assert conversation._is_the_right_voice(_fixture("speaker_a_3.wav")) is True
    assert conversation._is_the_right_voice(_fixture("speaker_b_1.wav")) is False
    assert conversation.voice_rejections == 1
    speaker.close()


@needs_model
def test_voice_matching_is_fast_enough_to_be_invisible(tmp_path):
    """It runs on every utterance that contains the wake word, so it sits
    directly in the path between you speaking and Vesper answering."""
    import time

    store = voiceprint.VoicePrint(tmp_path / "vp.json")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])
    clip = _fixture("speaker_a_3.wav")
    store.compare(clip)  # warm

    started = time.perf_counter()
    for _ in range(20):
        store.compare(clip)
    each = (time.perf_counter() - started) / 20
    # Measured at about 57ms with this model. The bar is what a person would
    # notice appended to their answer, not the measurement itself.
    assert each < 0.30, f"{each * 1000:.0f}ms per utterance is too slow"


# --- the tray, minus the parts that need a message pump ---------------------


def test_the_tooltip_says_which_of_the_three_states_it_is_in():
    """Asleep and awake are both "listening", and the difference between them
    is what he will act on, which is the thing worth reading off a tooltip."""
    from vesper.ui.tray import TrayState

    state = TrayState(listening=True, detail="say the wake word")
    assert "asleep" in state.tooltip()

    state.awake = True
    assert "awake" in state.tooltip()

    # Paused beats both. It is the only one of the three where the microphone
    # is not being acted on at all.
    state.listening = False
    assert "paused" in state.tooltip()


def test_a_long_tooltip_is_truncated_rather_than_dropped():
    """Windows truncates at 128 characters and on some builds silently fails to
    show the icon at all if the string is longer. A missing icon is the one
    failure that leaves no way to quit."""
    from vesper.ui.tray import TrayState

    state = TrayState(detail="x" * 500)
    assert len(state.tooltip()) <= 127


def test_a_failing_menu_handler_does_not_strand_the_icon():
    """The callbacks run on the tray thread. An exception escaping one would
    kill the message pump, leaving an icon that no longer responds to Quit."""
    from vesper.ui.tray import TrayIcon

    tray = TrayIcon(on_quit=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    tray._safely(tray._on_quit)  # must not raise


def test_the_tray_reports_failure_instead_of_pretending():
    """`start()` returning True when there is no icon would mean nobody is told
    the only way to quit is missing."""
    from vesper.ui.tray import TrayIcon

    tray = TrayIcon()
    assert tray.available is False


# --- enrolment --------------------------------------------------------------


def test_enrolment_refuses_rather_than_saving_a_useless_profile(tmp_path, monkeypatch):
    """With too few usable clips there is nothing to compare against later.
    Saving anyway would produce a profile that cannot recognise its owner,
    which is the worst outcome this feature has."""
    from vesper import config as config_module
    from vesper import enroll

    cfg = config_module.Config()
    monkeypatch.setattr(cfg, "voiceprint_path", lambda: tmp_path / "vp.json")
    monkeypatch.setattr(enroll, "available", lambda: True)
    # A microphone that never delivers anything, the deaf-hardware case.
    monkeypatch.setattr(enroll, "Microphone", lambda **kw: FakeMic())

    assert enroll.run(cfg, prompts=["say something"]) == 1
    assert not (tmp_path / "vp.json").exists()


def test_enrolment_stops_if_the_model_cannot_be_fetched(tmp_path, monkeypatch):
    """Recording four sentences and only then discovering there is nothing to
    process them with would waste the one interaction this feature asks for."""
    from vesper import config as config_module
    from vesper import enroll

    cfg = config_module.Config()
    monkeypatch.setattr(cfg, "voiceprint_path", lambda: tmp_path / "vp.json")
    monkeypatch.setattr(enroll, "available", lambda: False)
    monkeypatch.setattr(enroll, "model_path", lambda download=False: None)

    recorded = []
    monkeypatch.setattr(enroll, "Microphone",
                        lambda **kw: recorded.append("opened") or FakeMic())

    assert enroll.run(cfg, prompts=["say something"]) == 1
    assert recorded == [], "opened the microphone before checking for the model"


# --- the dashboard ----------------------------------------------------------


def test_the_status_snapshot_has_everything_the_dashboard_reads():
    """The two are wired by name across a thread boundary, so a renamed field
    would show a permanent zero rather than raise anything."""
    conversation, speaker, _ = _conversation()
    status = conversation.status()
    speaker.close()

    for key in ("paused", "uptime_s", "turns", "cost_usd", "interruptions",
                "approvals", "refusals", "undos", "voice_rejections",
                "echo_rejections", "last_heard", "pending"):
        assert key in status, f"the dashboard reads {key} and status omits it"


def test_the_status_snapshot_is_plain_data():
    """It crosses a thread boundary once a second. Handing over anything live
    would mean the dashboard reading the audio loop's state as it mutates."""
    conversation, speaker, _ = _conversation()
    status = conversation.status()
    speaker.close()

    for key, value in status.items():
        assert isinstance(value, (int, float, str, bool)), f"{key} is {type(value)}"


def test_the_dashboard_never_touches_tk_from_another_thread():
    """The rule that had to be learned the hard way. Calling any Tk method
    cross-thread, `after()` included, makes Tcl call Tcl_Panic, which aborts
    the process outright rather than raising something catchable."""
    import inspect

    from vesper.ui import dashboard

    for name in ("show", "close"):
        source = inspect.getsource(getattr(dashboard.Dashboard, name))
        assert "_root." not in source, f"{name} touches Tk from the calling thread"
        assert ".after(" not in source, f"{name} schedules on Tk from outside"


def test_a_dashboard_that_cannot_redraw_says_so_once():
    """A failed redraw was swallowed whole, so the window froze on its last
    good frame and nothing anywhere said why. Reporting every tick would be no
    better: four times a second, the log becomes its own outage."""
    from vesper.ui import dashboard as dashboard_module

    said: list[str] = []

    class Root:
        def after(self, *_args):
            pass

    def broken_snapshot():
        raise RuntimeError("snapshot is broken")

    board = dashboard_module.Dashboard(broken_snapshot, on_error=said.append)
    board._root = Root()
    # Only one tick in four redraws, so eight ticks is two attempts.
    for _ in range(8):
        board._refresh()

    assert board._apply_failures == 2, "the tick stopped running"
    assert len(said) == 1, "it repeated itself instead of counting"
    assert "snapshot is broken" in said[0]


def test_a_dashboard_that_recovers_keeps_running():
    """The failure counter must not be a latch: one bad snapshot should not
    stop every later one from being drawn."""
    from vesper.ui import dashboard as dashboard_module

    drawn: list[dict] = []
    fail = [True]

    class Root:
        def after(self, *_args):
            pass

    def snapshot():
        if fail[0]:
            raise RuntimeError("not yet")
        return {"state": "listening"}

    board = dashboard_module.Dashboard(snapshot, on_error=lambda _m: None)
    board._root = Root()
    board._apply = drawn.append
    board._refresh()
    assert board._apply_failures == 1 and drawn == []

    fail[0] = False
    for _ in range(4):
        board._refresh()
    assert drawn == [{"state": "listening"}]


def test_closing_the_dashboard_waits_for_its_thread():
    """A daemon thread still holding Tk objects at process exit is the same
    wrong-thread teardown, just later."""
    import inspect

    from vesper.ui import dashboard

    assert "join" in inspect.getsource(dashboard.Dashboard.close)


# --- the icon ---------------------------------------------------------------


def test_the_icon_is_a_file_windows_will_actually_load(tmp_path):
    """Written by hand, so the failure mode is a structurally invalid .ico that
    Windows silently ignores, leaving no tray icon and therefore no way to quit."""
    from vesper.ui import icon

    icons = icon.ensure(tmp_path)
    assert set(icons) == {"listening", "paused", "working"}
    for path in icons.values():
        assert path.exists() and path.stat().st_size > 1000
        header = path.read_bytes()[:6]
        assert header[:4] == b"\x00\x00\x01\x00", "not an ICO header"


def test_the_states_are_visibly_different():
    """The icon exists so a glance tells you whether it is listening. Two states
    that render alike would defeat the entire point of drawing one."""
    from vesper.ui import icon

    rendered = {
        state: icon._render(32, *colours) for state, colours in icon.STATES.items()
    }
    listening = rendered["listening"].astype(int)
    paused = rendered["paused"].astype(int)
    assert np.abs(listening - paused).mean() > 8, "listening and paused look alike"


def test_the_small_icon_keeps_its_shape():
    """Drawn at 128 and shrunk. The first version used a needle-thin star that
    looked right large and turned into a grey smudge at the 16 pixels you
    actually see, so the star is deliberately blunter when small."""
    from vesper.ui import icon

    small = icon._render(16, *icon.LISTENING)
    star = small[..., :3].astype(int).sum(axis=2)
    lit = (star > star.mean()).sum()
    assert 20 <= lit <= 200, f"{lit} lit pixels of 256 is not a legible mark"


def test_the_log_records_what_it_heard_and_what_it_threw_away(tmp_path):
    """The only thing that can answer "I said the wake word and nothing
    happened". Without it there is no way to tell a microphone that heard
    nothing from a transcription that came out wrong from a wake gate that
    said no, and running from login there is no terminal to watch."""
    log = LogFile(tmp_path / "vesper.log")
    ui = RecordingUI()
    attach(ui, log)

    ui.heard("hey vesper what time is it", True)
    ui.heard("something said to a colleague", False)
    ui.discarded("heard myself")

    written = (tmp_path / "vesper.log").read_text(encoding="utf-8")
    assert "hey vesper what time is it" in written
    assert "said to a colleague" in written
    assert "heard myself" in written
    # And the terminal still behaves exactly as it did.
    assert ui.heard_lines[0] == ("hey vesper what time is it", True)
    assert ui.discards == ["heard myself"]


# --- not hearing itself -----------------------------------------------------


def test_the_microphone_stays_deaf_through_the_tail_of_its_own_voice():
    """`speaking` clears when the last block is written to the sound card, not
    when the last sound leaves the speaker. Unmuting on that edge hands the
    endpointer the final syllable of Vesper's own voice, which is what made it
    cut itself off on nearly every reply and then reject the result as "not
    your voice"."""
    conversation, speaker, _ = _conversation()
    conversation.config.half_duplex = True
    conversation.config.tail_mute_ms = 300

    loud = np.full(480, 0.4, dtype=np.float32)

    # While speaking: ignored, and the tail clock starts.
    speaker._speaking.set()
    conversation._handle_block(loud)
    speaker._speaking.clear()

    # Immediately after: still its own voice in the air, still ignored.
    assert conversation._in_speech_tail()
    conversation._handle_block(loud)
    assert conversation.turns == 0

    # Once the tail has passed, the microphone counts again.
    conversation._spoke_until -= 1.0
    assert not conversation._in_speech_tail()
    speaker.close()


def test_full_duplex_still_allows_being_talked_over():
    """Half duplex is the default because it has to be, not because barge-in
    stopped mattering. On headphones it should still work."""
    conversation, speaker, _ = _conversation()
    conversation.config.half_duplex = False
    conversation._spoke_until = 0.0

    assert not conversation._in_speech_tail()
    speaker.close()


def test_nothing_is_muted_when_it_has_never_spoken():
    """A fresh start must not begin deaf."""
    conversation, speaker, _ = _conversation()
    assert not conversation._in_speech_tail()
    speaker.close()


# --- one at a time ----------------------------------------------------------


def test_only_one_vesper_can_hold_the_lock():
    """Two copies both hold the microphone, so every utterance is transcribed,
    answered and spoken twice, over each other. With autostart on, a second
    copy is the expected accident: the shortcut runs at login and starting it
    by hand is the obvious move when it seems unresponsive. It happened."""
    from vesper import single

    # Its own mutex name, so the result does not depend on whether a real
    # Vesper happens to be running on this machine right now.
    name = r"Global\VesperTestLock"
    assert single.claim(name) is True
    try:
        assert single.claim(name) is True, "the holder must be able to re-claim"
    finally:
        single.release()


def test_releasing_a_lock_that_was_never_taken_is_harmless():
    from vesper import single

    single.release()
    single.release()


def test_a_short_reply_is_never_rejected_as_the_wrong_voice(tmp_path):
    """A real "Sure." scored 0.22 against its own owner and was thrown away.
    "Sure" is how you approve things, so that reading turned a safety feature
    into a way of silently ignoring consent."""
    from vesper.stt import voiceprint as vp

    store = vp.VoicePrint(tmp_path / "vp.json")
    if not vp.available():
        pytest.skip("speaker model not downloaded")
    store.enrol([_fixture(f"speaker_a_{i}.wav") for i in (1, 2)])

    # A different speaker, but only a moment of them.
    brief = _fixture("speaker_b_1.wav")[: int(16_000 * 1.0)]
    verdict, _ = store.compare(brief)
    assert verdict != vp.DIFFERENT, "rejected a clip too short to judge"

    # Given enough of them, it still rejects.
    full = _fixture("speaker_b_1.wav")
    assert store.compare(full)[0] == vp.DIFFERENT
