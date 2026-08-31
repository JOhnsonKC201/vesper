"""Changing voice without editing a file.

Vesper shipped with a voice picker that only appeared when the ElevenLabs cloud
voice was running, and the default install is Piper, so the setting was not
there. Worse, in the one build that did show it, the picker was dead: the
dashboard read the selection from a `voice_var` StringVar that an earlier
refactor to a Listbox had deleted, so clicking a voice returned None and did
nothing at all.

These cover the catalogue of what a machine can speak in, the panel that
switches between them, and the two places the old picker was broken.

No Tk here. The dashboard's own rules are checked the way `test_voicepanel.py`
already checks them, by driving the methods with stand-in widgets, because a
cross-thread Tk call aborts the process rather than raising.
"""

from __future__ import annotations

import threading
import types

import pytest

from vesper.tts import catalog, shaping


def _models(directory, *stems):
    for stem in stems:
        (directory / f"{stem}.onnx").write_bytes(b"not really a model")
    return directory


# --- the catalogue ----------------------------------------------------------


def test_the_deliveries_offered_are_the_ones_that_exist():
    """A preset added to shaping and forgotten here would be unreachable."""
    assert set(catalog.DELIVERIES) == set(shaping.PRESETS)


def test_every_model_is_listed_once_per_delivery(tmp_path):
    """The delivery is half the sound. Offering the model alone would hide the
    control that changes it most."""
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))
    assert [v.character for v in voices] == list(catalog.DELIVERIES)


def test_a_file_name_reads_as_a_person_and_a_place(tmp_path):
    voice = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))[0]
    assert voice.name == "Alan"
    assert voice.label == "Alan  British, jarvis"


def test_two_qualities_of_one_voice_do_not_read_alike(tmp_path):
    """Otherwise the list has two identical rows and no way to tell them apart."""
    labels = {v.label for v in catalog.piper_voices(
        _models(tmp_path, "en_US-ryan-medium", "en_US-ryan-high"))}
    assert len(labels) == 2 * len(catalog.DELIVERIES)
    assert "Ryan  American high, jarvis" in labels


def test_an_id_carries_the_model_and_the_delivery_together(tmp_path):
    """One id, because they are one choice: restoring it restores how it sounded."""
    voice = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))[1]
    assert voice.voice_id == "en_GB-alan-medium:natural"
    assert catalog.split_piper_id(voice.voice_id) == ("en_GB-alan-medium", "natural")


def test_a_bare_model_name_reads_as_the_default_delivery():
    assert catalog.split_piper_id("en_GB-alan-medium")[1] == catalog.DELIVERIES[0]


def test_no_voices_directory_is_not_an_error(tmp_path):
    """A fresh install before the first download. The picker shows SAPI only."""
    assert catalog.piper_voices(tmp_path / "never-made") == []


def test_a_windows_voice_reads_as_a_name():
    assert catalog._sapi_name("Microsoft David Desktop - English (United States)") == "David"
    assert catalog._sapi_name("Some Other Engine") == "Some Other Engine"


def test_the_running_voice_can_be_recognised_in_the_list(tmp_path):
    """So the picker opens with the current voice selected rather than the first."""
    live = types.SimpleNamespace(
        name="en_GB-alan-medium",
        character=types.SimpleNamespace(name="broadcast"),
    )
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))
    assert catalog.find(voices, catalog.current_id(live)).character == "broadcast"


def test_a_cloud_voice_is_not_mistaken_for_a_local_one():
    """ElevenTTS has a name too, and no place in this list."""
    assert catalog.current_id(types.SimpleNamespace(name="eleven:George")) == ""


def test_building_a_voice_loads_the_model_the_id_names(tmp_path, monkeypatch):
    from vesper.tts import piper_voice

    built = {}

    class FakePiper:
        def __init__(self, path, *, speed, volume, character):
            built.update(path=path, speed=speed, character=character.name)

    monkeypatch.setattr(piper_voice, "PiperTTS", FakePiper)
    record = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))[1]
    catalog.build(record, voices_dir=tmp_path, speed=1.15, volume=0.9)

    assert built["path"] == tmp_path / "en_GB-alan-medium.onnx"
    assert built["speed"] == 1.15
    assert built["character"] == "natural"


# --- the panel --------------------------------------------------------------


class FakeSpeaker:
    """Enough Speaker to switch a voice on: what it is using, and the swap."""

    def __init__(self, voice=None):
        self.voice = voice or types.SimpleNamespace(name="none")
        self.waited = []

    def use_voice(self, voice):
        self.voice = voice

    def wait_until_idle(self, timeout=None):
        self.waited.append(timeout)
        return True


class FakeLocal:
    """A backend that records what it was asked to say, and whether it was shut."""

    def __init__(self, character="jarvis"):
        self.name = "en_GB-alan-medium"
        self.character = types.SimpleNamespace(name=character)
        self.said: list[str] = []
        self.closed = False

    def speak(self, text, stop):
        self.said.append(text)

    def close(self):
        self.closed = True


def _panel(tmp_path, speaker=None, **kwargs):
    from vesper.ui.voicepanel import LocalVoicePanel

    _models(tmp_path, "en_GB-alan-medium")
    panel = LocalVoicePanel(speaker or FakeSpeaker(), voices_dir=tmp_path, **kwargs)
    # Piper voices only. SAPI is a COM call, and what is installed on the
    # machine running the tests is not this file's business.
    panel._catalogue = list(catalog.piper_voices(tmp_path))
    return panel


def test_choosing_a_local_voice_hands_it_to_the_speaker(tmp_path, monkeypatch):
    """A local backend cannot be re-pointed the way the cloud one can. It holds
    a loaded model, so switching means building one and swapping it in."""
    speaker = FakeSpeaker()
    panel = _panel(tmp_path, speaker)
    made = FakeLocal("natural")
    monkeypatch.setattr(panel, "_build", lambda record: made)

    panel.choose("en_GB-alan-medium:natural")

    assert speaker.voice is made, "the running assistant kept its old voice"


def test_a_local_choice_survives_a_restart(tmp_path, monkeypatch):
    """Including which delivery, which is the half that changes it most."""
    from vesper import voicechoice

    path = tmp_path / "choice.json"
    panel = _panel(tmp_path, choice_path=path)
    monkeypatch.setattr(panel, "_build", lambda record: FakeLocal())

    panel.choose("en_GB-alan-medium:broadcast")

    saved = voicechoice.load(path)
    assert saved.voice_id == "en_GB-alan-medium:broadcast"
    assert saved.engine == "piper" and saved.local
    assert saved.name == "Alan"


def test_an_id_the_catalogue_does_not_know_is_refused(tmp_path):
    """Ids come back from a window. Silently doing nothing would look like a
    switch that worked."""
    with pytest.raises(LookupError):
        _panel(tmp_path).choose("en_GB-nobody-medium:jarvis")


def test_a_preview_speaks_the_sample_without_switching_to_it(tmp_path, monkeypatch):
    """The point of a preview is hearing a voice before committing to it."""
    from vesper.ui.voicepanel import SAMPLE_LINE

    speaker = FakeSpeaker(FakeLocal())
    panel = _panel(tmp_path, speaker)
    candidate = FakeLocal("broadcast")
    monkeypatch.setattr(panel, "_build", lambda record: candidate)
    before = speaker.voice

    note = panel.preview("en_GB-alan-medium:broadcast")

    assert candidate.said == [SAMPLE_LINE]
    assert speaker.voice is before, "a preview switched voice"
    assert "Alan" in note


def test_a_preview_closes_what_it_built(tmp_path, monkeypatch):
    """Every preview loads a second model and opens a second output stream.
    Leaving them open would leak a device handle per click."""
    panel = _panel(tmp_path)
    candidate = FakeLocal()
    monkeypatch.setattr(panel, "_build", lambda record: candidate)

    panel.preview("en_GB-alan-medium:jarvis")

    assert candidate.closed


def test_a_preview_waits_for_a_reply_in_progress(tmp_path, monkeypatch):
    """Two voices at once is the most broken thing a speech app can do. The
    cloud panel gets this from ElevenTTS's lock; there is none to share here."""
    speaker = FakeSpeaker()
    panel = _panel(tmp_path, speaker)
    monkeypatch.setattr(panel, "_build", lambda record: FakeLocal())

    panel.preview("en_GB-alan-medium:jarvis")

    assert speaker.waited, "it talked over whatever Vesper was saying"


def test_a_cancelled_preview_does_not_start(tmp_path, monkeypatch):
    """Closing the window while one is queued behind a long reply."""
    panel = _panel(tmp_path)
    candidate = FakeLocal()
    monkeypatch.setattr(panel, "_build", lambda record: candidate)
    panel.speaker.wait_until_idle = lambda timeout=None: panel.cancel()

    assert panel.preview("en_GB-alan-medium:jarvis") == "stopped"
    assert candidate.said == []


def test_a_voice_that_will_not_load_says_so_in_words(tmp_path, monkeypatch):
    """"FileNotFoundError" in a status window is a bug report, not an answer."""
    panel = _panel(tmp_path)

    def boom(record):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(panel, "_build", boom)
    assert panel.preview("en_GB-alan-medium:jarvis") == "Alan would not load (FileNotFoundError)"


def test_the_running_voice_is_the_one_the_picker_shows_as_current(tmp_path):
    panel = _panel(tmp_path, FakeSpeaker(FakeLocal("natural")))
    assert panel.current() == "Alan  British, natural"


def test_the_list_is_read_once_rather_than_on_every_open(tmp_path):
    """Enumerating SAPI is a COM call, and the window rebuilds on every open."""
    panel = _panel(tmp_path)
    panel._catalogue = None
    first = panel.list()
    assert panel.list() is first


def test_a_local_voice_has_no_allowance_to_report(tmp_path):
    """Which is what leaves the spend bar off the window entirely."""
    assert _panel(tmp_path).budget() == (0, 0)


def test_the_steady_line_says_how_to_get_more_voices(tmp_path):
    """The one useful thing to tell someone looking at a short list."""
    status = _panel(tmp_path).status()
    assert "piper.download_voices" in status
    assert "network" in status


# --- swapping the voice on a running assistant ------------------------------


def test_the_speaker_can_change_voice_without_a_restart():
    """Which is the whole reason the picker can exist for local backends."""
    from vesper.audio.speaker import Speaker

    from conftest import FakeVoice

    first, second = FakeVoice(), FakeVoice()
    speaker = Speaker(first)
    try:
        speaker.use_voice(second)
        speaker.say("after the swap")
        assert speaker.wait_until_idle(timeout=2.0)
    finally:
        speaker.close()

    assert second.lines == ["after the swap"]
    assert first.lines == []
    assert first.closed, "the old backend kept its output stream open"


def test_swapping_cuts_off_what_is_being_said():
    """Piper holds the whole utterance inside its own lock, so closing the old
    voice mid-sentence would block the caller for as long as the line lasts."""
    from vesper.audio.speaker import Speaker

    from conftest import FakeVoice

    slow = FakeVoice(duration=5.0)
    speaker = Speaker(slow)
    try:
        speaker.say("a long sentence")
        for _ in range(200):
            if speaker.speaking:
                break
            threading.Event().wait(0.01)
        speaker.use_voice(FakeVoice())
        assert speaker.wait_until_idle(timeout=2.0), "the swap waited for the line"
    finally:
        speaker.close()


def test_swapping_to_the_same_voice_leaves_it_alone():
    """A picker that re-selects the current row must not close the voice out
    from under a reply in progress."""
    from vesper.audio.speaker import Speaker

    from conftest import FakeVoice

    voice = FakeVoice()
    speaker = Speaker(voice)
    try:
        speaker.use_voice(voice)
        speaker.use_voice(None)
        assert speaker.voice is voice
        assert not voice.closed, "it closed the voice it was already using"
    finally:
        speaker.close()


# --- the two places the picker was dead -------------------------------------


class FakeListbox:
    """The one method the dashboard asks a Listbox for."""

    def __init__(self, *selected):
        self.selected = selected

    def curselection(self):
        return self.selected


class FakeLabel:
    def __init__(self):
        self.text = ""

    def config(self, **kwargs):
        self.text = kwargs.get("text", self.text)


class RecordingPanel:
    """A voice panel that records what the window asked it to do."""

    def __init__(self, voices=(), cap=0):
        self.voices = list(voices)
        self.chosen: list[str] = []
        self.cap = cap

    def list(self):
        return self.voices

    def current(self):
        return ""

    def choose(self, voice_id):
        self.chosen.append(voice_id)

    def preview(self, voice_id):
        return "previewed"

    def budget(self):
        return 0, self.cap

    def status(self):
        return "the steady line"


def _board(panel, selection):
    from vesper.ui.dashboard import Dashboard

    board = Dashboard(lambda: {}, voices=panel)
    board._fields["voice_list"] = FakeListbox(*selection)
    return board


def _settles(board, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not board._voice_busy.is_set():
            return True
        time.sleep(0.01)
    return False


def test_the_selected_row_is_read_from_the_widget_that_holds_it(tmp_path):
    """The picker was rewritten from an OptionMenu to a Listbox, and this was
    left reading the StringVar the rewrite deleted. It returned None every
    time, so clicking a voice did nothing and Preview did nothing, in the only
    build that showed the picker at all.

    Mutation that fails this: read `_fields["voice_var"]` again.
    """
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))
    board = _board(RecordingPanel(voices), selection=(2,))

    assert board._selected_voice().voice_id == "en_GB-alan-medium:broadcast"


def test_nothing_selected_is_not_a_crash(tmp_path):
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))
    assert _board(RecordingPanel(voices), selection=())._selected_voice() is None
    assert _board(RecordingPanel(voices), selection=(99,))._selected_voice() is None


def test_clicking_a_voice_actually_switches_it(tmp_path):
    """End to end through the window's own handler, minus Tk."""
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))
    panel = RecordingPanel(voices)
    board = _board(panel, selection=(1,))

    board._pick()

    assert _settles(board), "the switch never finished"
    assert panel.chosen == ["en_GB-alan-medium:natural"]


def test_switching_never_runs_on_the_tk_thread():
    """Picking a local voice loads an ONNX model and hands it to the Speaker,
    which is about a second. A window that stops redrawing for a second on
    every click reads as crashed."""
    import inspect

    from vesper.ui import dashboard as dashboard_module

    source = inspect.getsource(dashboard_module.Dashboard._pick)
    assert "Thread(" in source and ".start()" in source
    assert "self.voices.choose" not in source, "the switch is called inline"


def test_the_switch_worker_never_touches_tk():
    import inspect

    from vesper.ui import dashboard as dashboard_module

    source = inspect.getsource(dashboard_module.Dashboard._pick_worker)
    for forbidden in ("_root", ".after(", ".config(", "_fields["):
        assert forbidden not in source, f"the worker touches Tk: {forbidden}"


def test_the_steady_line_comes_from_the_panel_not_the_window():
    """A local voice buys nothing, so "no allowance set" was both true and
    useless. Only the panel knows what is worth saying."""
    board = _board(RecordingPanel(), selection=())
    board._fields["voice_note"] = FakeLabel()

    board._apply_voice()

    assert board._fields["voice_note"].text == "the steady line"


def test_something_that_just_happened_beats_the_steady_line():
    board = _board(RecordingPanel(), selection=())
    board._fields["voice_note"] = FakeLabel()
    board._voice_note = "Alan it is"

    board._apply_voice()

    assert board._fields["voice_note"].text == "Alan it is"


def test_the_spend_bar_is_left_off_when_there_is_nothing_to_spend():
    """A bar pinned at empty forever would be read as a warning about something."""
    import inspect

    from vesper.ui import dashboard as dashboard_module

    source = inspect.getsource(dashboard_module.Dashboard._voice_section)
    assert "if self._cap() > 0:" in source
    assert _board(RecordingPanel(cap=0), selection=())._cap() == 0
    assert _board(RecordingPanel(cap=9_000), selection=())._cap() == 9_000


# --- and it is still there after a restart ----------------------------------


def _terminal_ui():
    import io

    from rich.console import Console

    from vesper.ui.tui import TerminalUI

    return TerminalUI(console=Console(file=io.StringIO()))


def test_a_remembered_local_voice_comes_back_after_a_restart(monkeypatch, tmp_path):
    """Down to the delivery. Restoring the model alone would bring back a voice
    that sounds different from the one that was picked, which reads as the
    setting not having worked.

    Mutation that fails this: pass `cfg.voice.character` unconditionally in
    `build_local_voice`.
    """
    from vesper import config as config_module
    from vesper import main as main_module
    from vesper import voicechoice
    from vesper.tts.piper_voice import PiperTTS

    cfg = config_module.Config()
    if PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model) is None:
        pytest.skip("no piper voice downloaded")

    choice = tmp_path / "choice.json"
    voicechoice.save(choice, f"{cfg.voice.model}:natural", "Alan", engine="piper")
    cfg.voice.character = "jarvis"  # what the file says, and it loses
    monkeypatch.setattr(cfg, "voice_choice_path", lambda: choice)

    voice, _ = main_module.build_voice(cfg, _terminal_ui())
    try:
        assert voice.character.name == "natural", "the click was ignored"
    finally:
        voice.close()


def test_a_remembered_local_voice_is_never_sent_to_elevenlabs(monkeypatch, tmp_path):
    """The two engines share one choice file. A Piper id is not a voice id, and
    posting one would fail every request until the file was deleted by hand."""
    from vesper import config as config_module
    from vesper import main as main_module
    from vesper import voicechoice
    from vesper.tts.piper_voice import PiperTTS

    cfg = config_module.Config()
    if PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model) is None:
        pytest.skip("no piper voice downloaded")

    choice = tmp_path / "choice.json"
    voicechoice.save(choice, f"{cfg.voice.model}:natural", "Alan", engine="piper")

    monkeypatch.delenv("VESPER_ELEVEN_API_KEY", raising=False)
    cfg.voice.engine = "elevenlabs"
    cfg.voice.eleven.api_key = "sk" + "_notreal"
    cfg.voice.eleven.prewarm = False
    monkeypatch.setattr(cfg, "voice_cache_path", lambda: tmp_path / "cache")
    monkeypatch.setattr(cfg, "voice_budget_path", lambda: tmp_path / "b.json")
    monkeypatch.setattr(cfg, "voice_choice_path", lambda: choice)

    voice, _ = main_module.build_voice(cfg, _terminal_ui())
    try:
        assert voice.voice_id == cfg.voice.eleven.voice_id
        assert voice.fallback.character.name == "natural", "the local half ignored it"
    finally:
        voice.close()


def test_a_remembered_windows_voice_comes_back_too(monkeypatch, tmp_path):
    """A SAPI pick has to beat `engine: piper` in the file, or picking David
    would last exactly until the next restart."""
    from vesper import config as config_module
    from vesper import main as main_module
    from vesper import voicechoice
    from vesper.tts.sapi import SapiTTS

    if not SapiTTS.available():
        pytest.skip("no windows speech api")

    choice = tmp_path / "choice.json"
    voicechoice.save(choice, "Microsoft David Desktop", "David", engine="sapi")

    cfg = config_module.Config()
    monkeypatch.setattr(cfg, "voice_choice_path", lambda: choice)

    voice, label = main_module.build_voice(cfg, _terminal_ui())
    try:
        assert label == "windows sapi"
        assert voice.voice_hint == "Microsoft David Desktop"
    finally:
        voice.close()


def test_clicking_the_voice_already_speaking_changes_nothing(tmp_path):
    """Tk's mouse binding fires <<ListboxSelect>> whether or not the selection
    changed, so a click on the highlighted row arrives like any other. Acting
    on it would rebuild the voice already speaking and hand it to the Speaker,
    which barges in: Vesper stops mid-sentence and nothing says why.

    Mutation that fails this: delete the `_is_current` guard from `_pick`.
    """
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))

    class Running(RecordingPanel):
        def current(self):
            return "Alan  British, natural"

    panel = Running(voices)
    board = _board(panel, selection=(1,))  # the row the window selects itself

    board._pick()

    assert _settles(board)
    assert panel.chosen == [], "opening the window switched voice"


def test_a_different_row_still_switches(tmp_path):
    """The guard must not be so broad that the picker stops picking."""
    voices = catalog.piper_voices(_models(tmp_path, "en_GB-alan-medium"))

    class Running(RecordingPanel):
        def current(self):
            return "Alan  British, natural"

    panel = Running(voices)
    board = _board(panel, selection=(2,))

    board._pick()

    assert _settles(board)
    assert panel.chosen == ["en_GB-alan-medium:broadcast"]
