"""The voice picker: the adapter, and the threading rules it must not break.

Tk is never instantiated in this suite, so the picker is tested in two halves.
`VoicePanel` is a plain object and gets driven directly. The dashboard's half
is checked by source inspection, the same way `test_unattended.py` already
checks that `show` and `close` never touch Tk from the calling thread. That
looks odd until you remember why it exists: a cross-thread Tk call does not
raise, it calls Tcl_Panic and aborts the process, so there is nothing for a
normal test to catch and the mistake has to be caught before it runs.
"""

import inspect
import threading

import pytest

from vesper.tts import eleven_api
from vesper.tts.budget import Budget
from vesper.tts.cache import AudioCache
from vesper.ui import dashboard as dashboard_module
from vesper.ui.voicepanel import SAMPLE_LINE, VoicePanel


class FakeCloudVoice:
    """Stands in for ElevenTTS: records what it was asked to say and why."""

    def __init__(self, *, reason="cloud", cap=9_000, spent=0):
        self.client = self
        self.voice_id = eleven_api.DEFAULT_VOICE_ID
        self.budget = Budget(None, cap=cap)
        self.budget.spend(spent)
        self.cache = AudioCache(None)
        self.last_reason = reason
        self.said: list[tuple[str, str]] = []
        self.name = "eleven:George"
        self._reason = reason

    # the client half
    def list_voices(self):
        return eleven_api.KNOWN_VOICES

    # the voice half
    def use(self, voice_id):
        self.voice_id = voice_id

    def say_as(self, text, voice_id, stop):
        self.said.append((text, voice_id))
        self.last_reason = self._reason
        if self._reason == "cloud":
            self.budget.spend(len(text))


# --- the adapter ------------------------------------------------------------


def test_the_picker_offers_the_verified_voice_list():
    panel = VoicePanel(FakeCloudVoice())
    assert panel.list() == list(eleven_api.KNOWN_VOICES)


def test_the_list_is_fetched_once_rather_than_on_every_open():
    voice = FakeCloudVoice()
    calls = []
    voice.list_voices = lambda: calls.append(1) or eleven_api.KNOWN_VOICES

    panel = VoicePanel(voice)
    panel.list()
    panel.list()

    assert len(calls) == 1


def test_a_failing_catalogue_still_gives_the_picker_something_to_show():
    voice = FakeCloudVoice()

    def boom():
        raise RuntimeError("no")

    voice.list_voices = boom

    assert VoicePanel(voice).list() == list(eleven_api.KNOWN_VOICES)


def test_choosing_a_voice_switches_it_and_remembers_it(tmp_path):
    from vesper import voicechoice

    voice = FakeCloudVoice()
    path = tmp_path / "choice.json"
    panel = VoicePanel(voice, choice_path=path)

    panel.choose("onwK4e9ZLuTAKqWW03F9")

    assert voice.voice_id == "onwK4e9ZLuTAKqWW03F9"
    assert voicechoice.load(path).voice_id == "onwK4e9ZLuTAKqWW03F9"
    assert voicechoice.load(path).name == "Daniel"


def test_a_preview_speaks_the_sample_in_the_candidate_not_the_current_voice():
    """The point of a preview is hearing it before committing."""
    voice = FakeCloudVoice()
    panel = VoicePanel(voice)

    panel.preview("onwK4e9ZLuTAKqWW03F9")

    assert voice.said == [(SAMPLE_LINE, "onwK4e9ZLuTAKqWW03F9")]
    assert voice.voice_id == eleven_api.DEFAULT_VOICE_ID, "a preview switched voice"


def test_a_preview_reports_what_it_cost():
    panel = VoicePanel(FakeCloudVoice())
    note = panel.preview(eleven_api.DEFAULT_VOICE_ID)
    assert str(len(SAMPLE_LINE)) in note and "left this month" in note


def test_a_cached_preview_says_it_was_free():
    panel = VoicePanel(FakeCloudVoice(reason="cache"))
    assert "no charge" in panel.preview(eleven_api.DEFAULT_VOICE_ID)


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("no-key", "no ElevenLabs key"),
        ("budget", "allowance spent"),
        ("unauthorized", "refused"),
        ("paid-voice", "paid plan"),
        ("timeout", "did not answer"),
        ("network", "no connection"),
    ],
)
def test_a_failed_preview_explains_itself_in_words(reason, expected):
    """"spoke locally (network)" in a window is a bug report, not an answer."""
    panel = VoicePanel(FakeCloudVoice(reason=reason))
    assert expected in panel.preview(eleven_api.DEFAULT_VOICE_ID)


def test_the_budget_is_reported_as_two_plain_numbers():
    panel = VoicePanel(FakeCloudVoice(cap=9_000, spent=1_234))
    assert panel.budget() == (1_234, 9_000)


# --- the rules the dashboard must keep --------------------------------------


def test_the_preview_never_runs_on_the_tk_thread():
    """A preview is an HTTP call with a 20 second timeout. Running it inline
    would freeze the window for that long, and a frozen window looks crashed."""
    source = inspect.getsource(dashboard_module.Dashboard._preview)
    assert "Thread(" in source, "the preview blocks the UI thread"
    # Constructing a Thread and never starting it leaves "Thread(" in the
    # source while the feature is permanently dead. The first version of this
    # test could not tell the difference.
    assert ".start()" in source, "the worker thread is created but never run"
    assert "self.voices.preview" not in source, "the preview is called inline"


def test_the_preview_worker_never_touches_tk():
    """The worker runs off the Tk thread, so one `.config()` in here would
    abort the process rather than raise."""
    source = inspect.getsource(dashboard_module.Dashboard._preview_worker)
    for forbidden in ("_root", ".after(", ".config(", "_fields["):
        assert forbidden not in source, f"the preview worker touches Tk: {forbidden}"


def test_every_voice_widget_is_registered_for_teardown():
    """`_run`'s finally clears `_fields` on the owning thread. A widget or a
    StringVar left anywhere else is finalised later by the main thread, which
    is the exact Tcl_Panic this class is built to avoid."""
    import re

    source = inspect.getsource(dashboard_module.Dashboard._voice_section)

    # Every construction bound to a name, and every one of them must either be
    # registered or be a container. The first version computed a `created`
    # total and then never compared it to anything, so the only live check was
    # a hardcoded ">= 4" with slack to spare: turning a registered widget into
    # a bare local left it green while reintroducing the abort.
    constructed = re.findall(
        r"^\s*([A-Za-z_][\w\[\]\"'.]*)\s*=\s*(?:tk\.\w+\(|self\._button\()",
        source,
        re.M,
    )
    assert constructed, "the parser stopped matching; fix it, not the code"

    # Containers can be locals: root.destroy() walks the widget tree and takes
    # its children with it. A StringVar is not in that tree, which is the whole
    # reason the registration rule exists.
    containers = {"panel", "row", "track", "fill", "listbox"}
    unregistered = [
        name for name in constructed
        if not name.startswith("self._fields[") and name not in containers
    ]
    assert unregistered == [], (
        f"{unregistered} are held outside _fields, where _run's finally cannot "
        f"clear them on the owning thread"
    )

    assert "tk.OptionMenu(" not in source, (
        "OptionMenu builds a Menu, and the second Tk interpreter on a new "
        "thread cannot create one: Tcl_Panic, exit code 3, no traceback. "
        "scripts/dashboard_soak.py is what caught it."
    )
    assert "self._voice_var" not in source, (
        "a Tk variable was stashed on self, where _fields.clear() cannot reach it"
    )


def test_the_dashboard_works_with_no_voice_panel_at_all():
    """Piper-only users must not get a dead control."""
    board = dashboard_module.Dashboard(lambda: {}, voices=None)
    assert board.voices is None
    # _apply must not reach the voice branch, which would raise on None.
    source = inspect.getsource(dashboard_module.Dashboard._apply)
    assert "if self.voices is not None:" in source


def test_the_status_snapshot_still_only_holds_scalars():
    """The dashboard reads this across a thread boundary once a second."""
    import numpy as np

    from vesper.audio.speaker import Speaker
    from vesper.conversation import Conversation, ConversationConfig
    from vesper.wake import WakeConfig, WakeGate

    from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI

    speaker = Speaker(FakeVoice())
    conversation = Conversation(
        brain=FakeBrain([]), stt=FakeSTT([]), speaker=speaker, mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(greet_on_start=False), ui=RecordingUI(),
    )
    try:
        status = conversation.status()
    finally:
        speaker.close()

    assert "voice" in status
    for key, value in status.items():
        assert isinstance(value, (int, float, str, bool)), f"{key} is {type(value)}"
    assert np is not None
