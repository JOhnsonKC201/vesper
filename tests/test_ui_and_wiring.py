"""Terminal output, microphone capture, and application wiring.

The microphone tests install a fake `sounddevice` module rather than requiring
hardware, using the same idea as Echo Flow's conftest: stub the native library,
keep the logic under test real. That matters here because the two most valuable
behaviours in mic.py are both failure paths that are awkward to trigger with a
working device.
"""

import io
import sys
import threading
import types

import numpy as np
import pytest
from rich.console import Console

from vesper import config as config_module
from vesper import main as main_module
from vesper.audio.mic import BLOCK_FRAMES, Microphone
from vesper.brain.protocol import TurnComplete
from vesper.tts.base import NullVoice
from vesper.tts.piper_voice import PiperTTS
from vesper.tts.sapi import SapiTTS
from vesper.ui.tui import TerminalUI


# --- terminal ui ------------------------------------------------------------


@pytest.fixture
def ui_and_output():
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, highlight=False)
    return TerminalUI(console=console), buffer


def test_banner_names_every_moving_part(ui_and_output):
    ui, out = ui_and_output
    ui.banner(brain="sonnet", voice="alan", ears="base.en", wake="vesper", cwd="C:/x")
    text = out.getvalue()
    for expected in ("Vesper", "sonnet", "alan", "base.en", "vesper", "C:/x"):
        assert expected in text


def test_what_you_said_is_shown(ui_and_output):
    ui, out = ui_and_output
    ui.heard("what time is it", addressed=True)
    assert "what time is it" in out.getvalue()


def test_unaddressed_speech_is_marked_as_such(ui_and_output):
    ui, out = ui_and_output
    ui.heard("talking to someone else", addressed=False)
    assert "not addressed" in out.getvalue()


def test_screen_blocks_are_printed(ui_and_output):
    ui, out = ui_and_output
    ui.screen("main.py:40  raise ValueError")
    assert "main.py:40" in out.getvalue()


def test_empty_screen_block_prints_nothing(ui_and_output):
    ui, out = ui_and_output
    ui.screen("   ")
    assert out.getvalue().strip() == ""


def test_tool_calls_are_shown_with_their_command(ui_and_output):
    ui, out = ui_and_output
    ui.tool("Bash", "git status")
    text = out.getvalue()
    assert "Bash" in text and "git status" in text


def test_permission_requests_stand_out(ui_and_output):
    ui, out = ui_and_output
    ui.permission("Write", "C:/notes.txt")
    text = out.getvalue()
    assert "needs your ok" in text and "Write" in text


def test_cost_and_latency_are_reported(ui_and_output):
    ui, out = ui_and_output
    ui.answered(TurnComplete(text="x", cost_usd=0.008, turns=2), 3.2, 1.6)
    text = out.getvalue()
    assert "1.6s to first word" in text
    assert "3.2s total" in text
    assert "$0.008" in text


def test_cost_can_be_suppressed():
    buffer = io.StringIO()
    ui = TerminalUI(show_cost=False, console=Console(file=buffer, width=100))
    ui.answered(TurnComplete(text="x", cost_usd=0.5), 1.0, 0.5)
    assert buffer.getvalue().strip() == ""


def test_session_cost_accumulates(ui_and_output):
    ui, _ = ui_and_output
    ui.answered(TurnComplete(text="a", cost_usd=0.01), 1.0, 0.5)
    ui.answered(TurnComplete(text="b", cost_usd=0.02), 1.0, 0.5)
    assert ui.session_cost == pytest.approx(0.03)
    assert ui.turns == 2


def test_routine_discards_are_not_printed(ui_and_output):
    """Noise gating happens constantly; narrating it would drown the transcript."""
    ui, out = ui_and_output
    ui.discarded("too-quiet")
    assert out.getvalue().strip() == ""


def test_hearing_itself_is_worth_showing(ui_and_output):
    ui, out = ui_and_output
    ui.discarded("heard myself")
    assert "heard myself" in out.getvalue()


def test_ui_is_safe_from_several_threads(ui_and_output):
    ui, out = ui_and_output
    threads = [
        threading.Thread(target=lambda n=n: ui.spoke(f"line {n}")) for n in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert out.getvalue().count("line") == 20


# --- microphone -------------------------------------------------------------


class FakeStream:
    def __init__(self, *, fail_on_start=False, fail_on_stop=False, **kwargs):
        self.kwargs = kwargs
        self.fail_on_start = fail_on_start
        self.fail_on_stop = fail_on_stop
        self.started = self.stopped = self.closed = False

    def start(self):
        if self.fail_on_start:
            raise RuntimeError("device busy")
        self.started = True

    def stop(self):
        if self.fail_on_stop:
            raise RuntimeError("device removed")
        self.stopped = True

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sounddevice(monkeypatch):
    """Install a stand-in sounddevice so mic logic runs without hardware."""
    module = types.ModuleType("sounddevice")
    module.streams = []
    module.fail_on_start = False
    module.fail_on_stop = False

    def InputStream(**kwargs):
        stream = FakeStream(
            fail_on_start=module.fail_on_start,
            fail_on_stop=module.fail_on_stop,
            **kwargs,
        )
        module.streams.append(stream)
        return stream

    module.InputStream = InputStream
    module.query_devices = lambda: [
        {"name": "Fake Mic", "max_input_channels": 1},
        {"name": "Speakers", "max_input_channels": 0},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def test_microphone_opens_a_16k_mono_float_stream(fake_sounddevice):
    mic = Microphone()
    mic.start()
    assert mic.running
    kwargs = fake_sounddevice.streams[0].kwargs
    assert kwargs["samplerate"] == 16000
    assert kwargs["channels"] == 1
    assert kwargs["dtype"] == "float32"
    assert kwargs["blocksize"] == BLOCK_FRAMES
    mic.stop()


def test_a_failed_start_closes_the_stream(fake_sounddevice):
    """sounddevice has no __del__; a dropped handle orphans the device."""
    fake_sounddevice.fail_on_start = True
    mic = Microphone()
    with pytest.raises(RuntimeError):
        mic.start()
    assert fake_sounddevice.streams[0].closed is True
    assert mic.running is False


def test_a_failing_stop_still_clears_the_running_flag(fake_sounddevice):
    """Otherwise every later start() silently early-returns, forever."""
    errors = []
    mic = Microphone(on_error=errors.append)
    mic.start()
    fake_sounddevice.streams[0].fail_on_stop = True
    mic.stop()
    assert mic.running is False
    assert len(errors) == 1


def test_starting_twice_opens_one_stream(fake_sounddevice):
    mic = Microphone()
    mic.start()
    mic.start()
    assert len(fake_sounddevice.streams) == 1
    mic.stop()


def test_context_manager_starts_and_stops(fake_sounddevice):
    with Microphone() as mic:
        assert mic.running
    assert mic.running is False


def test_blocks_are_readable_in_order(fake_sounddevice):
    mic = Microphone()
    for value in (0.1, 0.2, 0.3):
        mic._callback(np.full((BLOCK_FRAMES, 1), value, dtype=np.float32), 0, None, None)
    assert mic.read(timeout=0.1)[0] == pytest.approx(0.1)
    assert mic.read(timeout=0.1)[0] == pytest.approx(0.2)
    assert mic.read(timeout=0.1)[0] == pytest.approx(0.3)


def test_read_times_out_to_none_rather_than_blocking(fake_sounddevice):
    assert Microphone().read(timeout=0.01) is None


def test_preroll_keeps_recent_audio(fake_sounddevice):
    """By the time speech is detected the first syllable is already past."""
    mic = Microphone(preroll_ms=400)
    for _ in range(20):
        mic._callback(np.full((BLOCK_FRAMES, 1), 0.5, dtype=np.float32), 0, None, None)
    preroll = mic.preroll()
    assert 0 < len(preroll) <= 16000 * 0.5
    assert np.allclose(preroll, 0.5)


def test_preroll_is_empty_before_any_audio(fake_sounddevice):
    assert Microphone().preroll().size == 0


def test_drain_discards_buffered_audio(fake_sounddevice):
    mic = Microphone()
    for _ in range(5):
        mic._callback(np.zeros((BLOCK_FRAMES, 1), dtype=np.float32), 0, None, None)
    mic.drain()
    assert mic.read(timeout=0.01) is None
    assert mic.preroll().size == 0


def test_a_full_queue_drops_old_audio_instead_of_blocking(fake_sounddevice):
    """Blocking inside the PortAudio callback glitches audio system wide."""
    mic = Microphone()
    block = np.zeros((BLOCK_FRAMES, 1), dtype=np.float32)
    for _ in range(400):
        mic._callback(block, 0, None, None)  # queue caps at 256
    assert mic.read(timeout=0.1) is not None


def test_callback_status_is_reported_not_raised(fake_sounddevice):
    errors = []
    mic = Microphone(on_error=errors.append)
    mic._callback(np.zeros((BLOCK_FRAMES, 1), dtype=np.float32), 0, None, "overflow")
    assert len(errors) == 1


def test_device_listing_returns_only_inputs(fake_sounddevice):
    devices = Microphone.list_devices()
    assert [d["name"] for d in devices] == ["Fake Mic"]


# --- voice backends ---------------------------------------------------------


def test_find_voice_prefers_the_configured_model(tmp_path):
    (tmp_path / "en_US-other-medium.onnx").touch()
    (tmp_path / "en_GB-alan-medium.onnx").touch()
    found = PiperTTS.find_voice(tmp_path, "en_GB-alan-medium")
    assert found.stem == "en_GB-alan-medium"


def test_find_voice_falls_back_to_any_model(tmp_path):
    (tmp_path / "some-voice.onnx").touch()
    assert PiperTTS.find_voice(tmp_path, "not-installed").stem == "some-voice"


def test_find_voice_returns_none_when_empty(tmp_path):
    assert PiperTTS.find_voice(tmp_path, "anything") is None
    assert PiperTTS.find_voice(tmp_path / "missing", "anything") is None


def test_missing_model_file_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        PiperTTS(tmp_path / "nope.onnx")


def test_sapi_is_available_on_this_machine():
    assert SapiTTS.available() is True
    assert len(SapiTTS.list_voices()) > 0


def test_null_voice_records_instead_of_speaking():
    voice = NullVoice()
    voice.speak("hello", threading.Event())
    assert voice.spoken == ["hello"]
    voice.close()


# --- wiring -----------------------------------------------------------------


def test_engine_none_selects_the_silent_voice():
    cfg = config_module.Config()
    cfg.voice.engine = "none"
    voice, label = main_module.build_voice(cfg, TerminalUI(console=Console(file=io.StringIO())))
    assert isinstance(voice, NullVoice)
    assert label == "silent"


def test_missing_piper_model_falls_back_to_sapi(tmp_path):
    cfg = config_module.Config()
    cfg.voice.engine = "piper"
    cfg.voice.voices_dir = str(tmp_path)
    buffer = io.StringIO()
    ui = TerminalUI(console=Console(file=buffer, width=120))
    voice, label = main_module.build_voice(cfg, ui)
    assert label == "windows sapi"
    assert "download_voices" in buffer.getvalue(), "must say how to fix it"


def test_piper_is_selected_when_the_model_is_present():
    """The name says "when the model is present", but nothing checked that.

    The model is a 63MB download living under the gitignored `var/`, so on any
    machine that has not run `python -m piper.download_voices` this failed
    rather than skipping. That is a broken test, not a caught bug, and it is
    what would have made the first CI run red.
    """
    cfg = config_module.Config()
    from vesper.tts.piper_voice import PiperTTS

    if PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model) is None:
        pytest.skip("no piper voice downloaded; run python -m piper.download_voices")

    voice, label = main_module.build_voice(
        cfg, TerminalUI(console=Console(file=io.StringIO()))
    )
    assert "piper" in label
    voice.close()


def test_self_check_passes_on_this_machine(capsys):
    assert main_module.check(config_module.Config()) == 0
    output = capsys.readouterr().out
    assert "claude cli" in output
    assert "FAIL" not in output


def test_devices_flag_lists_microphones(capsys):
    assert main_module.main(["--devices"]) == 0
    assert "input devices" in capsys.readouterr().out


def test_no_voice_flag_silences_the_assistant(monkeypatch):
    seen = {}

    def fake_run_text(cfg, one_shot=""):
        seen["engine"] = cfg.voice.engine
        return 0

    monkeypatch.setattr(main_module, "run_text", fake_run_text)
    main_module.main(["--no-voice", "--say", "hello"])
    assert seen["engine"] == "none"
