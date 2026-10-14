"""A voice profile from before calibration has to say so, out loud.

The calibration fix (per-clip vectors and a bar measured from your own clips)
only applies to a profile recorded after it. The one on this machine is from
2026-08-31, before it: one averaged centroid and the old 0.55 guess. On
2026-10-13 it scored its owner 0.52, 0.16 and 0.49, "unsure" every time.

The existing warning could not catch that. It waits for 20 scored utterances
in one run and writes one line to the log, and this assistant is launched by
hand for short sessions with no console. So a stale profile is now named at
the greeting, where the person who can fix it hears it, with the one action
that fixes it.
"""

from types import SimpleNamespace

import pytest

from vesper import conversation as conversation_module
from vesper.conversation import ConversationConfig
from test_conversation import build, settle


def _voiceprint(*, enrolled=True, calibrated=False):
    return SimpleNamespace(
        enrolled=enrolled,
        profile=SimpleNamespace(calibrated=calibrated, match_threshold=0.55),
    )


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    monkeypatch.setattr(conversation_module.voiceprint, "warm", lambda: None)


def _greet(voiceprint):
    conv, _, _, voice, speaker, ui = build(config=ConversationConfig(greet_on_start=False))
    conv.voiceprint = voiceprint
    conv.config.greet_on_start = True
    conv.start()
    settle(speaker)
    conv.stop()
    speaker.close()
    return " ".join(voice.lines), ui


def test_an_uncalibrated_profile_is_named_at_the_greeting():
    said, ui = _greet(_voiceprint(calibrated=False))
    assert "out of date" in said
    assert "Teach Vesper my voice" in said
    assert any("calibrat" in w for w in ui.warnings)


def test_a_calibrated_profile_greets_as_before():
    said, ui = _greet(_voiceprint(calibrated=True))
    assert "out of date" not in said
    assert not any("calibrat" in w for w in ui.warnings)


def test_no_profile_at_all_is_not_nagged_about():
    """Never enrolled is a choice: it answers everyone, as documented."""
    said, _ = _greet(_voiceprint(enrolled=False))
    assert "out of date" not in said


def test_the_never_matched_warning_names_the_fix_a_person_can_click():
    conv, _, _, _, speaker, ui = build()
    speaker.close()
    conv.voice_scored = conversation_module.VOICE_DOUBT_AFTER
    conv.voice_matches = 0
    conv._maybe_warn_about_voice()
    assert any("Teach Vesper my voice" in w for w in ui.warnings)


def test_the_teach_shortcut_target_exists_and_enrolls():
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "teach_voice.bat"
    text = script.read_text(encoding="utf-8")
    assert "--enroll" in text
    assert "pause" in text.lower()
