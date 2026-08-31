"""The voice character, tested as signal rather than as opinion.

You cannot unit test whether something sounds like JARVIS. You can test that the
low cut cuts lows, that the presence lift lifts presence, that the room arrives
when it should, and that none of it clips or changes the length of the audio,
which are the things that would actually make it sound broken.

Measured against a real render of en_GB-alan-medium at the tuned settings:

    band          natural   jarvis    delta
    sub 20-90       37.3     34.2     -3.1 dB
    low 90-300      49.8     49.6     -0.2 dB
    mid             38.0     38.7     +0.8 dB
    presence        28.7     33.0     +4.4 dB
    air             25.1     26.4     +1.4 dB

The mid figure is the one that keeps the comparison honest. An early version was
2.2dB louder overall, and a louder voice always wins an A/B regardless of
whether it is better, so the output gain is set to land the level on top of the
untouched voice.
"""

import numpy as np
import pytest

from vesper.tts import shaping
from vesper.tts.shaping import JARVIS, NATURAL, Character, preset, shape

RATE = 22050


def tone(hz: float, seconds: float = 0.5, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * hz * t) * level * 32767).astype(np.int16)


def rms_db(samples: np.ndarray) -> float:
    audio = samples.astype(np.float64) / 32767.0
    return 20 * np.log10(max(np.sqrt(np.mean(audio**2)), 1e-12))


# --- the untouched voice ----------------------------------------------------


def test_natural_is_a_true_passthrough():
    """The escape hatch has to be exact. If any of this sounds wrong, setting
    character to natural must give back the voice as it was, not an
    approximation of it."""
    samples = tone(440)
    assert np.array_equal(shape(samples, RATE, NATURAL), samples)


def test_an_unknown_character_falls_back_to_untouched():
    assert preset("nonsense") is NATURAL
    assert preset("") is NATURAL
    assert preset("JARVIS").name == "jarvis"


def test_empty_audio_is_handled():
    assert shape(np.zeros(0, dtype=np.int16), RATE, JARVIS).size == 0


# --- tone -------------------------------------------------------------------


def test_the_low_cut_cuts_lows_and_leaves_speech_alone():
    quiet = Character(name="t", low_cut_hz=95.0)
    rumble = rms_db(shape(tone(50), RATE, quiet)) - rms_db(tone(50))
    speech = rms_db(shape(tone(700), RATE, quiet)) - rms_db(tone(700))
    assert rumble < -6, f"50Hz rumble only dropped {rumble:.1f}dB"
    assert abs(speech) < 1.0, f"700Hz moved {speech:.1f}dB, it should not"


def test_the_presence_lift_lands_where_it_is_aimed():
    lift = Character(name="t", presence_db=6.0, presence_hz=3300.0)
    at_target = rms_db(shape(tone(3300), RATE, lift)) - rms_db(tone(3300))
    two_octaves_below = rms_db(shape(tone(800), RATE, lift)) - rms_db(tone(800))
    assert 4.0 < at_target < 7.0, f"aimed for 6dB, got {at_target:.1f}dB"
    assert two_octaves_below < 1.5, "the bell is far too wide"


def test_the_air_shelf_lifts_only_the_top():
    airy = Character(name="t", air_db=4.0, air_hz=8000.0)
    top = rms_db(shape(tone(12000), RATE, airy)) - rms_db(tone(12000))
    body = rms_db(shape(tone(1000), RATE, airy)) - rms_db(tone(1000))
    assert top > 2.0, f"top only moved {top:.1f}dB"
    assert abs(body) < 1.0


# --- space ------------------------------------------------------------------


def test_the_room_arrives_after_the_voice_and_not_before():
    """A reflection that precedes the sound it reflects is the classic way to
    make a room sound wrong. It should be silent until the first tap."""
    room = Character(name="t", reflections=((20.0, 0.5),), room=0.5)
    click = np.zeros(RATE // 2, dtype=np.int16)
    click[0] = 20000
    out = shape(click, RATE, room)

    first_tap = int(RATE * 20 / 1000)
    assert np.abs(out[1:first_tap - 2]).max() < 200, "something arrived early"
    assert np.abs(out[first_tap - 2:first_tap + 3]).max() > 1000, "no reflection"


def test_the_room_does_not_ring_into_the_next_sentence():
    """Vesper speaks in short sentences and barge-in cuts them off mid-word. A
    reverb tail would still be sounding after the audio was meant to stop."""
    longest = max(delay for delay, _ in JARVIS.reflections)
    assert longest < 60, f"{longest}ms tap is a tail, not an early reflection"


# --- staying inside the rails -----------------------------------------------


def test_loud_input_does_not_clip_or_wrap():
    """int16 wrapping is the worst possible failure here: it turns a loud vowel
    into a burst of noise rather than a quiet one."""
    loud = (np.sin(np.linspace(0, 400 * np.pi, RATE)) * 32000).astype(np.int16)
    out = shape(loud, RATE, JARVIS)
    assert out.dtype == np.int16
    assert np.abs(out).max() <= 32767
    # A wrap shows up as a sign flip against the input at the peaks.
    peaks = np.abs(loud) > 30000
    same_sign = np.sign(out[peaks]) == np.sign(loud[peaks])
    assert same_sign.mean() > 0.95, "samples wrapped around instead of clipping"


def test_length_is_preserved_so_playback_timing_is_unchanged():
    samples = tone(440, seconds=0.3)
    assert shape(samples, RATE, JARVIS).size == samples.size


def test_the_tuned_character_is_level_matched_to_the_untouched_one():
    """Guards the honesty of the A/B. Louder always sounds better, so if this
    drifts the comparison stops meaning anything."""
    speech_ish = sum(tone(hz, level=0.15) for hz in (220, 440, 900, 1800, 3300))
    delta = rms_db(shape(speech_ish, RATE, JARVIS)) - rms_db(speech_ish)
    assert abs(delta) < 2.5, f"jarvis is {delta:+.1f}dB off the natural voice"


def test_a_tiny_chunk_is_left_alone_rather_than_mangled():
    """Piper can emit very short chunks. An FFT over eight samples is noise."""
    tiny = np.array([100, -100, 50, -50], dtype=np.int16)
    assert shape(tiny, RATE, JARVIS).size == tiny.size


# --- what gets passed to the model ------------------------------------------


def test_the_composed_delivery_is_a_model_setting_not_a_filter():
    """The evenness that makes it sound composed comes from Piper's own noise
    parameters. No amount of EQ afterwards can produce it, so it must actually
    reach the synthesiser."""
    assert JARVIS.noise_scale is not None and JARVIS.noise_scale < 1.0
    assert JARVIS.noise_w_scale is not None and JARVIS.noise_w_scale < 1.0
    assert NATURAL.noise_scale is None, "natural must not touch Piper's defaults"


def test_every_preset_is_reachable_by_name():
    for name, character in shaping.PRESETS.items():
        assert preset(name) is character
        assert character.name == name
