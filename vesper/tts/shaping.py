"""Giving the voice a texture, without giving it a new dependency.

Piper produces a clean, close-mic'd, slightly dry read. That is the right raw
material and the wrong finished sound for an assistant that is meant to feel
like it is in the room with you rather than reading to you from a script.

Three things separate the two, and none of them is the voice model:

  Tone. A gentle low cut clears the chest boom a close mic exaggerates, a lift
  around three kilohertz buys intelligibility at low volume, and a little air on
  top stops it sounding like a phone call.

  Space. A completely dry voice sounds like it is inside your head. Three
  quiet early reflections put it a few feet away in a small hard room, which is what
  a ceiling speaker actually sounds like. This is the single biggest change and
  also the easiest to overdo: past about fifteen percent it stops sounding like
  a room and starts sounding like a bathroom.

  Evenness. Piper's own noise parameters control how much the delivery wanders
  in pitch and timing. Turning them down is what makes a voice sound composed
  rather than chatty, and it is a model-level control, not something you can
  add afterwards with a filter.

Everything here is numpy. scipy would be the obvious way to do the filtering and
it is a forty megabyte dependency for four biquads, on a machine that already
has Windows Application Control blocking wheels. So the tone shaping is done as
a single smooth curve in the frequency domain, which is one transform per chunk
and measurably cheaper than the synthesis it follows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_INT16_MAX = 32767.0


@dataclass(frozen=True)
class Character:
    """How the voice should sound. All of it optional, none of it required."""

    name: str = "natural"

    # --- model level, passed to Piper ---
    # Piper's defaults wander on purpose, which reads as human. Lower values
    # read as composed. None leaves Piper's own default alone.
    noise_scale: float | None = None      # timbre and pitch variation
    noise_w_scale: float | None = None    # phoneme length variation

    # --- tone, in decibels ---
    low_cut_hz: float = 0.0               # 0 disables
    presence_db: float = 0.0              # lift around presence_hz
    presence_hz: float = 3200.0
    presence_q: float = 0.9               # width, larger is narrower
    air_db: float = 0.0                   # shelf above air_hz
    air_hz: float = 9000.0

    # --- space ---
    # Delays in milliseconds and their levels, relative to the dry signal.
    reflections: tuple[tuple[float, float], ...] = ()
    # Overall wet level. Kept separate so a room can be dialled back without
    # rebalancing every tap.
    room: float = 0.0

    # --- glue ---
    drive: float = 0.0                    # gentle saturation, 0 disables
    output_gain: float = 1.0

    @property
    def shapes_audio(self) -> bool:
        """Is there any post-processing to do at all?"""
        return bool(
            self.low_cut_hz or self.presence_db or self.air_db
            or (self.reflections and self.room) or self.drive
            or self.output_gain != 1.0
        )


# The natural voice, untouched. Also the escape hatch: if any of this sounds
# wrong, `character: natural` is exactly what Vesper sounded like before.
NATURAL = Character(name="natural")

# Measured and tuned by ear against en_GB-alan-medium. The numbers are
# deliberately restrained. Every one of these was first tried at roughly twice
# the value and every one of them sounded like a novelty filter.
JARVIS = Character(
    name="jarvis",
    # Composed rather than chatty. This is most of the character.
    noise_scale=0.55,
    noise_w_scale=0.6,
    # Close-mic chest boom out, articulation up, a little air so it does not
    # sound like it is coming down a phone line.
    low_cut_hz=95.0,
    presence_db=3.0,
    presence_hz=3300.0,
    presence_q=0.8,
    air_db=1.5,
    air_hz=8500.0,
    # A small hard room, a few feet away. Three taps: more starts to smear
    # consonants, which costs intelligibility for no gain in atmosphere.
    reflections=((11.0, 0.5), (23.0, 0.32), (41.0, 0.18)),
    room=0.16,
    drive=0.08,
    output_gain=0.82,
)

# Tone without the room, for headphones or a noisy desk where reflections just
# muddy things.
BROADCAST = replace(
    JARVIS, name="broadcast", reflections=(), room=0.0, air_db=2.5, drive=0.08
)

# Tuned by ear for a female medium voice (en_GB-jenny_dioco, en_US-hfc_female),
# where the jarvis numbers, made for Alan, turn thin and sibilant: the 3.3 kHz
# lift lands on the s sounds and there is little below 95 Hz to cut. Less lift,
# lower down; more of Piper's own pitch and timing variation, so it reads as a
# person talking rather than a composed announcer; and a smaller, nearer room.
WARM = Character(
    name="warm",
    noise_scale=0.62,
    noise_w_scale=0.72,
    low_cut_hz=120.0,
    presence_db=1.5,
    presence_hz=2600.0,
    presence_q=0.7,
    air_db=1.0,
    air_hz=9000.0,
    reflections=((9.0, 0.5), (19.0, 0.3), (33.0, 0.15)),
    room=0.10,
    drive=0.06,
    output_gain=0.86,
)

PRESETS: dict[str, Character] = {
    "natural": NATURAL,
    "jarvis": JARVIS,
    "broadcast": BROADCAST,
    "warm": WARM,
}


def preset(name: str) -> Character:
    """Look up a character by name, falling back to the untouched voice."""
    return PRESETS.get((name or "natural").strip().lower(), NATURAL)


# --- tone -------------------------------------------------------------------


def _response(freqs: np.ndarray, character: Character) -> np.ndarray:
    """The gain curve to apply, as a multiplier per frequency bin.

    Built as a smooth curve rather than as biquads because it is applied in the
    frequency domain anyway, and because a curve is far easier to reason about
    than a cascade of coefficients when tuning by ear.
    """
    gain = np.ones_like(freqs)

    if character.low_cut_hz > 0:
        # A gentle second-order-ish slope. Steeper sounds like a telephone.
        ratio = np.maximum(freqs, 1.0) / character.low_cut_hz
        gain *= ratio**2 / np.sqrt(1.0 + ratio**4)

    if character.presence_db:
        # A bell centred on presence_hz, in octaves so the width is musical.
        octaves = np.log2(np.maximum(freqs, 1.0) / character.presence_hz)
        bell = np.exp(-((octaves * character.presence_q) ** 2) * 2.0)
        gain *= 10.0 ** (character.presence_db * bell / 20.0)

    if character.air_db:
        # A shelf that reaches full lift about an octave above air_hz.
        shelf = 1.0 / (1.0 + (character.air_hz / np.maximum(freqs, 1.0)) ** 2)
        gain *= 10.0 ** (character.air_db * shelf / 20.0)

    return gain


def _apply_tone(samples: np.ndarray, rate: int, character: Character) -> np.ndarray:
    if not (character.low_cut_hz or character.presence_db or character.air_db):
        return samples
    if samples.size < 32:
        return samples
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
    return np.fft.irfft(spectrum * _response(freqs, character), n=samples.size)


# --- space ------------------------------------------------------------------


def _apply_room(samples: np.ndarray, rate: int, character: Character) -> np.ndarray:
    """A few early reflections, mixed in quietly behind the dry voice.

    Not a reverb tail. Tails need a decay long enough to overlap the next
    sentence, and Vesper speaks in short sentences with barge-in cutting them
    off, so a tail would still be ringing when the audio is meant to have
    stopped dead.
    """
    if not character.reflections or character.room <= 0:
        return samples
    wet = np.zeros_like(samples)
    for delay_ms, level in character.reflections:
        offset = int(rate * delay_ms / 1000.0)
        if offset <= 0 or offset >= samples.size:
            continue
        wet[offset:] += samples[:-offset] * level
    return samples * (1.0 - character.room * 0.5) + wet * character.room


# --- glue -------------------------------------------------------------------


def _apply_drive(samples: np.ndarray, character: Character) -> np.ndarray:
    """Very gentle saturation. Evens out peaks and adds a little warmth.

    Cheaper and more predictable than a compressor, which would need attack and
    release times and therefore a sequential loop over every sample.
    """
    if character.drive <= 0:
        return samples
    amount = 1.0 + character.drive * 4.0
    return np.tanh(samples * amount) / np.tanh(amount)


def shape(samples: np.ndarray, rate: int, character: Character) -> np.ndarray:
    """Apply a character to one chunk of int16 audio. Returns int16.

    Chunk by chunk rather than to a whole utterance, because playback is
    streamed and buffering the lot would put the first word seconds late. The
    room taps are short enough that the seam between chunks is inaudible.
    """
    if samples.size == 0 or not character.shapes_audio:
        return samples

    audio = samples.astype(np.float32) / _INT16_MAX
    audio = _apply_tone(audio, rate, character)
    audio = _apply_room(audio, rate, character)
    audio = _apply_drive(audio, character)
    audio *= character.output_gain

    # Clip rather than renormalise. Renormalising per chunk would make the
    # volume pump between sentences, which is far more noticeable than the
    # occasional clipped peak.
    np.clip(audio, -1.0, 1.0, out=audio)
    return (audio * _INT16_MAX).astype(np.int16)
