"""Every voice this machine can speak in with the network unplugged.

The cloud picker has `eleven_api.KNOWN_VOICES`: a fixed list of what can be
chosen, with a name and a blurb for each. Nothing equivalent existed for the
local backends, so the dashboard had nothing to draw and the only way to change
voice offline was to edit `config.yaml` and restart. This is that list.

Two things are voices here rather than one. A Piper model is half the sound; the
character applied to it (`vesper/tts/shaping.py`) is the other half, and the
difference between `jarvis` and `natural` on the same model is larger than the
difference between two models. Offering the model alone would hide the control
that actually changes how Vesper sounds, so every model is listed once per
delivery and an id carries both.

Windows SAPI voices are listed too. They sound like a 2005 satnav, which is
exactly why they are worth offering: they are the ones that are already
installed, on a machine where a Piper download may never have happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The three characters from shaping.PRESETS, ordered as they are worth trying
# rather than alphabetically: the configured default first. `test_catalog`
# asserts this stays in step with PRESETS, so adding a fourth there and
# forgetting it here fails rather than silently hiding it.
DELIVERIES = ("jarvis", "natural", "broadcast", "warm")

# Piper names every model `<locale>-<voice>-<quality>.onnx`, so the locale is
# free information. Only the ones a British-English install is likely to have
# are spelled out; anything else falls back to the raw locale, which is still
# more use in a list than nothing.
LOCALES = {
    "en_GB": "British",
    "en_US": "American",
    "en_AU": "Australian",
    "en_IE": "Irish",
    "en_CA": "Canadian",
    "en_IN": "Indian",
}


@dataclass(frozen=True)
class LocalVoice:
    """One voice the picker can offer, from a backend that needs no network."""

    engine: str  # piper | sapi
    voice_id: str
    name: str
    detail: str = ""
    character: str = ""
    # Always true. The dashboard greys anything that is not, which only the
    # cloud list ever uses, and this keeps the two lists one shape.
    free: bool = True

    @property
    def label(self) -> str:
        return f"{self.name}  {self.detail}".strip()


def piper_id(stem: str, character: str) -> str:
    """The id for a model and a delivery, which are one choice, not two."""
    return f"{stem}:{character}"


def split_piper_id(voice_id: str) -> tuple[str, str]:
    """The inverse. A bare stem reads as the default delivery."""
    stem, _, character = (voice_id or "").partition(":")
    return stem, character or DELIVERIES[0]


def discover(voices_dir: str | Path, *, sapi: bool = True) -> tuple[LocalVoice, ...]:
    """Everything installed, Piper first because it is the better voice."""
    found = list(piper_voices(voices_dir))
    if sapi:
        found.extend(sapi_voices())
    return tuple(found)


def piper_voices(voices_dir: str | Path) -> list[LocalVoice]:
    """Every downloaded model, once per delivery."""
    directory = Path(voices_dir)
    if not directory.is_dir():
        return []

    voices: list[LocalVoice] = []
    for model in sorted(directory.glob("*.onnx")):
        name, place = _describe(model.stem)
        for delivery in DELIVERIES:
            voices.append(
                LocalVoice(
                    engine="piper",
                    voice_id=piper_id(model.stem, delivery),
                    name=name,
                    detail=f"{place}, {delivery}",
                    character=delivery,
                )
            )
    return voices


def sapi_voices() -> list[LocalVoice]:
    """Whatever Windows has. Empty everywhere else, which is not an error."""
    from .sapi import SapiTTS

    return [
        LocalVoice(engine="sapi", voice_id=description,
                   name=_sapi_name(description), detail="Windows")
        for description in SapiTTS.list_voices()
    ]


def find(voices, voice_id: str):
    """The record for an id, or None. Ids come back from a UI, so never trust one."""
    for voice in voices or ():
        if voice.voice_id == voice_id:
            return voice
    return None


def build(voice: LocalVoice, *, voices_dir: str | Path,
          speed: float = 1.0, volume: float = 1.0):
    """Make the backend this record describes. Raises if it will not load."""
    if voice.engine == "piper":
        from . import shaping
        from .piper_voice import PiperTTS

        stem, character = split_piper_id(voice.voice_id)
        return PiperTTS(
            Path(voices_dir) / f"{stem}.onnx",
            speed=speed,
            volume=volume,
            character=shaping.preset(character),
        )

    from .sapi import SapiTTS

    return SapiTTS(voice_hint=voice.voice_id)


def current_id(voice) -> str:
    """The catalogue id of a backend that is already running.

    Read off the object rather than remembered, because the voice can also be
    set by `config.yaml` at startup, and a picker that highlighted the wrong
    row would be worse than one that highlighted none.
    """
    character = getattr(voice, "character", None)
    name = getattr(voice, "name", "")
    if character is not None and name:
        return piper_id(name, getattr(character, "name", "") or DELIVERIES[0])
    return getattr(voice, "voice_hint", "") or ""


def _describe(stem: str) -> tuple[str, str]:
    """A model file name turned into something worth reading in a list."""
    parts = stem.split("-")
    locale = parts[0] if parts else ""
    name = (parts[1] if len(parts) > 1 else stem).replace("_", " ").title()
    quality = parts[2] if len(parts) > 2 else ""
    place = LOCALES.get(locale, locale or "local")
    # Quality is only worth the width when it is not the usual one, and then it
    # is the only thing telling two rows of the same voice apart.
    if quality and quality != "medium":
        place = f"{place} {quality}"
    return name, place


def _sapi_name(description: str) -> str:
    """"Microsoft David Desktop - English (United States)" reads as "David"."""
    head = description.split(" - ")[0].strip()
    for noise in ("Microsoft ", "Desktop"):
        head = head.replace(noise, "").strip()
    return head or description
