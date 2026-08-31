"""Render the same lines through every voice character, so you can pick by ear.

Tuning a voice by reading numbers in a config file does not work. This writes
one wav per character into var/voice-ab/ and prints how long each took to
synthesise, so the choice is made by listening and the cost of making it is
visible.

    python scripts/voice_ab.py
    python scripts/voice_ab.py "say something else entirely"
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper import config as config_module
from vesper.tts import shaping
from vesper.tts.piper_voice import PiperTTS

# Three lines chosen for what they expose. The first is ordinary conversation,
# the second is the permission question you will hear most often, and the third
# is full of numbers and consonants, which is where a presence lift either earns
# its place or turns into sibilance.
LINES = [
    "Vesper here. I'm listening.",
    "I want to run git commit. Do I do this for you?",
    "The disk is at ninety two percent, and six processes are using the GPU.",
]


def write_wav(path: Path, samples, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def main() -> int:
    cfg = config_module.load()
    model = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
    if model is None:
        print(f"no piper voice in {cfg.voices_path()}")
        return 1

    lines = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else LINES
    out = config_module.ROOT / "var" / "voice-ab"
    text = "  ".join(lines)

    print(f"\nvoice: {model.stem}")
    print(f"text : {text[:70]}{'...' if len(text) > 70 else ''}\n")

    for name in shaping.PRESETS:
        character = shaping.preset(name)
        voice = PiperTTS(
            model,
            speed=cfg.voice.speed,
            volume=cfg.voice.volume,
            character=character,
        )
        started = time.monotonic()
        samples = voice.synthesize(text)
        elapsed = time.monotonic() - started
        seconds = len(samples) / voice.sample_rate if len(samples) else 0.0
        target = out / f"{name}.wav"
        write_wav(target, samples, voice.sample_rate)
        voice.close()

        realtime = (seconds / elapsed) if elapsed else 0.0
        print(
            f"  {name:10} {seconds:5.1f}s of audio in {elapsed:5.2f}s "
            f"({realtime:4.0f}x realtime)   {target}"
        )

    print(f"\nplay them from {out}")
    print("then set voice.character in config.yaml to the one you want\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
