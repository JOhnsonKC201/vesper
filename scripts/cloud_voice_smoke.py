"""Prove the cloud voice end to end, and measure what it costs.

The unit tests fake the network, which is right for a suite that has to run
offline, but it means nothing in `pytest` has ever spoken to ElevenLabs. This
does, once, deliberately, and prints the numbers that decide whether the
feature is worth having: time to first audio, characters spent, and how much
the cache saves on the second pass.

    python scripts/cloud_voice_smoke.py            # writes wav, no speakers
    python scripts/cloud_voice_smoke.py --play     # actually make noise

It spends real characters, roughly 200 of the monthly allowance. It refuses to
run without a key rather than quietly measuring Piper and calling it a pass.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper import config as config_module  # noqa: E402
from vesper.tts import eleven_api  # noqa: E402
from vesper.tts.budget import Budget  # noqa: E402
from vesper.tts.cache import AudioCache  # noqa: E402
from vesper.tts.eleven import ElevenTTS  # noqa: E402
from vesper.tts.eleven_api import ElevenClient  # noqa: E402

LINE = "Good evening. Everything is where you left it, and the house is quiet."


class Capture:
    """A stand-in output stream that keeps the samples instead of playing."""

    def __init__(self, **kwargs):
        self.blocks: list[np.ndarray] = []
        self.first_at: float | None = None

    def start(self):
        pass

    def write(self, block):
        if self.first_at is None:
            self.first_at = time.monotonic()
        self.blocks.append(np.asarray(block).copy())

    def abort(self):
        pass

    def close(self):
        pass

    @property
    def samples(self):
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0, "<i2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="use the real speakers")
    parser.add_argument("--voice", default="", help="voice id, defaults to config")
    args = parser.parse_args()

    cfg = config_module.load()
    key = cfg.eleven_key()
    if not key:
        print("no key. Set VESPER_ELEVEN_API_KEY or voice.eleven.api_key first.")
        return 2

    out = Path(__file__).resolve().parent.parent / "var" / "cloud-voice"
    out.mkdir(parents=True, exist_ok=True)

    captured: list[Capture] = []
    if not args.play:
        import types

        module = types.ModuleType("sounddevice")
        module.OutputStream = lambda **kw: captured[-1]
        sys.modules["sounddevice"] = module

    voice_id = args.voice or cfg.voice.eleven.voice_id
    budget = Budget(out / "smoke-budget.json", cap=cfg.voice.eleven.monthly_characters)
    voice = ElevenTTS(
        _Silent(),
        client=ElevenClient(key, timeout_s=cfg.voice.eleven.timeout_s),
        cache=AudioCache(out / "cache"),
        budget=budget,
        voice_id=voice_id,
        model_id=cfg.voice.eleven.model_id,
        log=lambda message: print(f"  log: {message}"),
    )

    print(f"voice   : {eleven_api.voice_named(voice_id)} ({voice_id})")
    print(f"model   : {voice.model_id}")
    print(f"text    : {len(LINE)} characters")
    print()

    # Always measure a genuine cold path. Without this the second run of the
    # day reports a cache hit as if it were the network.
    voice.cache.clear()

    for label in ("cold (network)", "warm (cache)"):
        captured.append(Capture())
        # The backend keeps its output stream open between utterances, which is
        # correct, but it means the second pass would write into the first
        # capture and report zero samples. Drop it so each pass is measured on
        # its own device.
        voice._teardown()
        started = time.monotonic()
        before = budget.spent
        voice.speak(LINE, threading.Event())
        elapsed = time.monotonic() - started

        stream = captured[-1]
        first = (stream.first_at - started) if stream.first_at else float("nan")
        samples = stream.samples
        seconds = samples.size / eleven_api.SAMPLE_RATE if samples.size else 0.0
        print(f"{label}")
        print(f"  reason        : {voice.last_reason}")
        print(f"  first audio   : {first * 1000:6.0f} ms")
        print(f"  complete      : {elapsed * 1000:6.0f} ms")
        print(f"  audio         : {seconds:.2f}s")
        print(f"  charged       : {budget.spent - before} characters")
        print()

        if samples.size and not args.play:
            path = out / f"{label.split()[0]}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(eleven_api.SAMPLE_RATE)
                handle.writeframes(samples.tobytes())
            peak = int(np.abs(samples).max())
            print(f"  wrote {path.name}, peak {peak} (0 would mean silence)")
            print()

    print(f"budget  : {budget.summary()}")
    print(f"cache   : {voice.cache.size():,} bytes on disk")
    return 0


class _Silent:
    """A fallback that records rather than speaks, so a fallback is visible."""

    name = "none"

    def speak(self, text, stop):
        print(f"  FELL BACK TO LOCAL: {text[:40]!r}")

    def close(self):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
