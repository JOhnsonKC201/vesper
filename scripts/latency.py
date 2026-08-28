"""Measure where the time actually goes, so regressions are visible.

Every number in the README came from this. Run it after changing a model, a
flag, or anything in the audio path.

    python scripts/latency.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper.brain.claude import BrainConfig, ClaudeBrain
from vesper.brain.persona import build_system_prompt
from vesper.brain.protocol import TextDelta, ToolStarted, TurnComplete
from vesper.config import load
from vesper.stt.whisper import Listener, WhisperConfig
from vesper.tts.piper_voice import PiperTTS

QUESTIONS = [
    "What time is it?",
    "How much memory is free?",
    "Is the battery charging?",
]


def report(label: str, samples: list[float], unit: str = "ms") -> None:
    if not samples:
        print(f"  {label:<28} no samples")
        return
    scale = 1000 if unit == "ms" else 1
    values = [s * scale for s in samples]
    print(
        f"  {label:<28} {statistics.median(values):7.0f}{unit} median   "
        f"({min(values):.0f} to {max(values):.0f})"
    )


def main() -> int:
    cfg = load()
    print("\nvesper latency profile\n")

    # --- speech to text -----------------------------------------------------
    model = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
    piper = PiperTTS(model, speed=cfg.voice.speed)
    stt = Listener(WhisperConfig(model=cfg.listening.whisper_model))
    stt.load()

    asr, synth = [], []
    for question in QUESTIONS:
        rendered = piper.synthesize(question).astype(np.float32) / 32768.0
        count = int(len(rendered) * 16000 / piper.sample_rate)
        clip = np.interp(
            np.linspace(0, len(rendered) - 1, count), np.arange(len(rendered)), rendered
        ).astype(np.float32)

        started = time.monotonic()
        stt.transcribe(clip)
        asr.append(time.monotonic() - started)

        started = time.monotonic()
        audio = piper.synthesize(question)
        elapsed = time.monotonic() - started
        synth.append(elapsed / (len(audio) / piper.sample_rate))

    print(f"  whisper {cfg.listening.whisper_model} on {stt.resolved_device}")
    report("  transcription", asr)
    print(f"  piper {piper.name}")
    print(f"    synthesis                 {1 / statistics.median(synth):.1f}x realtime")

    # --- brain --------------------------------------------------------------
    brain = ClaudeBrain(
        BrainConfig(
            model=cfg.brain.model,
            system_prompt=build_system_prompt(cfg.identity.user),
            tools=("Bash",),
            allowed_tools=("Bash(python*)",),
        )
    )
    brain.start()

    # Two different first-word numbers matter. `ttft` is when Claude produces
    # text, which for anything needing a shell command is after the tool round
    # trip. `audible` is when the user actually hears something, because a tool
    # call triggers a spoken holding phrase. The second is what people feel.
    ttft, audible, totals, cache_writes = [], [], [], []
    try:
        for question in QUESTIONS:
            started = time.monotonic()
            first_text = first_sound = None
            for event in brain.ask(question):
                now = time.monotonic() - started
                if isinstance(event, ToolStarted) and first_sound is None:
                    first_sound = now  # a filler is spoken here
                elif isinstance(event, TextDelta):
                    if first_text is None:
                        first_text = now
                    if first_sound is None:
                        first_sound = now
                elif isinstance(event, TurnComplete):
                    totals.append(now)
                    cache_writes.append(event.cache_write_tokens)
            if first_text is not None:
                ttft.append(first_text)
            if first_sound is not None:
                audible.append(first_sound)
    finally:
        brain.stop()

    print(f"\n  claude {cfg.brain.model} via cli, safe mode")
    report("  time to first token", ttft)
    report("  time to first audio", audible)
    report("  full turn", totals)
    print(f"    prompt tokens written     {max(cache_writes) if cache_writes else 0}")
    print(f"    cost per turn             ${brain.total_cost_usd / max(1, brain.turn_count):.4f}")

    end_silence = cfg.listening.end_silence_ms / 1000
    heard_at = statistics.median(audible or ttft or [0])
    budget = end_silence + statistics.median(asr) + heard_at
    print()
    print(f"  end of speech to first word   {budget:.2f}s")
    print(f"    endpoint silence          {end_silence:.2f}s")
    print(f"    transcription             {statistics.median(asr):.2f}s")
    print(f"    claude first audio        {heard_at:.2f}s")
    print()

    piper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
