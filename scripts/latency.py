"""Measure where the time actually goes, so regressions are visible.

Every number in the README came from this. Run it after changing a model, a
flag, or anything in the audio path.

    python scripts/latency.py
    python scripts/latency.py --model fable --effort low
    python scripts/latency.py --questions world

`--questions system` (the default) asks three things about this machine, each
needing a shell command. `--questions world` asks three things about the world,
each needing a web search, which is the other shape a turn takes now. Each run
is three paid turns, about nine cents on opus.

The last line of every run is a single summary, made for pasting into the
comment next to `brain.effort` in config.yaml.
"""

from __future__ import annotations

import argparse
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

QUESTIONS = {
    "system": [
        "What time is it?",
        "How much memory is free?",
        "Is the battery charging?",
    ],
    "world": [
        "What's the weather in Baltimore tomorrow?",
        "Who won the last Ravens game?",
        "What is the population of Nigeria?",
    ],
}


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


def _median_ms(samples: list[float]) -> str:
    return f"{statistics.median(samples) * 1000:.0f}ms" if samples else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="", help="override brain.model")
    parser.add_argument("--effort", default=None, help="override brain.effort")
    parser.add_argument("--fallback-model", default=None, help="override brain.fallback_model")
    parser.add_argument(
        "--questions", choices=sorted(QUESTIONS), default="system",
        help="system: three shell questions. world: three web questions.",
    )
    args = parser.parse_args()

    cfg = load()
    model = args.model or cfg.brain.model
    effort = cfg.brain.effort if args.effort is None else args.effort
    fallback = cfg.brain.fallback_model if args.fallback_model is None else args.fallback_model
    questions = QUESTIONS[args.questions]
    print("\nvesper latency profile\n")

    # --- speech to text -----------------------------------------------------
    voice = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
    piper = PiperTTS(voice, speed=cfg.voice.speed)
    stt = Listener(WhisperConfig(model=cfg.listening.whisper_model))
    stt.load()

    asr, synth = [], []
    for question in questions:
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
    # The world questions need the search and nothing else. The system ones
    # need a shell, and the read allowlist here is deliberately the one thing
    # the real config forbids: this is a benchmark, not the assistant, and a
    # question that stops to ask would measure the wrong thing.
    if args.questions == "world":
        tools, allowed = ("WebSearch",), ()
    else:
        tools, allowed = ("Bash",), ("Bash(python*)",)
    brain = ClaudeBrain(
        BrainConfig(
            model=model,
            effort=effort,
            fallback_model=fallback,
            system_prompt=build_system_prompt(cfg.identity.user),
            tools=tools,
            allowed_tools=allowed,
            free_web=True,
        )
    )
    brain.start()

    # Two different first-word numbers matter. `ttft` is when Claude produces
    # text, which for anything needing a tool is after the round trip. `audible`
    # is when the user actually hears something, because a tool call triggers a
    # spoken holding phrase. The second is what people feel.
    ttft, audible, totals, cache_writes = [], [], [], []
    try:
        for question in questions:
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

    label = f"{model}" + (f", effort {effort}" if effort else "") + (
        f", fallback {fallback}" if fallback else ""
    )
    print(f"\n  claude {label} via cli, safe mode, {args.questions} questions")
    report("  time to first token", ttft)
    report("  time to first audio", audible)
    report("  full turn", totals)
    print(f"    prompt tokens written     {max(cache_writes) if cache_writes else 0}")
    cost = brain.total_cost_usd / max(1, brain.turn_count)
    print(f"    cost per turn             ${cost:.4f}")

    end_silence = cfg.listening.end_silence_ms / 1000
    heard_at = statistics.median(audible or ttft or [0])
    budget = end_silence + statistics.median(asr) + heard_at
    print()
    print(f"  end of speech to first word   {budget:.2f}s")
    print(f"    endpoint silence          {end_silence:.2f}s")
    print(f"    transcription             {statistics.median(asr):.2f}s")
    print(f"    claude first audio        {heard_at:.2f}s")
    print()
    print(
        f"  summary: {label}, {args.questions}: first token {_median_ms(ttft)}, "
        f"first audio {_median_ms(audible)}, full turn {_median_ms(totals)}, "
        f"${cost:.3f}/turn"
    )
    print()

    piper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
