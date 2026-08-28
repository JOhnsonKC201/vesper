"""End to end proof, with real audio through every real component.

Piper synthesizes a spoken question. The audio is resampled to 16kHz and pushed
through the microphone seam block by block, exactly as a live mic would. From
there nothing is faked: Silero endpoints it, Whisper transcribes it, the wake
gate decides it was addressed, Claude answers with real shell access, and the
reply is routed through the sentence assembler to the voice.

Playback is silenced by default so this can run at three in the morning without
waking anyone. Pass --audible to actually hear it.

    python scripts/voice_smoke.py
    python scripts/voice_smoke.py --audible
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper.audio.mic import BLOCK_FRAMES
from vesper.audio.speaker import Speaker
from vesper.audio.vad import EndpointConfig
from vesper.brain.claude import BrainConfig, ClaudeBrain
from vesper.brain.persona import build_system_prompt
from vesper.config import load
from vesper.conversation import Conversation, ConversationConfig
from vesper.stt.whisper import Listener, WhisperConfig, wake_word_prompt
from vesper.tts.piper_voice import PiperTTS
from vesper.wake import WakeConfig, WakeGate

QUESTIONS = [
    "Vesper, how many CPU cores does this machine have?",
    "And how much free space is on the C drive?",
    "What was the first thing I asked you?",
]


class RecordingVoice:
    """Renders speech but does not play it, so tests stay silent."""

    name = "recording"

    def __init__(self, piper: PiperTTS, audible: bool) -> None:
        self.piper = piper
        self.audible = audible
        self.lines: list[str] = []

    def speak(self, text: str, stop: threading.Event) -> None:
        self.lines.append(text)
        if self.audible:
            self.piper.speak(text, stop)
        else:
            self.piper.synthesize(text)  # real synthesis, no device

    def close(self) -> None:
        self.piper.close()


class ScriptedMic:
    """Feeds pre-rendered audio through the same interface as a real mic.

    Keeps a real pre-roll ring, like Microphone does. Without it the endpointer
    discards audio until it has confirmed speech, which eats the onset of the
    first word: "Vesper, how many cores" arrives as "But how many cores" and
    the assistant never wakes. A harness that skips this tests a machine the
    user does not have.
    """

    def __init__(self, preroll_ms: int = 400) -> None:
        self.blocks: list[np.ndarray] = []
        self._history: collections.deque = collections.deque(
            maxlen=max(1, (preroll_ms * 16000 // 1000) // BLOCK_FRAMES)
        )

    def load(self, audio: np.ndarray) -> None:
        # Trailing silence so the endpointer sees the turn finish.
        padded = np.concatenate([np.zeros(4800, np.float32), audio, np.zeros(24000, np.float32)])
        self.blocks = [
            padded[i : i + BLOCK_FRAMES]
            for i in range(0, len(padded) - BLOCK_FRAMES + 1, BLOCK_FRAMES)
        ]

    def start(self): pass
    def stop(self): pass

    def drain(self):
        self.blocks.clear()
        self._history.clear()

    def read(self, timeout=1.0):
        if not self.blocks:
            return None
        block = self.blocks.pop(0)
        self._history.append(block)
        return block

    def preroll(self):
        blocks = list(self._history)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


class TraceUI:
    def __init__(self):
        self.t0 = time.monotonic()
        self.transcripts = []

    def _stamp(self):
        return f"{time.monotonic() - self.t0:6.2f}s"

    def heard(self, text, addressed):
        self.transcripts.append(text)
        mark = "addressed" if addressed else "ignored"
        print(f"  {self._stamp()}  heard [{mark}]: {text}")

    def thinking(self, what): pass
    def tool(self, name, detail): print(f"  {self._stamp()}  tool  {name}: {detail[:70]}")
    def screen(self, text): print(f"  {self._stamp()}  screen: {text[:100]}")
    def permission(self, tool, detail): print(f"  {self._stamp()}  needs ok: {tool}")
    def interrupted(self, dropped): print(f"  {self._stamp()}  interrupted")
    def discarded(self, reason): print(f"  {self._stamp()}  discarded: {reason}")
    def error(self, message): print(f"  {self._stamp()}  ERROR: {message}")
    def warn(self, message): print(f"  {self._stamp()}  warn: {message}")
    def proactive(self, text): print(f"  {self._stamp()}  unprompted: {text}")

    def answered(self, turn, total_s, first_speech_s):
        first = f"{first_speech_s:.2f}s" if first_speech_s else "n/a"
        print(
            f"  {self._stamp()}  done: {total_s:.2f}s total, first word {first}, "
            f"{turn.turns} steps, ${turn.cost_usd:.4f}"
        )


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    count = int(len(audio) * target_rate / source_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audible", action="store_true", help="actually play the replies")
    args = parser.parse_args()

    cfg = load()
    print("\nvoice smoke test: real audio, real whisper, real claude\n")

    model = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
    if model is None:
        print("no piper voice installed")
        return 1

    asker = PiperTTS(model, speed=1.0)      # renders the questions
    replier = PiperTTS(model, speed=cfg.voice.speed)  # speaks the answers

    ui = TraceUI()
    mic = ScriptedMic()
    voice = RecordingVoice(replier, args.audible)
    speaker = Speaker(voice)

    brain = ClaudeBrain(
        BrainConfig(
            system_prompt=build_system_prompt(cfg.identity.user, cfg.identity.personality),
            tools=("Bash",),
            allowed_tools=("Bash(python*)", "Bash(powershell*)"),
        )
    )

    conversation = Conversation(
        brain=brain,
        stt=Listener(
            WhisperConfig(
                model=cfg.listening.whisper_model,
                initial_prompt=wake_word_prompt(
                    cfg.identity.wake_words, cfg.identity.name
                ),
            )
        ),
        speaker=speaker,
        mic=mic,
        wake=WakeGate(WakeConfig(words=tuple(cfg.identity.wake_words))),
        config=ConversationConfig(greet_on_start=False),
        endpoint_config=EndpointConfig(end_silence_ms=cfg.listening.end_silence_ms),
        ui=ui,
    )

    conversation.stt.load()
    brain.start()

    try:
        for question in QUESTIONS:
            print(f"\n  spoken aloud: \"{question}\"")
            rendered = asker.synthesize(question).astype(np.float32) / 32768.0
            mic.load(resample(rendered, asker.sample_rate, 16000))
            conversation.endpointer.reset()

            while True:
                block = mic.read()
                if block is None:
                    break
                conversation._handle_block(block)
            speaker.wait_until_idle(timeout=120)
    finally:
        speaker.close()
        brain.stop()

    print("\n  ---")
    print(f"  transcribed : {len(ui.transcripts)} utterance(s)")
    for text in ui.transcripts:
        print(f"     {text!r}")
    print(f"  spoken back : {len(voice.lines)} line(s)")
    for line in voice.lines:
        print(f"     {line!r}")
    print(f"  turns       : {conversation.turns}")
    print(f"  total cost  : ${brain.total_cost_usd:.4f}")

    ok = conversation.turns == len(QUESTIONS) and len(voice.lines) >= len(QUESTIONS)
    print(f"\n  {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
