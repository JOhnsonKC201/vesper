"""The loop: listen, understand, answer, be interruptible.

Everything else in this package is a component. This is the thing that makes it
a conversation.

Three problems get solved here that do not exist in a dictation tool:

  Barge-in. The microphone stays live while Vesper is speaking. Sustained speech
  cuts it off mid-word. Without this it is a kiosk.

  Hearing itself. With speakers rather than headphones, the mic picks up Vesper's
  own voice, which would make it interrupt itself forever. Rather than acoustic
  echo cancellation, the transcript of any barge-in is compared against what
  Vesper just said; a strong match means it heard itself and the utterance is
  discarded. This is cheap, needs no extra model, and self-corrects.

  Turn-taking. Requiring the wake word before every sentence is exhausting, so
  once addressed, Vesper stays engaged for a short window and plain speech counts
  as a reply.
"""

from __future__ import annotations

import collections
import random
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig, Endpointer, VoiceActivity
from .brain.channels import ChannelRouter
from .brain.claude import ClaudeBrain
from .brain.persona import (
    INTERRUPTED_NOTE,
    STILL_WORKING_FILLERS,
    THINKING_FILLERS,
    clean_for_speech,
    frame_turn,
    split_channels,
)
from .brain.protocol import BrainError, PermissionNeeded, TextDelta, ToolStarted, TurnComplete
from .brain.sentences import SentenceAssembler
from .sensors import snapshot as sensors
from .stt.whisper import Listener
from .wake import WakeGate

_WORDS = re.compile(r"[a-z']+")

# Handled locally, never sent to Claude. Telling something to be quiet should
# not require a network round trip, and must work while it is mid-sentence.
_MUTE_PHRASES = (
    "be quiet", "shut up", "stop talking", "mute yourself", "mute",
    "stop interrupting", "leave me alone", "quiet please", "no more updates",
)
_UNMUTE_PHRASES = (
    "you can talk again", "unmute", "start talking again", "resume updates",
    "you can speak again",
)


def _matches(text: str, phrases: tuple[str, ...]) -> bool:
    stripped = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return any(stripped == p or stripped.startswith(p + " ") for p in phrases)


@dataclass
class ConversationConfig:
    # Half duplex mutes the mic while speaking. Correct choice on laptop
    # speakers in a loud room; loses barge-in, so full duplex is the default.
    half_duplex: bool = False
    # Consecutive speech blocks needed to cut Vesper off. Three 30ms blocks is
    # 90ms of real voice, short enough to feel instant and long enough that a
    # cough or a keyboard clack does not stop it mid-sentence.
    barge_in_blocks: int = 3
    barge_in_threshold: float = 0.75
    # Ignore the mic for a moment after playback starts, so the leading edge of
    # our own voice cannot register as an interruption.
    self_mute_ms: int = 250
    echo_similarity: float = 0.6
    # A second holding phrase if a turn is still running this long into it.
    still_working_after_s: float = 11.0
    greet_on_start: bool = True


class Conversation:
    """Wires microphone, transcription, Claude and voice into one loop."""

    def __init__(
        self,
        *,
        brain: ClaudeBrain,
        stt: Listener,
        speaker: Speaker,
        mic: Microphone,
        wake: WakeGate,
        config: ConversationConfig | None = None,
        endpoint_config: EndpointConfig | None = None,
        ui=None,
    ) -> None:
        self.brain = brain
        self.stt = stt
        self.speaker = speaker
        self.mic = mic
        self.wake = wake
        self.config = config or ConversationConfig()
        self.ui = ui or _SilentUI()

        self.endpointer = Endpointer(VoiceActivity(on_warning=self.ui.warn), endpoint_config)
        # A separate, stricter detector for interruptions. Reusing the
        # endpointer's VAD would corrupt its state machine mid-utterance.
        self._barge_vad = VoiceActivity(threshold=self.config.barge_in_threshold)

        self._running = threading.Event()
        self._spoken_recently: collections.deque[str] = collections.deque(maxlen=6)
        self._speaking_since = 0.0
        self._barge_run = 0
        self.turns = 0
        self.interruptions = 0
        self.echo_rejections = 0
        self._last_filler = ""
        self.proactive = None  # set by main once the ambient loop exists
        self.session_store = None  # set by main when cross-restart memory is on

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        sensors.take()  # prime psutil counters
        self.stt.load()
        # resume, so a stored session id from a previous launch is actually used
        self.brain.start(resume=bool(self.brain.session_id))
        self.mic.start()
        self._running.set()
        if self.config.greet_on_start:
            self.speaker.say("Vesper here. I'm listening.")

    def stop(self) -> None:
        self._running.clear()
        self.mic.stop()
        self.speaker.close()
        self.brain.stop()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def remember_session(self) -> None:
        """Persist the session id so the next launch resumes this conversation."""
        if self.session_store is None:
            return
        self.session_store.save(
            self.brain.session_id,
            turns=getattr(self.brain, "turn_count", 0),
            cost_usd=getattr(self.brain, "total_cost_usd", 0.0),
        )

    # --- main loop ----------------------------------------------------------

    def run(self) -> None:
        self.start()
        try:
            while self._running.is_set():
                block = self.mic.read(timeout=0.5)
                if block is None:
                    continue
                self._handle_block(block)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _handle_block(self, block: np.ndarray) -> None:
        speaking = self.speaker.speaking

        if speaking:
            if self.config.half_duplex:
                return  # deaf while talking, by choice
            if self._detect_barge_in(block):
                self._interrupt()
            # Fall through: the interrupting audio is also the start of the
            # next utterance, so it must still reach the endpointer.
        else:
            self._barge_run = 0

        utterance = self.endpointer.feed(block, preroll=self.mic.preroll())
        if utterance is not None:
            self._on_utterance(utterance)

    # --- barge-in -----------------------------------------------------------

    def _detect_barge_in(self, block: np.ndarray) -> bool:
        if (time.monotonic() - self._speaking_since) * 1000 < self.config.self_mute_ms:
            return False
        if self._barge_vad.is_speech(block):
            self._barge_run += 1
        else:
            self._barge_run = 0
        return self._barge_run >= self.config.barge_in_blocks

    def _interrupt(self) -> None:
        dropped = self.speaker.barge_in()
        self.brain.interrupt()
        self._barge_run = 0
        self.interruptions += 1
        self.ui.interrupted(dropped)

    # --- utterance handling -------------------------------------------------

    def _on_utterance(self, audio: np.ndarray) -> None:
        self.ui.thinking("transcribing")
        transcript = self.stt.transcribe(audio)
        if not transcript.ok:
            self.ui.discarded(transcript.rejected_reason or "unclear")
            return

        if self._is_own_voice(transcript.text):
            self.echo_rejections += 1
            self.ui.discarded("heard myself")
            return

        result = self.wake.check(transcript.text, time.time())
        self.ui.heard(transcript.text, addressed=result.triggered)
        if not result.triggered:
            return

        text = result.text.strip()

        if _matches(text, _MUTE_PHRASES):
            if self.proactive is not None:
                self.proactive.mute()
            self.speaker.barge_in()
            self._say("Quiet from now on.")
            return
        if _matches(text, _UNMUTE_PHRASES):
            if self.proactive is not None:
                self.proactive.unmute()
            self._say("Back on.")
            return

        if not text:
            # Just the name on its own. Acknowledge and open the window.
            self.wake.engage(time.time())
            self._say("Yes?")
            return

        self.respond(text)

    def _is_own_voice(self, text: str) -> bool:
        """Did the mic pick up Vesper's own speech coming out of the speakers?"""
        if not self._spoken_recently:
            return False
        heard = set(_WORDS.findall(text.lower()))
        if len(heard) < 2:
            return False
        for line in self._spoken_recently:
            said = set(_WORDS.findall(line.lower()))
            if not said:
                continue
            overlap = len(heard & said) / len(heard)
            if overlap >= self.config.echo_similarity:
                return True
        return False

    # --- answering ----------------------------------------------------------

    def _filler(self, pool: tuple[str, ...]) -> str:
        """A short holding phrase, never the same one twice running."""
        options = [f for f in pool if f != self._last_filler] or list(pool)
        chosen = random.choice(options)
        self._last_filler = chosen
        return chosen

    def _say(self, line: str) -> None:
        line = clean_for_speech(line)
        if not line:
            return
        self._spoken_recently.append(line)
        if not self.speaker.speaking:
            self._speaking_since = time.monotonic()
        self.speaker.say(line)

    def respond(self, text: str) -> None:
        """One full turn: ask Claude, speak the answer as it arrives."""
        self.turns += 1
        context = sensors.context_block()
        router = ChannelRouter()
        assembler = SentenceAssembler()
        started = time.monotonic()
        spoke_at: float | None = None
        spoken_anything = False
        said_filler = False
        said_second_filler = False
        screen_buffer = ""

        self.ui.thinking("thinking")
        for event in self.brain.ask(frame_turn(text, context)):
            if isinstance(event, TextDelta):
                last_delta = time.monotonic()
                spoken_delta, screen_delta = router.feed(event.text)
                screen_buffer += screen_delta
                # Emit a screen block once it closes, never delta by delta, or
                # the terminal fills with four-character fragments.
                if screen_buffer and not router.in_screen:
                    self.ui.screen(screen_buffer.strip())
                    screen_buffer = ""
                for sentence in assembler.feed(spoken_delta):
                    if spoke_at is None:
                        spoke_at = time.monotonic()
                    spoken_anything = True
                    self._say(sentence)

            elif isinstance(event, ToolStarted):
                self.ui.tool(event.name, event.detail)
                # Speak as soon as Claude starts working, not when it finishes.
                if not spoken_anything and not said_filler:
                    said_filler = True
                    spoke_at = spoke_at or time.monotonic()
                    self._say(self._filler(THINKING_FILLERS))
                elif (
                    not spoken_anything
                    and time.monotonic() - started > self.config.still_working_after_s
                    and not said_second_filler
                ):
                    said_second_filler = True
                    self._say(self._filler(STILL_WORKING_FILLERS))

            elif isinstance(event, PermissionNeeded):
                self.ui.permission(event.tool, event.detail)

            elif isinstance(event, BrainError):
                self.ui.error(event.message)
                self._say(event.message)
                return

            elif isinstance(event, TurnComplete):
                spoken_tail, screen_tail = router.flush()
                for sentence in assembler.feed(spoken_tail):
                    spoken_anything = True
                    self._say(sentence)
                leftover = (screen_buffer + screen_tail).strip()
                if leftover:
                    self.ui.screen(leftover)
                tail = assembler.flush()
                if tail:
                    if spoke_at is None:
                        spoke_at = time.monotonic()
                    spoken_anything = True
                    self._say(tail)

                # Fallback for a turn that produced no streaming deltas at all.
                # Partial messages can be absent, and silence would look like a
                # crash to the user rather than a missing flag.
                if not spoken_anything and event.text.strip():
                    fallback, screen_only = split_channels(event.text)
                    if screen_only:
                        self.ui.screen(screen_only)
                    if fallback:
                        spoke_at = spoke_at or time.monotonic()
                        self._say(fallback)

                self.wake.engage(time.time())
                self.ui.answered(
                    event,
                    total_s=time.monotonic() - started,
                    first_speech_s=(spoke_at - started) if spoke_at else None,
                )
                return

        # Falling out of the loop without a TurnComplete means the turn was
        # interrupted. Tell Claude, so it does not assume it was heard.
        self.brain.note(INTERRUPTED_NOTE)


class _SilentUI:
    """No-op UI. Keeps the loop usable headless and in tests."""

    def heard(self, text, addressed): pass
    def thinking(self, what): pass
    def tool(self, name, detail): pass
    def screen(self, text): pass
    def permission(self, tool, detail): pass
    def answered(self, turn, total_s, first_speech_s): pass
    def interrupted(self, dropped): pass
    def discarded(self, reason): pass
    def error(self, message): pass
    def warn(self, message): pass
