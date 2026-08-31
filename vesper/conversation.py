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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audit
from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig, Endpointer, VoiceActivity
from .brain.channels import ChannelRouter
from .brain.claude import ClaudeBrain
from .brain.consent import NO, YES, ActionRequest, hear_answer
from .brain.persona import (
    APPROVED_NOTE,
    DECLINED_NOTE,
    INTERRUPTED_NOTE,
    STILL_WORKING_FILLERS,
    THINKING_FILLERS,
    clean_for_speech,
    frame_turn,
    is_refusal_noise,
    split_channels,
)
from .brain.protocol import BrainError, PermissionNeeded, TextDelta, ToolStarted, TurnComplete
from .brain.sentences import SentenceAssembler
from .sensors import snapshot as sensors
from .stt import voiceprint
from .stt.whisper import Listener
from .undo import UndoStore
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
# Handled locally, like the mute phrases, and for a stronger reason: undoing a
# mistake must not depend on a network round trip, on Claude being willing, or
# on it correctly working out what it changed a minute ago. Vesper knows exactly
# what it copied aside, so it does the restore itself.
_UNDO_PHRASES = (
    "undo that", "undo", "undo it", "put it back", "revert that", "take that back",
    "roll that back", "change it back", "put that back", "reverse that",
)
# Also local, and for a blunter reason than the others: once Vesper starts from
# your login there is no console to press ctrl-c in. Without a spoken way out,
# stopping it means Task Manager, which skips the exit path entirely and leaves
# the claude child running.
_SHUTDOWN_PHRASES = (
    "shut down", "shutdown", "shut yourself down", "go to sleep", "goodbye vesper",
    "goodnight vesper", "good night vesper", "stop listening", "turn yourself off",
    "exit", "quit",
)


def _matches(text: str, phrases: tuple[str, ...]) -> bool:
    stripped = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return any(stripped == p or stripped.startswith(p + " ") for p in phrases)


def _without_refusal_noise(text: str) -> str:
    """Drop whole sentences that only restate a refusal, keep the rest."""
    if not text.strip():
        return ""
    splitter = SentenceAssembler()
    kept = [s for s in splitter.feed(text) if not is_refusal_noise(s)]
    last = splitter.flush()
    if last and not is_refusal_noise(last):
        kept.append(last)
    return " ".join(kept).strip()


@dataclass
class ConversationConfig:
    # Half duplex mutes the mic while speaking, and briefly after. On by
    # default because most machines have their speakers a few inches from their
    # microphone, where full duplex means Vesper hears its own voice and cuts
    # itself off mid-sentence on nearly every reply. Losing the ability to talk
    # over it is a much smaller loss than that. Turn it off for headphones.
    half_duplex: bool = True
    # Consecutive speech blocks needed to cut Vesper off. Three 30ms blocks is
    # 90ms of real voice, short enough to feel instant and long enough that a
    # cough or a keyboard clack does not stop it mid-sentence.
    barge_in_blocks: int = 3
    barge_in_threshold: float = 0.75
    # Ignore the mic for a moment after playback starts, so the leading edge of
    # our own voice cannot register as an interruption.
    self_mute_ms: int = 250
    # And for a moment after it stops. `speaking` clears when the last block is
    # written to the device, not when the last sound leaves the speaker, so
    # without this the tail of Vesper's own voice is still in the air with the
    # microphone already listening again.
    tail_mute_ms: int = 350
    echo_similarity: float = 0.6
    # A second holding phrase if a turn is still running this long into it.
    still_working_after_s: float = 11.0
    greet_on_start: bool = True
    # Ask before anything that changes the machine. Turning this off does not
    # make Vesper act without asking, it makes it unable to act at all: the
    # refusal still happens in the CLI, there is simply no way to answer it.
    consent_enabled: bool = True
    # How long a spoken "yes" still refers to the thing that was asked. Long
    # enough to think about it, short enough that a yes to something else forty
    # seconds later cannot land on a stale request.
    consent_window_s: float = 45.0
    # Where approvals and refusals are written. None disables the log.
    audit_log: Path | None = None
    # Where copies of changed files are kept, so "undo that" can put them back.
    # None disables it, which makes every approved change permanent.
    undo_dir: Path | None = None


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
        # A separate, stricter detector for interruptions, still separate
        # because reusing the endpointer's would corrupt its state machine
        # mid-utterance. Built on first use rather than at startup: it is a
        # second copy of the Silero model, worth about 10MB, and while half
        # duplex is on the barge-in path never runs at all, so on most machines
        # it was pure resident cost for a feature that was switched off.
        self._barge_vad_instance = None

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
        # The one thing Vesper is currently waiting to be told yes or no about.
        # At most one: stacking permission questions on someone who is talking
        # to a voice assistant is how you get an accidental yes.
        self._pending: ActionRequest | None = None
        self._pending_at = 0.0
        self.approvals = 0
        self.refusals = 0
        self.undos = 0
        self.voice_rejections = 0
        self._paused = False
        self._started_at = time.monotonic()
        self._last_heard = ""
        self._spoke_until = 0.0
        # Set by main when a profile has been enrolled. None means the
        # check is skipped entirely, which is the state before --enroll.
        self.voiceprint = None
        self.undo_store = (
            UndoStore(self.config.undo_dir) if self.config.undo_dir else None
        )

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        sensors.take()  # prime psutil counters
        self.stt.load()
        if self.voiceprint is not None and self.voiceprint.enrolled:
            voiceprint.warm()
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

    def status(self) -> dict:
        """What it has been doing, as plain data.

        Read once a second from the dashboard thread, so it returns a snapshot
        of simple values rather than anything the caller could hold and read
        while the audio loop mutates it underneath them.
        """
        return {
            "paused": self._paused,
            "uptime_s": time.monotonic() - self._started_at,
            "turns": self.turns,
            "cost_usd": getattr(self.brain, "total_cost_usd", 0.0),
            "interruptions": self.interruptions,
            "approvals": self.approvals,
            "refusals": self.refusals,
            "undos": self.undos,
            "voice_rejections": self.voice_rejections,
            "echo_rejections": self.echo_rejections,
            "last_heard": self._last_heard,
            "pending": self._pending.spoken() if self._pending else "",
            # A name rather than the backend object, because every value here
            # has to stay a scalar: the dashboard reads this across a thread
            # boundary, and a test asserts nothing richer than int, float, str
            # or bool ever appears in it.
            "voice": str(getattr(self.speaker.voice, "name", "")),
        }

    def shutdown(self) -> None:
        """Ask the loop to end. Safe to call from any thread.

        The tray menu, a spoken "shut down" and a SIGTERM all arrive here, and
        two of the three are on other threads. It only clears the running flag:
        the loop then falls out of `run()` on its own and `stop()` happens in
        the one place it always did, so the session is still written and the
        claude child is still terminated.
        """
        self._running.clear()

    def pause(self, paused: bool = True) -> None:
        """Stop or resume acting on what the microphone hears.

        The stream stays open rather than being torn down, because reopening a
        device is where audio stacks go wrong, and because the point of pausing
        is that resuming is instant.
        """
        self._paused = paused

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
        if self._paused:
            # Paused from the tray. The endpointer is reset rather than left
            # holding a half collected utterance, so resuming does not finish a
            # sentence begun before you paused it.
            self.endpointer.reset()
            return

        speaking = self.speaker.speaking
        if speaking:
            self._spoke_until = time.monotonic()

        if speaking or self._in_speech_tail():
            if self.config.half_duplex:
                # Deaf while talking, and for a moment afterwards. The tail
                # matters as much as the speech: when the worker finishes
                # writing, the sound card still has audio buffered and playing,
                # so unmuting the instant `speaking` clears hands the endpointer
                # the last syllable of Vesper's own voice. Reset rather than
                # merely skip, or that syllable becomes the start of the next
                # utterance.
                self.endpointer.reset()
                return
            if speaking and self._detect_barge_in(block):
                self._interrupt()
            # Fall through: the interrupting audio is also the start of the
            # next utterance, so it must still reach the endpointer.
        else:
            self._barge_run = 0

        # The accessor, not the array. Built only when an utterance actually
        # opens, rather than 33 times a second and thrown away.
        utterance = self.endpointer.feed(block, preroll=self.mic.preroll)
        if utterance is not None:
            self._on_utterance(utterance)

    # --- barge-in -----------------------------------------------------------

    def _in_speech_tail(self) -> bool:
        """Is Vesper's own voice probably still coming out of the speakers?"""
        if not self._spoke_until:
            return False
        return (time.monotonic() - self._spoke_until) * 1000 < self.config.tail_mute_ms

    @property
    def _barge_vad(self):
        if self._barge_vad_instance is None:
            self._barge_vad_instance = VoiceActivity(
                threshold=self.config.barge_in_threshold
            )
        return self._barge_vad_instance

    @_barge_vad.setter
    def _barge_vad(self, detector) -> None:
        # Settable so a test can inject one. Losing that to save ten megabytes
        # would be a poor trade.
        self._barge_vad_instance = detector

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
        if result.triggered:
            self._last_heard = transcript.text[:80]
        if not result.triggered:
            # Not addressed to Vesper, so a pending question stays pending and
            # unanswered. Both halves matter: a "yes" said to someone else in
            # the room must not approve anything, and a word said across the
            # room must not throw away the request you were about to approve.
            return

        # Only now, on an utterance that was actually addressed to Vesper. A
        # handful of times an hour rather than thirty times a second, which is
        # what keeps voice matching off the idle budget entirely.
        if not self._is_the_right_voice(audio):
            return

        text = result.text.strip()

        # Asking the question engages the follow-up window, so a bare "yes"
        # counts as addressed for the next few seconds, which is how people
        # actually reply. After that window it takes "Vesper, yes", and the
        # wake word is stripped before it gets here either way.
        if text and self._consent_is_live():
            if self._settle_consent(text, echo=False):
                return
            # Addressed, and not an answer: a new question, so the old request
            # lapses rather than waiting for a yes that now means something
            # else.
            self._lapse_consent()

        if self._handle_local(text):
            return

        if not text:
            # Just the name on its own. Acknowledge and open the window.
            self.wake.engage(time.time())
            self._say("Yes?")
            return

        self.respond(text)

    def _is_the_right_voice(self, audio: np.ndarray) -> bool:
        """Was that you? Only a confident no is treated as a no.

        Nothing enrolled, a clip too short to judge, or any failure at all all
        answer yes. The failure being avoided is an assistant that stops
        answering its owner, which is far worse than one that occasionally
        answers a video, and it is the same call the wake word already makes.
        """
        if self.voiceprint is None or not self.voiceprint.enrolled:
            return True

        verdict, score = self.voiceprint.compare(audio)
        if verdict == voiceprint.DIFFERENT:
            self.voice_rejections += 1
            self.ui.discarded(f"not your voice ({score:.2f})")
            self.ui.info(f"ignored an utterance, voice score {score:.2f}")
            return False
        if verdict == voiceprint.UNSURE:
            # Answered anyway, per your choice, but written down so the
            # question "is this threshold right for my room" has real data.
            self.ui.info(f"unsure it was you, voice score {score:.2f}, answering anyway")
        return True

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

    # --- consent ------------------------------------------------------------

    def _consent_is_live(self) -> bool:
        if self._pending is None:
            return False
        if time.monotonic() - self._pending_at > self.config.consent_window_s:
            self._lapse_consent()
            return False
        return True

    def _log_decision(self, decision: str, request: ActionRequest) -> None:
        self.ui.decision(decision, request.written())
        if self.config.audit_log is not None:
            audit.record(self.config.audit_log, decision, request.written())

    def _ask_consent(self, request: ActionRequest, others: int = 0) -> None:
        """Say what Vesper wants to do, and open the window to answer it."""
        if not request.approvable:
            # Nothing to put to the user. Either the command could not be read
            # with confidence, or it runs so many things that no one spoken
            # sentence could describe it honestly. Both used to end in a bare
            # `Bash` grant off the back of "run a command", which is the widest
            # possible permission from the vaguest possible question.
            self._log_decision(audit.DECLINED, request)
            self._say(
                f"I'd have to {request.spoken()}, and I can't put that to you "
                "clearly enough to ask, so I'm leaving it."
            )
            self.brain.note(DECLINED_NOTE.format(action=request.written()))
            return

        self._pending = request
        self._pending_at = time.monotonic()
        self.ui.permission(request.tool, request.written())
        question = f"I want to {request.spoken()}. Do I do this for you?"
        if others:
            more = "one more thing" if others == 1 else f"{others} more things"
            question = (
                f"I want to {request.spoken()}, and there is {more} after it. "
                "Do I do this one for you?"
            )
        self._say(question)
        # Keep the follow-up window open too, so a longer answer than "yes"
        # still counts as talking to Vesper rather than near it.
        self.wake.engage(time.time())

    def _settle_consent(self, text: str, *, echo: bool = True) -> bool:
        """Handle a reply to the pending question. True if it was an answer.

        `echo` is off for typed input, where the terminal has already shown the
        line and repeating it reads as a stutter.
        """
        request = self._pending
        if request is None:
            return False
        answer = hear_answer(text)
        if answer not in (YES, NO):
            return False
        if echo:
            self.ui.heard(text, addressed=True)
        self._pending = None
        if answer == YES:
            self._approve(request)
        else:
            self._decline(request)
        return True

    def _lapse_consent(self) -> None:
        """Let an unanswered request expire. Never approves anything."""
        request, self._pending = self._pending, None
        if request is not None:
            self._log_decision(audit.IGNORED, request)

    def undo_last(self) -> bool:
        """Put back whatever the last approved action changed."""
        if self.undo_store is None:
            self._say("I'm not keeping copies, so there's nothing to put back.")
            return False
        done, sentence = self.undo_store.undo_latest()
        if done:
            self.undos += 1
            if self.config.audit_log is not None:
                audit.record(self.config.audit_log, audit.UNDONE, sentence)
            self.ui.decision(audit.UNDONE, sentence)
        self._say(sentence)
        return done

    def _approve(self, request: ActionRequest) -> None:
        """Grant exactly this action, run it, then hand the permission back."""
        self.approvals += 1
        self._log_decision(audit.APPROVED, request)
        # Before the grant, not after: once the process is respawned with the
        # permission, the action can happen at any moment.
        if self.undo_store is not None:
            self.undo_store.keep_copy(
                request.written(), request.tool, request.tool_input
            )
        self._say("Doing it.")
        self.brain.grant(request.grants())
        try:
            self._run_turn(APPROVED_NOTE.format(action=request.written()))
        finally:
            # Always, including after an error or an interruption. A grant that
            # outlives the action it was given for is the failure mode this
            # whole design exists to prevent.
            self.brain.revoke_soon()

    def _decline(self, request: ActionRequest) -> None:
        self.refusals += 1
        self._log_decision(audit.DECLINED, request)
        self._say("Leaving it.")
        self.brain.note(DECLINED_NOTE.format(action=request.written()))

    # --- answering ----------------------------------------------------------

    def _handle_local(self, text: str) -> bool:
        """Commands Vesper answers itself. True if this was one of them.

        None of these should cost a turn or depend on the network. Undo in
        particular must work when Claude is unreachable, since "put it back" is
        exactly what you say when something has gone wrong.
        """
        if _matches(text, _MUTE_PHRASES):
            if self.proactive is not None:
                self.proactive.mute()
            self.speaker.barge_in()
            self._say("Quiet from now on.")
            return True
        if _matches(text, _UNMUTE_PHRASES):
            if self.proactive is not None:
                self.proactive.unmute()
            self._say("Back on.")
            return True
        if _matches(text, _UNDO_PHRASES):
            self.undo_last()
            return True
        if _matches(text, _SHUTDOWN_PHRASES):
            self._say("Shutting down.")
            # Let the sentence actually play before the speaker is torn down,
            # otherwise the last thing it does is cut itself off.
            self.speaker.wait_until_idle(timeout=5.0)
            self.shutdown()
            return True
        return False

    def hear(self, text: str) -> None:
        """One turn from text that is already transcribed or typed.

        The single door into a turn for anything that is not the microphone.
        Typed input has to come through here rather than calling respond()
        directly: without it, typing "yes" to a permission question would be
        sent to Claude as a new question and the request would silently lapse.
        """
        text = text.strip()
        if not text:
            return
        if self._consent_is_live():
            if self._settle_consent(text, echo=False):
                return
            self._lapse_consent()
        if self._handle_local(text):
            return
        self.respond(text)

    def respond(self, text: str) -> None:
        """One full turn: ask Claude, speak the answer as it arrives."""
        self.turns += 1
        self._run_turn(frame_turn(text, sensors.context_block()))

    def _run_turn(self, payload: str) -> None:
        """Drive one turn end to end: speak it, then ask about anything it was
        stopped from doing."""
        router = ChannelRouter()
        assembler = SentenceAssembler()
        started = time.monotonic()
        spoke_at: float | None = None
        spoken_anything = False
        said_filler = False
        said_second_filler = False
        screen_buffer = ""

        requests: list[ActionRequest] = []
        seen: set[str] = set()

        self.ui.thinking("thinking")
        for event in self.brain.ask(payload):
            if isinstance(event, TextDelta):
                spoken_delta, screen_delta = router.feed(event.text)
                screen_buffer += screen_delta
                # Emit a screen block once it closes, never delta by delta, or
                # the terminal fills with four-character fragments.
                if screen_buffer and not router.in_screen:
                    self.ui.screen(screen_buffer.strip())
                    screen_buffer = ""
                for sentence in assembler.feed(spoken_delta):
                    # Once something has been refused this turn, anything that
                    # only restates the refusal is dropped: the question about
                    # to be asked says it better, and says it correctly.
                    if requests and is_refusal_noise(sentence):
                        continue
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
                # Collected, not asked about yet. The turn is still running and
                # Claude usually has a sentence to say about what it was doing;
                # interrupting that to ask a question would talk over it.
                request = ActionRequest(
                    tool=event.tool,
                    tool_input=event.tool_input,
                    tool_use_id=event.tool_use_id,
                    message=event.message,
                )
                if request.key not in seen:
                    seen.add(request.key)
                    requests.append(request)

            elif isinstance(event, BrainError):
                self.ui.error(event.message)
                self._say(event.message)
                return

            elif isinstance(event, TurnComplete):
                spoken_tail, screen_tail = router.flush()
                for sentence in assembler.feed(spoken_tail):
                    if requests and is_refusal_noise(sentence):
                        continue
                    spoken_anything = True
                    self._say(sentence)
                leftover = (screen_buffer + screen_tail).strip()
                if leftover:
                    self.ui.screen(leftover)
                tail = assembler.flush()
                if tail and not (requests and is_refusal_noise(tail)):
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
                    if requests:
                        # Sentence by sentence, not as one blob: a reply that
                        # ends in "I need permission" usually starts with
                        # something worth hearing.
                        fallback = _without_refusal_noise(fallback)
                    if fallback:
                        spoke_at = spoke_at or time.monotonic()
                        self._say(fallback)

                self.wake.engage(time.time())
                self.ui.answered(
                    event,
                    total_s=time.monotonic() - started,
                    first_speech_s=(spoke_at - started) if spoke_at else None,
                )
                if requests and self.config.consent_enabled:
                    # Only the first is put to the user; the rest are recorded
                    # as refused rather than vanishing. They were being dropped
                    # entirely, which left the audit log claiming to hold every
                    # decision while quietly missing some.
                    for ignored in requests[1:]:
                        self._log_decision(audit.DECLINED, ignored)
                    self._ask_consent(requests[0], others=len(requests) - 1)
                elif requests:
                    for refused in requests:
                        self._log_decision(audit.DECLINED, refused)
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
    def decision(self, decision, action): pass
    def answered(self, turn, total_s, first_speech_s): pass
    def interrupted(self, dropped): pass
    def discarded(self, reason): pass
    def error(self, message): pass
    def warn(self, message): pass
    def info(self, message): pass
