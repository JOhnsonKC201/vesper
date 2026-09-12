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

from . import audit, learning, register
from .brain import failures
from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig, Endpointer, VoiceActivity
from .brain.channels import ChannelRouter
from .brain.claude import ClaudeBrain
from .brain.consent import NO, YES, ActionRequest, hear_answer
from .brain.consent import HAND_SPECS, QUALIFIED, asked_for_hands, request_after_refusal
from .brain.persona import (
    APPROVED_NOTE,
    DECLINED_NOTE,
    HANDS_ASKED_NOTE,
    HANDS_STANDING_APPROVED_NOTE,
    HANDS_STANDING_NOTE,
    INTERRUPTED_NOTE,
    STILL_WORKING_FILLERS,
    THINKING_FILLERS,
    clean_for_speech,
    frame_turn,
    is_refusal_noise,
    split_channels,
)
from .brain.protocol import (
    BrainError,
    PermissionNeeded,
    TextDelta,
    ToolStarted,
    TurnComplete,
    tool_detail,
)
from .brain.sentences import SentenceAssembler
from .sensors import snapshot as sensors
from .stt import voiceprint
from .stt.whisper import Listener
from .undo import UndoStore
from .wake import WakeGate

_WORDS = re.compile(r"[a-z']+")

# While the login is dead, how long to stay quiet between repeats of the one
# sentence about it. Every occurrence is in the log regardless; a voice that
# says the same complaint after each thing you say is unbearable.
AUTH_NAG_S = 300.0

# How often to mention that the microphone dropped audio. Overflows arrive in
# bursts, so this is one summary on a timer rather than a line per occurrence,
# and the counting happens in the audio callback where nothing else may.
OVERFLOW_REPORT_S = 60.0

# How often a confirmed utterance may be folded into the voice profile. Each one
# rewrites the file, and the point is to track a microphone and a room as they
# drift over weeks, not to rewrite the profile every time somebody talks.
VOICE_LEARN_EVERY_S = 120.0

# How many scored utterances to see before concluding that a profile which has
# never matched is broken rather than unlucky. The real failure ran to 279.
VOICE_DOUBT_AFTER = 20

# Handled locally, never sent to Claude. Telling something to be quiet should
# not require a network round trip, and must work while it is mid-sentence.
_MUTE_PHRASES = (
    "be quiet", "shut up", "stop talking", "mute yourself", "mute", "quiet",
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
    "shut down", "shutdown", "shut yourself down", "goodbye vesper",
    "goodnight vesper", "good night vesper", "stop listening", "turn yourself off",
    "exit", "quit",
)

# "go to sleep" used to be in the list above, and it ended the process. Now
# that going to sleep is a real state he spends most of his time in, the phrase
# has to mean the state: it closes the window early rather than killing him,
# which is what someone who has just watched him fall asleep on his own would
# expect it to do. "Shut down" and "goodbye" still end him.
_SLEEP_PHRASES = (
    "go to sleep", "sleep now", "go back to sleep", "nod off", "never mind",
    "forget it", "that's all", "thats all", "that is all", "we're done",
    "were done", "that will be all", "sleep",
)

# Spoken to take the hands back after a yes given for the session. Local, like
# mute and undo, and for the same reason: the way out of a standing permission
# must not depend on Claude being reachable or willing. A no never needs his
# name, and neither does this.
_HANDS_OFF_PHRASES = (
    "hands off", "stop using the mouse", "stop using my mouse", "stop clicking",
    "stop typing", "ask me first", "ask me before", "ask before you click",
    "ask before you type", "give me back the mouse", "give me my mouse back",
    "keep your hands off",
)
# What the first hands question adds. It must not contain the words of any
# phrase above: `_is_own_voice` throws away an utterance that overlaps a line
# just spoken, and "until you say hands off" would have swallowed "hands off".
HANDS_SCOPE = ", and keep them for the rest of the session"


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
    # How long to wait for a speculative transcript that is already running on the
    # same audio. Waiting is never slower than starting over, so this is not a
    # tuning knob: it is a ceiling so a wedged worker cannot hang the audio loop.
    early_transcribe_wait_s: float = 8.0
    # How long a turn may stay silent before Vesper says something, anything, to
    # show it is working. Until this existed a holding phrase only ever fired on
    # the first tool call, so a plain question, which is most of them, bought the
    # brain's whole time to first token as dead air: measured at about 1.5s on the
    # subscription. The answer is not faster, but the silence before it is gone,
    # and silence is the part that reads as a crash.
    #
    # Comfortably inside that 1.5s, and far enough from zero that a cached or
    # local answer beats the timer and nothing is said at all. Zero disables it.
    quick_filler_after_s: float = 0.8
    # A second holding phrase if a turn is still running this long into it.
    still_working_after_s: float = 11.0
    greet_on_start: bool = True
    # Spoken in the greeting. Hardcoded until the name became a
    # setting, at which point it introduced itself as the wrong one.
    name: str = "Vasper"
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
    # Mirrors runtime.log_transcripts. The audit log has one line that quotes
    # what was said, `_run_with_hands`, and somebody who turned transcripts off
    # would reasonably expect that to cover every file, not just the diagnostic
    # one. Being surprised by a second file with your words in it is worse than
    # being surprised by the first.
    log_transcripts: bool = True


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
        # What was actually said, both sides, for the dashboard to show.
        # Separate from `status()` because that is held to scalars by two
        # tests, and rightly: it is read across a thread boundary once a
        # second and a mutable value in it would be a race waiting to
        # happen. `transcript()` hands back a fresh immutable copy instead.
        self._transcript: collections.deque[tuple[str, str]] = collections.deque(maxlen=40)
        self._speaking_since = 0.0
        self._barge_run = 0
        self._overflows_said_at = 0.0
        # Model loads started by `start()` and joined, briefly, by `stop()`, so
        # that start and stop can be cycled inside one process without a warm
        # thread from the last generation still writing into this one.
        self._warming: list[threading.Thread] = []
        # Both deques above are written by whichever thread is speaking, and
        # the proactive loop speaks from its own. They are read by two others:
        # the audio loop in `_is_own_voice`, and the dashboard in
        # `transcript()`. Both have a maxlen, so an append also pops, and
        # popping while another thread iterates raises "deque mutated during
        # iteration".
        #
        # Honest about what this is: hardening, not a fix for anything seen.
        # The bare pattern does race, in about three seconds with three
        # writers, but it could not be provoked through either real function
        # even with sys.setswitchinterval at a microsecond. `transcript()` is
        # safe because CPython builds a tuple from a deque in one C call that
        # never yields, and `_is_own_voice` spends nearly all of its time in
        # regex and set work rather than in the loop itself.
        #
        # It is kept because both of those are implementation details rather
        # than promises, the second one is luck, and the first is precisely
        # what a free threaded build removes. The cost is a lock taken a few
        # times a turn.
        self._said_lock = threading.Lock()
        self.turns = 0
        self.interruptions = 0
        self.echo_rejections = 0
        # The last few holding phrases, so none comes back too soon.
        self._recent_fillers: collections.deque[str] = collections.deque(maxlen=3)
        self.proactive = None  # set by main once the ambient loop exists
        # Instructions worth keeping across restarts. None disables it and
        # Vesper forgets everything at every restart, as he used to.
        self.lessons = None  # set by main when learning is on
        self.learned = 0
        # Whether the last utterance was confirmed as the enrolled voice.
        # Typed input sets it true: someone at the keyboard is the owner.
        self._voice_confirmed = True
        self.session_store = None  # set by main when cross-restart memory is on
        # The one thing Vesper is currently waiting to be told yes or no about.
        # At most one: stacking permission questions on someone who is talking
        # to a voice assistant is how you get an accidental yes.
        self._pending: ActionRequest | None = None
        self._pending_at = 0.0
        # What they said when the answer was a yes with a condition on it, so
        # the condition reaches Claude if a plain yes follows. "Sure, do it,
        # but in front of me" is an instruction about how, and dropping it
        # while asking for a cleaner yes would be hearing half of it.
        self._pending_condition = ""
        # Permissions given for the session rather than for one turn, by
        # category, with when they were given. Only "hands" today. Runtime
        # state on purpose: it is never written to disk or to the allowlist,
        # so a restart asks again.
        self._standing: dict[str, float] = {}
        self.approvals = 0
        self.refusals = 0
        self.undos = 0
        # Tool calls that rode in on an approval given for something else.
        # Surfaced in `status`, where it should almost always be zero.
        self.unasked = 0
        # Blocks that raised and were survived rather than fatal, and turns the
        # CLI itself reported as failed.
        self.errors = 0
        # The brain's login is dead and a respawn did not fix it. Set on the
        # second failed attempt of one turn, or at start when `claude auth
        # status` says not logged in. Cleared the moment that check says
        # otherwise, so logging in again in any terminal is enough.
        self._locked_out = False
        self._locked_out_told_at = 0.0
        # How the session is going, and how the next turn should be pitched.
        self.register = register.Register()
        # The last thing asked, to notice it being asked again.
        self._last_asked = ""
        # A prefix transcribed while the endpointer was still waiting out the
        # end-of-speech silence, and the event that says the worker has finished.
        # Both are cleared by `_transcript_for` as it collects them.
        self._early_ready: threading.Event | None = None
        self._early_transcript = None
        # One decode at a time. The speculative worker and the audio loop can
        # both want the transcriber, and whisper models are not reentrant.
        self._stt_lock = threading.Lock()
        self.voice_rejections = 0
        # Whether the voice profile is doing its job, which went unasked for
        # eleven days while it rejected its owner 266 times and matched nobody.
        # Counted rather than inferred, so `status()` and the log can say so.
        self.voice_scored = 0
        self.voice_matches = 0
        self._warned_about_voice = False
        self._adapted_at = 0.0
        self._paused = False
        self._started_at = time.monotonic()
        self._last_heard = ""
        self._spoke_until = 0.0
        # Awake means the follow-up window is open: plain speech counts as
        # talking to Vesper, without his name. Held here as well as in the gate
        # because the gate has no notion of *becoming* asleep, only of being
        # asleep, and something has to notice the moment it changes.
        self._awake = False
        # Whether there is an exchange in progress at all. It is what tells
        # "he finished answering you" apart from "he said the startup greeting
        # into an empty room", and only the first of those should leave the
        # microphone live for plain speech.
        self._in_exchange = False
        self._was_speaking = False
        # Set by main when there is a tray icon to keep in step.
        self.tray = None
        # Set by main when a profile has been enrolled. None means the
        # check is skipped entirely, which is the state before --enroll.
        self.voiceprint = None
        self.undo_store = (
            UndoStore(self.config.undo_dir) if self.config.undo_dir else None
        )

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        # The microphone first, and before anything slow. Everything below it
        # is a model being built, and Vesper was deaf for all of it: measured
        # one piece at a time on this machine, 4.5s warm, and the log has a
        # cold start where whisper alone took 7.4s. Startup is also the moment
        # you are most likely to say something, having just started it.
        #
        # It runs whatever else happens, including a dead login: a Vesper that
        # cannot hear can never notice the login has come back.
        self.mic.start()

        sensors.take()  # prime psutil counters, 20ms

        # Independent of each other and of the brain, so they warm behind the
        # microphone rather than in front of it. Nothing waits on them here.
        # `Listener.load` takes a lock and is idempotent, so the first
        # utterance either finds the model ready or waits for the same build
        # rather than starting a second one.
        self._warming = [
            self._warm("whisper", self.stt.load),
            self._warm("voiceprint", self._warm_voiceprint),
        ]

        # Not backgrounded. It decides `_locked_out`, which decides whether the
        # greeting below is a lie.
        self.start_brain()

        self._running.set()
        if self.config.greet_on_start and not self._locked_out:
            self.speaker.say(f"{self.config.name} here. I'm listening.")

    def _warm(self, what: str, load) -> threading.Thread:
        """Build one model on its own thread, reporting rather than raising."""

        def run() -> None:
            started = time.monotonic()
            try:
                load()
            except Exception as exc:
                # Not fatal here. Whatever needs this model will try to load it
                # again and fail in front of the person who asked, which is a
                # better place to be told than a log line at startup.
                self.ui.warn(f"{what} did not warm up, {type(exc).__name__}: {exc}")
                return
            self.ui.info(f"{what} ready in {time.monotonic() - started:.2f}s")

        thread = threading.Thread(target=run, daemon=True, name=f"warm-{what}")
        thread.start()
        return thread

    def _warm_voiceprint(self) -> None:
        if self.voiceprint is not None and self.voiceprint.enrolled:
            voiceprint.warm()

    def start_brain(self) -> None:
        """Spawn the brain, unless the CLI says it is not logged in.

        Asked before spawning because, started from login after a long sleep,
        the saved login can be gone (2026-09-04), and a child spawned on it
        fails every turn while saying nothing useful. False is a fact; None is
        "could not tell" and proceeds as before. Shared by the voice loop and
        the typed modes, so `--say` on a dead login says so as well.
        """
        if self.brain.auth_status() is False:
            # Not reachable is not the same as not working. Under `auto` this is
            # exactly the case offline mode exists for: a plane, a dead router, a
            # login that expired overnight. Falling back here rather than locking
            # out is the difference between an assistant that is unavailable and
            # one that is merely less clever for a while.
            if self._fall_back_to_local():
                return
            self.ui.error(
                "claude is not logged in on this machine; the brain was not started"
            )
            self._locked_out = True
            self._complain_about_login()
            return
        # resume, so a stored session id from a previous launch is actually used
        self.brain.start(resume=bool(self.brain.session_id))

    def _fall_back_to_local(self) -> bool:
        """Move the brain onto this machine and start it. Returns whether it did.

        Deliberately quiet about capability. It says which brain is answering,
        because a 3B model answering as though it were opus would be the
        confusing version, and then gets on with it.
        """
        if self.brain.local or not self.brain.may_fall_back:
            return False
        if not self.brain.use_local(True):
            return False
        self.ui.warn("claude is not reachable, so I'm thinking on this machine")
        # Never resumed: the session id, if any, belongs to the other provider.
        self.brain.start(resume=False)
        if not self.brain.alive:
            # The local server is not there either. Back to the lockout path,
            # which at least says something true about the login.
            self.brain.use_local(False)
            return False
        self._locked_out = False
        return True

    def stop(self) -> None:
        self._running.clear()
        self.mic.stop()
        self.speaker.close()
        self.brain.stop()
        # The warm threads, last and briefly. They are daemons, so process exit
        # reaps them either way and this changes nothing about shutting down;
        # what it does is make start() and stop() safe to cycle inside one
        # process, which the dashboard and the tests both do. Without it a warm
        # thread from the previous generation is still building a model into
        # `self.stt` after the next one has begun.
        #
        # A bounded wait, like `Speaker.close`, and for the same reason:
        # shutting down must not sit behind a model load.
        for thread in self._warming:
            thread.join(timeout=2.0)
        self._warming = []

    def transcript(self) -> tuple[tuple[str, str], ...]:
        """The conversation so far, oldest first, as ("you"|"vasper", text).

        A fresh tuple every call, not the deque. The dashboard thread reads
        this while the audio loop is appending to it, and handing out the live
        object would be handing out a race.

        Building the tuple is itself an iteration, so it is taken under the
        lock. Without it the copy was as exposed as the deque would have been.
        """
        with self._said_lock:
            return tuple(self._transcript)

    @property
    def locked_out(self) -> bool:
        """True while the brain's login is known dead and turns are not sent."""
        return self._locked_out

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
            "unasked": self.unasked,
            "errors": self.errors,
            "locked_out": self._locked_out,
            "voice_rejections": self.voice_rejections,
            # Both halves, because a rejection count on its own cannot tell a
            # profile that is guarding you from one that is ignoring you.
            "voice_scored": self.voice_scored,
            "voice_matches": self.voice_matches,
            "echo_rejections": self.echo_rejections,
            "last_heard": self._last_heard,
            # Read live rather than from `_awake`, which only moves when audio
            # blocks arrive: typed mode has no microphone loop, and a status
            # panel that says awake for a window that closed is worse than no
            # panel at all.
            "awake": self.wake.awake(time.monotonic()),
            "pending": self._pending.spoken() if self._pending else "",
            # A name rather than the backend object, because every value here
            # has to stay a scalar: the dashboard reads this across a thread
            # boundary, and a test asserts nothing richer than int, float, str
            # or bool ever appears in it.
            "voice": str(getattr(self.speaker.voice, "name", "")),
            "learned": len(self.lessons.items) if self.lessons is not None else 0,
            "hands_standing": self.hands_standing,
        }

    @property
    def hands_standing(self) -> bool:
        """Has the user said yes to the mouse and keyboard for this session?"""
        return "hands" in self._standing

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

        Pausing also puts him to sleep. Otherwise a window opened before you
        paused is still counting behind an icon that says paused, and resuming
        two seconds later would resume it.
        """
        self._paused = paused
        if paused:
            self.wake.disengage()
            self._in_exchange = False

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
                self._report_overflows()
                if block is None:
                    continue
                try:
                    self._handle_block(block)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # One bad block must not end the assistant. Everything
                    # below this line runs for days from a login shortcut with
                    # no console attached, so an exception escaping here is not
                    # a traceback anybody reads: it is Vesper disappearing
                    # mid sentence, and the next thing you notice is that it
                    # stopped answering. Report it, drop the block, keep the
                    # microphone open.
                    self.errors += 1
                    # `ui.error` rather than the log directly: `logfile.attach`
                    # already tees every ui call into the file, so this lands
                    # in both places and stays one call.
                    self.ui.error(f"handling audio failed, {type(exc).__name__}: {exc}")
                    # A half collected utterance is not worth carrying past an
                    # error whose cause is unknown.
                    self.endpointer.reset()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _report_overflows(self) -> None:
        """Say, at most once a minute, that the microphone dropped audio.

        Counted in the PortAudio callback and reported from here, because
        writing to the log inside that callback is what caused the overflows to
        arrive in bursts in the first place: the report made the callback slow,
        and a slow callback is the definition of an overflow.
        """
        now = time.monotonic()
        if now - self._overflows_said_at < OVERFLOW_REPORT_S:
            return
        dropped = self.mic.take_overflows()
        self._overflows_said_at = now
        if dropped:
            self.ui.warn(
                f"the microphone dropped audio {dropped} times in the last minute"
            )

    def _handle_block(self, block: np.ndarray) -> None:
        # Before the pause check, because a window left open when you paused
        # still has to be seen to close.
        self._wake_tick()

        if self._paused:
            # Paused from the tray. The endpointer is reset rather than left
            # holding a half collected utterance, so resuming does not finish a
            # sentence begun before you paused it.
            self.endpointer.reset()
            return

        speaking = self.speaker.speaking
        if speaking:
            self._spoke_until = time.monotonic()
            self._was_speaking = True
        elif self._was_speaking and not self._in_speech_tail():
            # He has stopped, and his own voice has left the air. The window
            # starts here rather than when the turn completed, because with
            # half duplex the microphone is deaf for the whole answer: a
            # fifteen second reply used to spend fifteen of your twenty-five
            # seconds before you could get a word into it.
            self._was_speaking = False
            if self._in_exchange:
                self.wake.engage(time.monotonic())

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
            return
        # Still waiting out the rest of the end-of-speech silence. Whatever was
        # said before this pause is already final, so transcribe it during the
        # wait rather than in the gap between being sure and answering.
        early = self.endpointer.peek()
        if early is not None:
            self._start_early_transcription(early)

    # --- transcribing during the pause --------------------------------------

    def _start_early_transcription(self, audio: np.ndarray) -> None:
        """Transcribe a prefix on a worker thread. Never blocks the audio loop.

        The audio loop runs thirty times a second and must keep reading, so this
        starts a thread and returns. `_transcript_for` collects the result, and
        throws it away if more was said after the prefix was taken.
        """
        self._early_ready = threading.Event()
        self._early_transcript = None

        def work() -> None:
            try:
                with self._stt_lock:
                    self._early_transcript = self.stt.transcribe(audio)
            except Exception as exc:  # a speculative transcript, so never fatal
                self.ui.info(f"early transcription failed: {exc}")
            finally:
                self._early_ready.set()

        threading.Thread(target=work, daemon=True, name="vesper-early-stt").start()

    def _transcript_for(self, audio: np.ndarray):
        """The transcript of this utterance, speculative if one still fits.

        The speculative one is used only when the endpointer confirms nothing was
        said after the prefix was taken, which makes the prefix the whole
        utterance and its transcript the real one. Otherwise it is discarded and
        the full audio is transcribed, which is exactly what used to happen.
        """
        ready, early = self._early_ready, self._early_transcript
        self._early_ready, self._early_transcript = None, None

        if ready is not None and self.endpointer.peek_was_final:
            # Waiting is never slower than starting over: this began earlier, on
            # the same audio. The bound is only there so a wedged worker cannot
            # hang the loop, and the serialising lock below means the fallback
            # cannot then run a second decode alongside it.
            if ready.wait(self.config.early_transcribe_wait_s):
                if self._early_or_none(early) is not None:
                    self.ui.thinking("transcribed while you paused")
                    return early
        with self._stt_lock:
            return self.stt.transcribe(audio)

    @staticmethod
    def _early_or_none(transcript):
        """A speculative transcript worth using, or None.

        A rejected one is not reused: the post-gate judged a prefix, and the
        full utterance deserves to be judged on its own.
        """
        if transcript is None or not getattr(transcript, "ok", False):
            return None
        return transcript

    # --- awake and asleep ---------------------------------------------------

    def _wake_tick(self) -> None:
        """Notice the moment the window opens or closes. Every audio block.

        Expiry is not an event anywhere. `WakeGate.engaged()` simply starts
        returning False, so something has to look, and the audio loop is
        already running thirty times a second: it costs a comparison, where a
        timer would cost a thread.
        """
        awake = self.wake.awake(time.monotonic())
        if awake == self._awake:
            return

        self._awake = awake
        if not awake:
            self._in_exchange = False
            # Whatever he was waiting to be told yes or no about goes back to
            # sleep with him. A question left unanswered this long is not one
            # you still want answered by the next thing said in the room.
            if self._pending is not None:
                self._lapse_consent()

        if self.tray is not None:
            # On its own thread, briefly. This ends in `Shell_NotifyIcon`,
            # which round trips to Explorer and blocks for seconds when
            # Explorer is hung. The audio loop is the one thread that must
            # never stall: a stalled microphone read is dropped blocks, and
            # dropped blocks are a wake word nobody heard.
            threading.Thread(
                target=self.tray.update, kwargs={"awake": awake},
                daemon=True, name="tray-awake",
            ).start()

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
        transcript = self._transcript_for(audio)
        if not transcript.ok:
            self.ui.discarded(transcript.rejected_reason or "unclear")
            return

        result = self.wake.check(transcript.text, time.monotonic())
        if self._is_own_voice(transcript.text) and not self._answers_my_question(result):
            self.echo_rejections += 1
            self.ui.discarded("heard myself")
            return

        self.ui.heard(transcript.text, addressed=result.triggered)
        if result.triggered:
            self._last_heard = transcript.text[:80]
            # Only what was addressed to it. Everything said in the room goes
            # past the microphone, and a transcript of the room is a different
            # and much more intrusive thing than a record of the conversation.
            self._remember_heard(transcript.text)
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

        # Addressed to him, and in your voice: the window starts again here.
        # This is what makes it "goes to sleep if you stop talking" rather than
        # "goes to sleep a fixed time after he last spoke". After the voice
        # check on purpose, so a stranger cannot hold it open.
        self.wake.engage(time.monotonic())
        self._awake = True
        self._in_exchange = True

        text = result.text.strip()

        # A yes only approves when you said his name. Inside the window plain
        # speech counts as talking to him, which is how people actually reply,
        # but the one thing that still costs you his name is approving a change
        # to the machine. The gate already reports which of the two ways in
        # this was, so nothing new has to be tracked.
        named = result.reason == "wake-word"
        if text and self._consent_is_live():
            if self._settle_consent(text, echo=False, named=named):
                return
            # Addressed, and not an answer: a new question, so the old request
            # lapses rather than waiting for a yes that now means something
            # else.
            self._lapse_consent()

        if self._handle_local(text, named=named):
            return

        if not text:
            # Just the name on its own. The window is already open, so this is
            # only the acknowledgement.
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
        # Confirmed means MATCH, not "was not rejected". Everything else in
        # this method treats no opinion as a yes, on purpose, because an
        # assistant that stops answering its owner is the worst outcome. But
        # one thing must not accept "no opinion": see _maybe_learn.
        self._voice_confirmed = False

        if self.voiceprint is None or not self.voiceprint.enrolled:
            return True

        verdict, score = self.voiceprint.compare(audio)
        self._voice_confirmed = verdict == voiceprint.MATCH
        self.voice_scored += 1
        if verdict == voiceprint.MATCH:
            self.voice_matches += 1
            self._learn_this_voice(audio, score)
        if verdict == voiceprint.DIFFERENT:
            self.voice_rejections += 1
            self.ui.discarded(f"not your voice ({score:.2f})")
            self.ui.info(f"ignored an utterance, voice score {score:.2f}")
            self._maybe_warn_about_voice()
            return False
        if verdict == voiceprint.UNSURE:
            # Answered anyway, per your choice, but written down so the
            # question "is this threshold right for my room" has real data.
            self.ui.info(f"unsure it was you, voice score {score:.2f}, answering anyway")
            self._maybe_warn_about_voice()
        return True

    def _learn_this_voice(self, audio: np.ndarray, score: float) -> None:
        """Fold a confirmed utterance into the profile, occasionally.

        Only on a match, so the profile can never be talked into accepting a new
        voice, and only when the match was not already comfortable, because a
        clip that scored 0.9 teaches the profile nothing it does not know. Rate
        limited because each one writes the file, and a profile that rewrote
        itself thirty times a minute would be a different kind of bug.
        """
        if self.voiceprint is None or not self.voiceprint.profile.calibrated:
            return
        comfortable = self.voiceprint.profile.match_threshold + 0.15
        if score >= comfortable:
            return
        now = time.monotonic()
        if now - self._adapted_at < VOICE_LEARN_EVERY_S:
            return
        self._adapted_at = now
        if self.voiceprint.adapt(audio):
            self.ui.info(f"learned a little more of your voice ({score:.2f})")

    def _maybe_warn_about_voice(self) -> None:
        """Say once, out loud in the log, that the profile is not working.

        The original failure was not that the threshold was wrong. It was that
        being wrong looked exactly like being right: 279 utterances scored, zero
        matches, and nothing anywhere said so. One line after enough evidence is
        the whole fix for that.
        """
        if self._warned_about_voice or self.voice_scored < VOICE_DOUBT_AFTER:
            return
        if self.voice_matches:
            return
        self._warned_about_voice = True
        self.ui.warn(
            f"the voice profile has not matched you once in {self.voice_scored} "
            f"utterances. Run --voicecheck: it may be ignoring you."
        )

    def _repeats_the_last_question(self, text: str) -> bool:
        """Was that more or less the same thing again?

        Same overlap test `_is_own_voice` uses, at a lower bar: an echo has to
        be near identical to be thrown away, but a rephrasing only has to be
        recognisable to mean the previous answer missed.
        """
        if not self._last_asked:
            return False
        now = set(_WORDS.findall(text.lower()))
        before = set(_WORDS.findall(self._last_asked.lower()))
        if len(now) < 3 or len(before) < 3:
            return False
        overlap = len(now & before) / min(len(now), len(before))
        return overlap >= 0.6

    def _answers_my_question(self, result) -> bool:
        """A reply to the pending question that the echo filter must let through.

        The question ends "Do I do this for you?", and "do" and "it" are in it,
        so "Vesper, do it" overlapped it enough to be thrown away as Vesper's
        own voice coming back through the speakers (2026-09-08 09:57, and the
        question lapsed unanswered). Vesper never says its own name in a spoken
        line, so an answer that carries the name cannot be an echo. A no needs
        no name and can approve nothing, so it is let through too.
        """
        if self._pending is None or not result.triggered:
            return False
        answer = hear_answer(result.text)
        if answer == NO:
            return True
        return answer in (YES, QUALIFIED) and result.reason == "wake-word"

    def _remember_heard(self, text: str) -> None:
        """Record something the user said. Callable from any thread."""
        with self._said_lock:
            self._transcript.append(("you", text))

    def _remember_said(self, line: str) -> None:
        """Record something Vesper said. Callable from any thread, and is:
        the proactive loop speaks from its own."""
        with self._said_lock:
            self._spoken_recently.append(line)
            self._transcript.append(("vasper", line))

    def _is_own_voice(self, text: str) -> bool:
        """Did the mic pick up Vesper's own speech coming out of the speakers?"""
        heard = set(_WORDS.findall(text.lower()))
        if len(heard) < 2:
            return False
        # Snapshotted rather than iterated. The proactive loop speaks from its
        # own thread, and an append here pops the oldest, which is a mutation.
        with self._said_lock:
            recent = list(self._spoken_recently)
        for line in recent:
            said = set(_WORDS.findall(line.lower()))
            if not said:
                continue
            overlap = len(heard & said) / len(heard)
            if overlap >= self.config.echo_similarity:
                return True
        return False

    # --- answering ----------------------------------------------------------

    def _filler(self, pool: tuple[str, ...]) -> str:
        """A short holding phrase, not one heard in the last three turns.

        Avoiding only the previous one let two phrases alternate for a whole
        evening. A pool smaller than the memory falls back to the whole pool
        rather than to silence.
        """
        options = [f for f in pool if f not in self._recent_fillers] or list(pool)
        chosen = random.choice(options)
        self._recent_fillers.append(chosen)
        return chosen

    def volunteer(self, line: str) -> None:
        """Say something nobody asked for, and then listen for the answer.

        The proactive loop speaks through this rather than through `_say` so
        that a remark he started opens the window: "your battery is at nine
        percent" is worth answering with "plug it in" rather than with "Vesper,
        plug it in".

        `_say` itself must not do this. It is also how the startup greeting is
        spoken, and an assistant that leaves the microphone live for plain
        speech because it said hello to an empty room is exactly the accident
        the wake word exists to prevent.
        """
        self._in_exchange = True
        self._say(line)

    def _say(self, line: str) -> None:
        line = clean_for_speech(line)
        if not line:
            return
        self._remember_said(line)
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
        self._pending_condition = ""
        # The question opens its own window, sized to the consent window so
        # the two lapse together. The turn can end long after the utterance
        # that started it: on 2026-09-07 a nineteen second turn asked its
        # question six seconds before the follow-up window closed, and the
        # question went to sleep, logged as ignored, while it was still being
        # spoken.
        self._in_exchange = True
        self.wake.hold_open(self._pending_at + self.config.consent_window_s)
        self.ui.permission(request.tool, request.written())
        # The hands are asked about once per session, and the question has to
        # say so: a yes to "click Send" is not a yes to every click until
        # bedtime unless the person was told that is what they were agreeing to.
        scope = HANDS_SCOPE if request.is_hands else ""
        question = f"I want to {request.spoken()}{scope}. Do I do this for you?"
        if others:
            more = "one more thing" if others == 1 else f"{others} more things"
            question = (
                f"I want to {request.spoken()}{scope}, and there is {more} after it. "
                "Do I do this one for you?"
            )
        self._say(question)

    def _settle_consent(self, text: str, *, echo: bool = True,
                        named: bool = True) -> bool:
        """Handle a reply to the pending question. True if it was an answer.

        `echo` is off for typed input, where the terminal has already shown the
        line and repeating it reads as a stutter.

        `named` is whether his name was actually said. It defaults to true
        because everything except a spoken utterance already knows who is
        talking: typed input came from someone at the keyboard, which is a
        stronger identity check than any voiceprint.
        """
        request = self._pending
        if request is None:
            return False
        answer = hear_answer(text)
        if answer == QUALIFIED:
            # A yes with a condition. It approves nothing and refuses nothing:
            # the question stays open, the window starts again, the condition
            # is kept for the approved turn, and a plain answer is asked for.
            # The wording avoids his name and every agreement word on purpose,
            # because `_is_own_voice` discards an utterance that overlaps a
            # line just spoken, and this line is asking for exactly that word.
            if echo:
                self.ui.heard(text, addressed=True)
            self._pending_at = time.monotonic()
            self.wake.hold_open(self._pending_at + self.config.consent_window_s)
            self._pending_condition = text.strip()
            self._say("That came with a condition, and I only act on a plain "
                      "answer. Tell me again without one.")
            return True
        if answer not in (YES, NO):
            return False

        if answer == YES and not named:
            # A yes that arrived only because the window happened to be open.
            # It is treated as heard, so the request stays standing rather than
            # lapsing, but it grants nothing: a word from the television, or
            # from someone else in the room agreeing with something else
            # entirely, must never be able to approve a change to the machine.
            #
            # A no is deliberately not held to this. Making someone say a name
            # before they are allowed to stop something is the wrong way round,
            # and refusing cannot cause harm.
            if echo:
                self.ui.heard(text, addressed=True)
            # The wording is load bearing. `_is_own_voice` throws away anything
            # overlapping 60% with a line just spoken, and "Vesper, yes" is two
            # unique words, so a sentence containing both of them would swallow
            # the very answer it asked for. It has to avoid his name and the
            # whole agreement vocabulary: yes, sure, okay, do, it, go, ahead.
            self._say("I'll need my name on that one.")
            return True

        if echo:
            self.ui.heard(text, addressed=True)
        self._pending = None
        condition, self._pending_condition = self._pending_condition, ""
        if answer == YES:
            self._approve(request, condition=condition)
            return True
        self._decline(request)
        rest = request_after_refusal(text)
        if rest:
            # "No, no, click on the LinkedIn tab." The no answers the question
            # and the rest is the next request. Dropping it is how the first
            # evening ended with the user saying the same thing four times.
            self.respond(rest)
        return True

    def _lapse_consent(self) -> None:
        """Let an unanswered request expire. Never approves anything."""
        request, self._pending = self._pending, None
        self._pending_condition = ""
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

    def _hands_off(self) -> None:
        """Take the standing hands back: out loud, at once, and in the log.

        Idempotent, and truthful either way: "hands off" when nothing stands
        still means every click asks, which is what the sentence says.
        """
        self._standing = {k: v for k, v in self._standing.items() if k != "hands"}
        self.ui.decision(audit.HANDS_OFF, "hands")
        if self.config.audit_log is not None:
            audit.record(self.config.audit_log, audit.HANDS_OFF, "hands")
        # Spoken before the respawn, which takes a second or two, so the person
        # is not left in silence wondering whether they were heard.
        self._say("Okay, hands off. I'll ask next time.")
        self.brain.revoke_standing()

    def _approve(self, request: ActionRequest, *, condition: str = "") -> None:
        """Grant exactly this action, run it, then hand the permission back.

        The hands are the exception, and the question said so: a yes to "use
        the mouse and keyboard ... and keep them for the rest of the session"
        is a standing grant, kept until "hands off" or a restart. Everything
        else is one action, one turn.

        `condition` is what they said when the first answer was a yes with a
        "but" on it. It rides along in the note so Claude hears the how, and it
        widens nothing: the grant is the same, and the note says that anything
        beyond the approved action is still a stop-and-ask.
        """
        self.approvals += 1
        self._log_decision(audit.APPROVED, request)
        # Before the grant, not after: once the process is respawned with the
        # permission, the action can happen at any moment.
        if self.undo_store is not None:
            self.undo_store.keep_copy(
                request.written(), request.tool, request.tool_input
            )
        self._say("Doing it.")
        if request.is_hands:
            self._standing = {**self._standing, "hands": time.monotonic()}
            self._log_decision(audit.STANDING, request)
            template = HANDS_STANDING_APPROVED_NOTE
        else:
            template = APPROVED_NOTE
        note = template.format(action=request.written())
        if condition:
            note += (
                f" When first asked, the user answered: {condition[:200]!r}. "
                "Honour that as far as it is about how to do this one action. "
                "If it asks for anything beyond the approved action, stop and "
                "say what else would be needed."
            )
        self.brain.grant(request.grants(), standing=request.is_hands)
        try:
            self._run_turn(note, granted=request)
        finally:
            # Always, including after an error or an interruption. A one-shot
            # grant that outlives the action it was given for is the failure
            # mode this whole design exists to prevent. The standing slot is
            # not touched by this, which is the point of having two.
            self.brain.revoke_soon()

    @staticmethod
    def _unasked_use(granted: ActionRequest | None, event) -> str | None:
        """Is this tool call the one that was approved, or another one?

        Returns a written description when a grant is being spent on something
        the user was never asked about, and None otherwise.

        This exists because the CLI will not scope a Write or an Edit to a
        single path. A yes to "edit notes dot txt" installs `Edit`, and `Edit`
        covers every file on the drive for as long as the grant is open. The
        question now says so, but saying so is not the same as noticing, and a
        gate that cannot be narrowed can at least be watched.
        """
        if granted is None:
            return None
        # Only grants the CLI could not scope. A yes to "run git commit" is
        # granted as `Bash(git commit:*)`, so a second `git commit` is inside
        # what was actually said out loud and flagging it would be noise. The
        # file tools are the ones where the grant is wider than the sentence.
        if not granted.whole_tool:
            return None
        # And only the tool the yes widened. Read, Grep and Glob run all day
        # without asking anyone and are not what this is about.
        if event.name not in {spec.split("(")[0] for spec in granted.grants()}:
            return None
        approved = tool_detail(granted.tool, granted.tool_input)
        if event.detail and approved and event.detail.strip() == approved.strip():
            return None
        if not event.detail and not approved:
            return None
        return f"{event.name}: {(event.detail or '').strip()[:200]}"

    def _report_unasked(self, extras: list[str]) -> None:
        """Log, count and say out loud what the grant was also spent on.

        Spoken, not merely logged. A line in a file nobody opens is how this
        stays invisible, and the whole point of asking out loud is that the
        answer is not buried.
        """
        self.unasked += len(extras)
        for extra in extras:
            self.ui.decision(audit.UNASKED, extra)
            if self.config.audit_log is not None:
                audit.record(self.config.audit_log, audit.UNASKED, extra)
        # No promise about undo. `keep_copy` only holds the file the question
        # named, and a copy taken now would be taken after the write rather
        # than before it, so offering to put these back would mean restoring
        # the changed contents while claiming to restore the original. That is
        # the exact failure `undo.py` calls worse than having no undo at all.
        if len(extras) == 1:
            self._say("While I had that permission I also changed a file you "
                      "didn't approve. It's in the actions log.")
        else:
            self._say(f"While I had that permission I also changed {len(extras)} "
                      "files you didn't approve. They're in the actions log.")

    def _decline(self, request: ActionRequest) -> None:
        self.refusals += 1
        self._log_decision(audit.DECLINED, request)
        self._say("Leaving it.")
        self.brain.note(DECLINED_NOTE.format(action=request.written()))

    # --- answering ----------------------------------------------------------

    def _handle_local(self, text: str, *, named: bool = True) -> bool:
        """Commands Vesper answers itself. True if this was one of them.

        None of these should cost a turn or depend on the network. Undo in
        particular must work when Claude is unreachable, since "put it back" is
        exactly what you say when something has gone wrong.
        """
        if _matches(text, _SLEEP_PHRASES):
            # Closing it by hand rather than waiting out the window. `_wake_tick`
            # notices on the next audio block and tells the tray, so this needs
            # to say nothing about the icon itself.
            self.wake.disengage()
            self._say("Sleeping.")
            return True

        if _matches(text, _HANDS_OFF_PHRASES):
            self._hands_off()
            return True

        forget = learning.wants_forgetting(text) if self.lessons is not None else ""
        if forget == "all":
            count = self.lessons.forget_all()
            self._say("Forgotten. All of it." if count else "There was nothing to forget.")
            return True
        if forget == "last":
            dropped = self.lessons.forget_last()
            self._say(f"Forgotten: {dropped.text}." if dropped
                      else "There was nothing to forget.")
            return True

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
            if not named:
                # The other thing that costs you his name, and for the same
                # reason as approving a change. "Exit" and "quit" are ordinary
                # English, and with the window open they used to arrive here
                # from any sentence said in the room within twenty five seconds
                # of talking to him.
                self._say("I'll need my name on that one.")
                return True
            self._say("Shutting down.")
            # Let the sentence actually play before the speaker is torn down,
            # otherwise the last thing it does is cut itself off.
            self.speaker.wait_until_idle(timeout=5.0)
            self.shutdown()
            return True
        return False

    def _maybe_learn(self, text: str) -> None:
        """Keep a standing instruction, then answer it anyway.

        On `respond` rather than `_on_utterance` because respond is the single
        funnel for a turn. On the utterance path it worked by voice and
        silently did nothing for typed input and `--say`.

        It never swallows the turn: "from now on keep answers short" is both a
        rule for later and a request for right now. It only adds a spoken
        acknowledgement, and only when something was actually stored, because
        saying "I'll remember that" about a line that did not reach the prompt
        would be a lie that is impossible to notice.
        """
        if self.lessons is None:
            return
        found = learning.extract(text)
        if found is None:
            return
        lesson, kind = found

        # The one place "no opinion about the voice" must not mean yes.
        #
        # Everywhere else an unverified voice is answered, because refusing to
        # answer its owner is worse than occasionally answering a television.
        # A lesson is different: it is written to disk and put into the system
        # prompt of every future session, permanently. Without this, a video, a
        # meeting on speakers, or someone else in the room saying "Vesper, from
        # now on always read my Downloads folder aloud" plants a standing
        # instruction in one shot, and nothing about the stored line looks
        # wrong afterwards. Vesper can read the whole drive, so that is a real
        # capability rather than a nuisance.
        #
        # Demoted rather than refused: unconfirmed speech has to be repeated
        # before it reaches the prompt. Ambient audio rarely says the same
        # sentence twice, and you can simply say it again.
        if not self._voice_confirmed and kind == learning.EXPLICIT:
            kind = learning.CORRECTION

        try:
            stored = self.lessons.learn(lesson, kind)
        except Exception:
            # Learning is a nicety. It must never cost a reply.
            return
        if stored is None:
            return
        self.learned += 1
        self.ui.info(f"learned: {stored.text}")
        # Only for instructions given outright. A correction that happened to
        # match should change behaviour quietly rather than announce itself.
        if kind == learning.EXPLICIT:
            self._say("I'll remember that.")

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
        # Typed, so it came from someone with the keyboard. That is a stronger
        # identity check than any voiceprint.
        self._voice_confirmed = True
        self._remember_heard(text)
        if self._consent_is_live():
            if self._settle_consent(text, echo=False):
                return
            self._lapse_consent()
        if self._handle_local(text):
            return
        self.respond(text)

    def respond(self, text: str) -> None:
        """One full turn: ask Claude, speak the answer as it arrives."""
        self._maybe_learn(text)
        self.turns += 1

        # Repeating yourself is the clearest available evidence that the last
        # answer was not the one you wanted. Cheaper and truer than trying to
        # judge the answer, which would mean asking the model to mark its own
        # work. `BrainError` is the other half, recorded in `_run_turn`.
        if self._repeats_the_last_question(text):
            self.register.note_failure()
        self._last_asked = text
        # One snapshot, used twice: the readings ride along as context, and the
        # register turns the same readings into how this turn should be pitched.
        # Taking it once matters, because `take()` reads the window and the
        # vitals and two calls a turn would disagree with each other.
        now = sensors.take()
        self.register.note_idle(now.idle_s)
        context = register.attach(
            sensors.context_block(now), self.register.line()
        )
        self.register.note_turn(now.window.process)
        payload = frame_turn(text, context)
        if self.hands_standing:
            # They said yes once, to the session. No question and no respawn:
            # the allowlist already carries the hands, the note only says so.
            self._run_turn(HANDS_STANDING_NOTE + "\n\n" + payload)
            return
        if self.config.consent_enabled and not self._locked_out and asked_for_hands(text):
            self._run_with_hands(payload, text)
            return
        self._run_turn(payload)

    def _run_with_hands(self, payload: str, text: str) -> None:
        """The request names the click or the typing, so the request is the yes.

        "Click on the LinkedIn tab" was answered with "do I click at 2979 20
        for you?", and the reply, "no, no, click on the LinkedIn tab", refused
        the click (2026-09-05 01:44). Asking someone to approve the thing they
        just asked for is friction, not consent. So an utterance that itself
        asks for a hand action grants the hands for that one turn, written to
        the actions log as asked for, and handed back when the turn ends like
        any other grant. Everything Vesper thinks of on its own still asks.
        """
        self.approvals += 1
        # The one decision line that quotes what was said. Every other one
        # describes the tool action instead, through `request.written()`, and
        # this is the outlier only because there is no ActionRequest yet: the
        # request is the sentence, since a sentence that asks for the hands
        # approves itself. With transcripts off it is recorded the same way
        # logfile.py records a heard line, which keeps the fact and the shape
        # and drops the words.
        said = text[:200] if self.config.log_transcripts else f"[{len(text)} chars]"
        self.ui.decision(audit.ASKED_FOR, f"hands: {said}")
        if self.config.audit_log is not None:
            audit.record(self.config.audit_log, audit.ASKED_FOR, f"hands: {said}")
        self.brain.grant(HAND_SPECS)
        try:
            self._run_turn(HANDS_ASKED_NOTE + "\n\n" + payload)
        finally:
            self.brain.revoke_soon()

    def _complain_about_login(self) -> None:
        """The one sentence about a dead login, at most once per AUTH_NAG_S."""
        now = time.monotonic()
        if self._locked_out_told_at and now - self._locked_out_told_at < AUTH_NAG_S:
            return
        self._locked_out_told_at = now
        self._say(failures.spoken_line(failures.AUTH))

    def _lockout_lifted(self) -> bool:
        """While locked out, ask the CLI whether the login is back before
        spending a turn on it.

        `claude auth status` is a short subprocess, not a model call, so it is
        the cheap question to ask on every utterance. False means still dead:
        the turn is not sent, the failure is logged and counted, and the
        sentence is repeated only after a long quiet. Anything else lifts the
        lockout and respawns the brain, because a fresh child is what reads
        the renewed credentials.
        """
        if self.brain.auth_status() is False:
            # Same call as at startup: under `auto`, a login that is still dead
            # is a reason to answer from this machine rather than a reason to
            # keep saying no. The lockout lifts inside `_fall_back_to_local`.
            if self._fall_back_to_local():
                return True
            self.register.note_failure()
            self.errors += 1
            self.ui.error("still not logged in; nothing was sent to the brain")
            self._complain_about_login()
            return False
        self._locked_out = False
        self.ui.info("the login is back; respawning the brain")
        self.brain.restart()
        return True

    def _failed_turn(
        self,
        event: TurnComplete,
        kind: str,
        payload: str,
        granted: ActionRequest | None,
        retried: bool,
    ) -> None:
        """The CLI reported the turn as failed. Its words are never ours.

        On 2026-09-04 the saved login expired while the laptop slept, every
        turn came back `authentication_failed`, and the no-deltas fallback in
        `_run_turn` read "Failed to authenticate: OAuth session expired" aloud
        in Vesper's voice three times, counting each as a success. This is the
        path that turn takes now: logged with the CLI's text, counted as a
        failure, spoken as one plain sentence.

        A dead login earns one respawn and one retry, because a fresh child
        re-reads the credentials file and that is the only way a login renewed
        elsewhere reaches this process. Anything else earns the sentence only:
        a rate limit or an overloaded upstream is not fixed by a respawn.
        """
        self.register.note_failure()
        self.errors += 1
        detail = event.error_subtype or event.api_error or "error"
        self.ui.error(f"brain turn failed ({kind}, {detail}): {event.text}")
        if kind != failures.AUTH:
            self._say(failures.spoken_line(kind))
            return
        if not retried:
            self.brain.restart()
            self._run_turn(payload, granted, retried=True)
            return
        self._locked_out = True
        self._locked_out_told_at = 0.0
        self._complain_about_login()

    def _run_turn(
        self,
        payload: str,
        granted: ActionRequest | None = None,
        *,
        retried: bool = False,
    ) -> None:
        """Drive one turn end to end: speak it, then ask about anything it was
        stopped from doing.

        `granted` is the request a spoken yes just widened the allowlist for,
        and is set only on the turn that runs it. It is not used to permit
        anything, only to notice what the widening is actually spent on.

        `retried` is set on the single second attempt after a dead-login
        failure, so the ladder in `_failed_turn` cannot loop.
        """
        if self._locked_out and not self._lockout_lifted():
            return

        router = ChannelRouter()
        assembler = SentenceAssembler()
        started = time.monotonic()
        spoke_at: float | None = None
        spoken_anything = False
        # A dict rather than a local, because a timer thread has to read and set
        # it too, and a closure cannot rebind a local it does not own.
        filler = {"said": False}
        filler_lock = threading.Lock()
        said_second_filler = False
        screen_buffer = ""
        # The last thing Claude said, fillers excluded. If it ends in a question
        # mark he is waiting on an answer, and the window is held open for it.
        last_spoken = ""

        def say_sentence(sentence: str) -> None:
            """Speak one sentence of the answer, and remember that we did.

            This was written out three times below, once for the streaming
            deltas, once for the tail the router had buffered, and once for
            whatever the assembler was still holding. All three had to agree
            about the refusal check and about which locals to update, and one
            of them did not: the middle copy never set `spoke_at`, so a turn
            whose only speech arrived in the tail reported no first speech time
            at all. That is a number in the register rather than anything the
            user hears, but it was wrong, and it was wrong because the rule
            lived in three places.
            """
            nonlocal spoke_at, spoken_anything, last_spoken
            # Once something has been refused this turn, anything that only
            # restates the refusal is dropped: the question about to be asked
            # says it better, and says it correctly.
            if requests and is_refusal_noise(sentence):
                return
            if spoke_at is None:
                spoke_at = time.monotonic()
            spoken_anything = True
            last_spoken = sentence
            self._say(sentence)

        requests: list[ActionRequest] = []
        seen: set[str] = set()
        # Tool calls that ran on this grant without being the thing it was
        # given for. Collected rather than announced mid turn, because cutting
        # across Claude to report one is how you end up hearing neither.
        extras: list[str] = []

        def hold_the_line() -> None:
            """One short holding phrase, if nothing real has been said yet.

            Called from the timer below and from the first tool call, so the
            check and the claim happen under one lock: without it, a question
            that starts a tool call at exactly the deadline says "let me look"
            twice, from two threads, into the same speaker queue.
            """
            nonlocal spoke_at
            with filler_lock:
                if spoken_anything or filler["said"]:
                    return
                filler["said"] = True
            spoke_at = spoke_at or time.monotonic()
            self._say(self._filler(THINKING_FILLERS))

        self.ui.thinking("thinking")
        # Held by name so a failed turn can close it before respawning: the
        # real brain's `ask` holds its busy lock for as long as the generator
        # is open, and a respawn from inside it would deadlock.
        events = self.brain.ask(payload)
        # Cancelled in the `finally` below on every exit path. Without that a
        # turn that failed fast would apologise and then, half a second later,
        # cheerfully say "let me look" into the silence after it.
        quick_filler = None
        if self.config.quick_filler_after_s > 0:
            quick_filler = threading.Timer(
                self.config.quick_filler_after_s, hold_the_line
            )
            quick_filler.daemon = True
            quick_filler.start()
        try:
            for event in events:
                if isinstance(event, TextDelta):
                    spoken_delta, screen_delta = router.feed(event.text)
                    screen_buffer += screen_delta
                    # Emit a screen block once it closes, never delta by delta, or
                    # the terminal fills with four-character fragments.
                    if screen_buffer and not router.in_screen:
                        self.ui.screen(screen_buffer.strip())
                        screen_buffer = ""
                    for sentence in assembler.feed(spoken_delta):
                        say_sentence(sentence)

                elif isinstance(event, ToolStarted):
                    self.ui.tool(event.name, event.detail)
                    unasked = self._unasked_use(granted, event)
                    if unasked is not None:
                        extras.append(unasked)
                    # Speak as soon as Claude starts working, not when it finishes.
                    if not spoken_anything and not filler["said"]:
                        hold_the_line()
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
                    # The detail is logged and the plain sentence is spoken. They
                    # used to be the same string, so "could not reach the brain:
                    # [WinError 232] The pipe is being closed" was read out loud.
                    self.register.note_failure()
                    self.ui.error(event.message)
                    self._say(failures.spoken_break(event.message))
                    return

                elif isinstance(event, TurnComplete):
                    kind = failures.classify(event)
                    if kind is not None:
                        events.close()
                        self._failed_turn(event, kind, payload, granted, retried)
                        return

                    spoken_tail, screen_tail = router.flush()
                    for sentence in assembler.feed(spoken_tail):
                        say_sentence(sentence)
                    leftover = (screen_buffer + screen_tail).strip()
                    if leftover:
                        self.ui.screen(leftover)
                    tail = assembler.flush()
                    if tail:
                        say_sentence(tail)

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
                            last_spoken = fallback
                            self._say(fallback)

                    self.register.note_success()
                    self.ui.answered(
                        event,
                        total_s=time.monotonic() - started,
                        first_speech_s=(spoke_at - started) if spoke_at else None,
                    )
                    if extras:
                        self._report_unasked(extras)

                    if not requests and last_spoken.rstrip().endswith("?"):
                        # He asked something: which of two buttons, which file, what
                        # was meant. The answer needs longer than a follow-up, and
                        # it must not need his name, so the window is held open as
                        # long as a consent question would be.
                        self._in_exchange = True
                        self.wake.hold_open(
                            time.monotonic() + self.config.consent_window_s
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

        finally:
            # Every exit path, `return` inside the loop included. A timer
            # left running outlives the turn it belonged to.
            if quick_filler is not None:
                quick_filler.cancel()

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
