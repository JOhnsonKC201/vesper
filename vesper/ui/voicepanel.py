"""What the dashboard needs in order to draw a voice picker.

The dashboard knows nothing about ElevenLabs, Piper or SAPI and should not. It
renders a `snapshot` dict for the counters and, if one is supplied, one of these
objects for the voice section. Everything specific to a speech provider lives
here, so adding a second one meant writing another one of these rather than
editing the window, which is exactly what `LocalVoicePanel` is.

There are two because they change voice in genuinely different ways. `ElevenTTS`
holds a voice id it re-reads on every request, so switching is assigning a
string. A local backend *is* the voice: switching means loading a different
model and handing it to the Speaker, which is why that one takes the Speaker
rather than a voice.

Both answer the same six questions, and the window asks nothing else: `list`,
`current`, `choose`, `preview`, `budget` and `status`.

It is also the seam that keeps the picker testable. Driving a real Tk window is
not something this suite does, so the tests exercise this and the dashboard's
threading rules separately.
"""

from __future__ import annotations

# Long enough to actually judge a voice, and short enough that trying six of
# them is roughly a tenth of the free monthly allowance rather than half of it.
SAMPLE_LINE = "Good evening. Everything is where you left it, and the house is quiet."


class VoicePanel:
    """Adapts an `ElevenTTS` to what the dashboard asks for."""

    def __init__(self, voice, *, choice_path=None, log=None) -> None:
        import threading

        self.voice = voice
        self.choice_path = choice_path
        self._log = log or (lambda message: None)
        self._voices = None
        # A real stop event rather than a fresh one per call. The first version
        # passed `threading.Event()` inline, which nothing anywhere could ever
        # set, so a preview could not be cut off by closing the window or by
        # anything else. Every other utterance in the system shares the
        # Speaker's stop event; this one is the panel's own, and `cancel()` is
        # what the dashboard calls when it closes.
        self._stop = threading.Event()

    # --- what the dashboard calls -------------------------------------------

    def list(self):
        """Every voice worth offering. Fetched once, then remembered.

        Cached because the dashboard rebuilds this on every open and a free key
        cannot list voices anyway, so the answer is a constant in practice.
        """
        if self._voices is None:
            try:
                self._voices = list(self.voice.client.list_voices())
            except Exception:
                from ..tts import eleven_api

                self._voices = list(eleven_api.KNOWN_VOICES)
        return self._voices

    def current(self) -> str:
        from ..tts import eleven_api

        return eleven_api.voice_named(self.voice.voice_id)

    def choose(self, voice_id: str) -> None:
        """Switch, and remember it across restarts."""
        self.voice.use(voice_id)
        if self.choice_path is not None:
            from .. import voicechoice

            voicechoice.save(
                self.choice_path, voice_id, self.current(), engine="elevenlabs"
            )
        self._log(f"voice switched to {self.current()}")

    def preview(self, voice_id: str) -> str:
        """Speak the sample line in a candidate voice. Returns what to show.

        Called on a worker thread by the dashboard, because this makes a
        network call. `say_as` takes the same lock as ordinary speech, so a
        preview waits its turn rather than talking over a reply in progress.
        """
        self._stop.clear()
        before = self.voice.budget.spent
        self.voice.say_as(SAMPLE_LINE, voice_id, self._stop)
        spent = self.voice.budget.spent - before
        if self._stop.is_set():
            return "stopped"

        reason = self.voice.last_reason
        if reason == "cache":
            return "played from the cache, no charge"
        if reason in ("cloud", ""):
            left = max(0, self.voice.budget.cap - self.voice.budget.spent)
            return f"{spent} characters, {left:,} left this month"
        return _explain(reason)

    def cancel(self) -> None:
        """Cut a preview short. Safe from any thread."""
        self._stop.set()

    def budget(self) -> tuple[int, int]:
        return self.voice.budget.spent, self.voice.budget.cap

    def status(self) -> str:
        """The steady line under the picker, shown when nothing just happened.

        Here rather than in the dashboard because it is the one sentence in the
        section that is specific to the provider, and the local panel's is not
        about an allowance at all.
        """
        spent, cap = self.budget()
        if cap <= 0:
            return "no allowance set; speaking locally"
        if spent >= cap:
            return "monthly allowance spent, speaking locally until next month"
        return f"{cap - spent:,} of {cap:,} characters left this month"


def _explain(reason: str) -> str:
    """Turn a failure token into something worth reading in a window."""
    return {
        "no-key": "no ElevenLabs key set, spoke locally",
        "budget": "monthly allowance spent, spoke locally",
        "unauthorized": "the API key was refused",
        "paid-voice": "that voice needs a paid plan",
        "paid-feature": "that needs a paid plan",
        "rate-limited": "too many requests, try again shortly",
        "timeout": "ElevenLabs did not answer, spoke locally",
        "network": "no connection, spoke locally",
        "server-error": "ElevenLabs had a problem, spoke locally",
    }.get(reason, f"spoke locally ({reason})")


class LocalVoicePanel:
    """The same picker over the backends that need no network.

    This exists because the window used to be given a panel only when the cloud
    voice was running, and the default install is Piper. The comment where that
    was decided said a local picker would be "a control with one entry and
    nothing to say", which was wrong twice: this machine has three deliveries
    per model, and every Windows install has David, Zira and Mark whether or not
    a Piper model was ever downloaded. So the setting simply was not there, and
    changing voice meant editing config.yaml and restarting.

    It takes the `Speaker` rather than a voice. A local backend is not a handle
    on a service that can be re-pointed; it holds a loaded model and an open
    output stream, so switching means building a new one and swapping it in.
    """

    def __init__(self, speaker, *, voices_dir, speed: float = 1.0,
                 volume: float = 1.0, choice_path=None, log=None) -> None:
        import threading

        self.speaker = speaker
        self.voices_dir = voices_dir
        self.speed = speed
        self.volume = volume
        self.choice_path = choice_path
        self._log = log or (lambda message: None)
        self._catalogue = None
        self._stop = threading.Event()

    # --- what the dashboard calls -------------------------------------------

    def list(self):
        """Everything installed. Enumerating SAPI is a COM call, so once only."""
        if self._catalogue is None:
            from ..tts import catalog

            try:
                self._catalogue = list(catalog.discover(self.voices_dir))
            except Exception:
                self._catalogue = []
        return self._catalogue

    def current(self) -> str:
        """The label of the voice speaking now, or "" if it is not one of ours."""
        from ..tts import catalog

        found = catalog.find(self.list(), catalog.current_id(self.speaker.voice))
        return found.label if found is not None else ""

    def choose(self, voice_id: str) -> None:
        """Load it, hand it to the Speaker, and remember it across restarts."""
        from ..tts import catalog

        record = catalog.find(self.list(), voice_id)
        if record is None:
            raise LookupError(f"no such voice: {voice_id}")

        self.speaker.use_voice(self._build(record))
        if self.choice_path is not None:
            from .. import voicechoice

            voicechoice.save(
                self.choice_path, record.voice_id, record.name, engine=record.engine
            )
        self._log(f"voice switched to {record.label}")

    def preview(self, voice_id: str) -> str:
        """Speak the sample line in a candidate without committing to it.

        Called on a worker thread by the dashboard: loading a Piper model is
        the better part of a second, and doing it inline would freeze the
        window. It builds its own backend and closes it again, so hearing a
        voice you then decide against leaves nothing behind.
        """
        from ..tts import catalog

        record = catalog.find(self.list(), voice_id)
        if record is None:
            return "no such voice"

        self._stop.clear()
        # Wait for a reply in progress rather than talking over it. The cloud
        # panel gets this for free by sharing ElevenTTS's lock; there is no
        # shared lock here, because the preview is a different object entirely.
        self.speaker.wait_until_idle(timeout=3.0)
        if self._stop.is_set():
            return "stopped"

        try:
            voice = self._build(record)
        except Exception as exc:
            return f"{record.name} would not load ({type(exc).__name__})"

        try:
            voice.speak(SAMPLE_LINE, self._stop)
        except Exception as exc:
            return f"{record.name} would not play ({type(exc).__name__})"
        finally:
            try:
                voice.close()
            except Exception:
                pass

        return "stopped" if self._stop.is_set() else f"that is {record.name}"

    def cancel(self) -> None:
        """Cut a preview short. Safe from any thread."""
        self._stop.set()

    def budget(self) -> tuple[int, int]:
        """No allowance, because nothing is being bought. The bar is left off."""
        return 0, 0

    def status(self) -> str:
        voices = self.list()
        if not voices:
            return "no voices installed"

        models = {v.voice_id.partition(":")[0] for v in voices if v.engine == "piper"}
        line = f"{len(voices)} voices here, and none of them need the network"
        if len(models) < 2:
            # The one genuinely useful thing to say to someone looking at a
            # short list is the command that makes it longer.
            line += ". More: python -m piper.download_voices NAME --data-dir var/voices"
        return line

    def _build(self, record):
        from ..tts import catalog

        return catalog.build(
            record, voices_dir=self.voices_dir, speed=self.speed, volume=self.volume
        )
