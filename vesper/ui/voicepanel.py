"""What the dashboard needs in order to draw a voice picker.

The dashboard knows nothing about ElevenLabs and should not. It renders a
`snapshot` dict for the counters and, if one is supplied, this object for the
voice section. Everything specific to a speech provider lives here, so adding a
second one later means writing another one of these rather than editing the
window.

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

            voicechoice.save(self.choice_path, voice_id, self.current())
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
