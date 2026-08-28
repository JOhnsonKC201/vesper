"""Windows SAPI voice: the fallback that always works.

This machine only has the legacy David/Zira/Mark voices, which sound like a
2005 satnav. It exists so Vesper is never mute: if the Piper model is missing,
corrupt, or the onnxruntime wheel fails on a future Python, the assistant still
talks while you fix it.

SAPI is COM, so every thread that touches it must CoInitialize first. The worker
thread in Speaker is long-lived, so we initialise lazily on first use from
whichever thread calls speak().
"""

from __future__ import annotations

import threading

# SpVoice.Speak flags
_ASYNC = 1
_PURGE_BEFORE_SPEAK = 2


class SapiTTS:
    """Speaks through the Windows Speech API."""

    def __init__(self, *, voice_hint: str = "", rate: int = 1, volume: int = 100) -> None:
        self.name = "sapi"
        self._voice_hint = voice_hint
        self._rate = max(-10, min(10, rate))
        self._volume = max(0, min(100, volume))
        self._voice = None
        self._thread_id: int | None = None
        self._lock = threading.Lock()

    def _ensure_voice(self):
        """Create the COM object, once, on the calling thread."""
        current = threading.get_ident()
        if self._voice is not None and self._thread_id == current:
            return self._voice

        import pythoncom
        import win32com.client

        try:
            pythoncom.CoInitialize()
        except Exception:
            pass  # already initialised on this thread

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = self._rate
        voice.Volume = self._volume

        if self._voice_hint:
            for token in voice.GetVoices():
                if self._voice_hint.lower() in token.GetDescription().lower():
                    voice.Voice = token
                    break

        self._voice = voice
        self._thread_id = current
        return voice

    # --- Voice protocol -----------------------------------------------------

    def speak(self, text: str, stop: threading.Event) -> None:
        text = text.strip()
        if not text or stop.is_set():
            return
        with self._lock:
            voice = self._ensure_voice()
            # Speak asynchronously so we can poll the stop event. A synchronous
            # Speak() cannot be interrupted, which would break barge-in.
            voice.Speak(text, _ASYNC)
            while True:
                if stop.is_set():
                    voice.Speak("", _PURGE_BEFORE_SPEAK)
                    return
                # WaitUntilDone returns True when finished, False on timeout.
                if voice.WaitUntilDone(50):
                    return

    def close(self) -> None:
        self._voice = None
        self._thread_id = None

    # --- discovery ----------------------------------------------------------

    @staticmethod
    def available() -> bool:
        try:
            import pythoncom
            import win32com.client

            try:
                pythoncom.CoInitialize()
            except Exception:
                pass
            win32com.client.Dispatch("SAPI.SpVoice")
            return True
        except Exception:
            return False

    @staticmethod
    def list_voices() -> list[str]:
        try:
            import pythoncom
            import win32com.client

            try:
                pythoncom.CoInitialize()
            except Exception:
                pass
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            return [t.GetDescription() for t in voice.GetVoices()]
        except Exception:
            return []
