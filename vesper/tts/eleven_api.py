"""A thin ElevenLabs client, and the free tier facts that shape it.

Everything here was measured against a real key rather than read off a docs
page, because the docs describe the product and the free tier is a different
product wearing the same API.

Three findings drove the design:

**PCM comes back raw, so there is no decoder in this project.** Asking for
`pcm_22050` returns headerless little-endian int16 at exactly Piper's sample
rate, so it drops straight into the same `sounddevice` stream through a
`np.frombuffer`. Asking for mp3 would have meant a decoder, and the only one
installed is PyAV, pinned to an old version because Windows Application Control
blocks the newer wheels.

**Most of the voice catalogue is not available on a free key.** Library voices
answer 402 `paid_plan_required`, and so do two of the twenty voices the
dashboard calls premade: Aria and Charlotte. The other eighteen work. That is
why `KNOWN_VOICES` is a constant rather than a fetch: a free key cannot list
voices at all, since `voices_read` is refused, and even a key that could would
list hundreds it is not allowed to speak with.

**Creating a voice is refused entirely on free**, with 403
`feature_not_available`, whatever permissions the key carries. So the designer
is built and says it needs a paid plan, rather than failing in a way that looks
like a bug.

Nothing here raises out of `speak`. Every failure becomes an `ElevenError`
carrying a short reason, and the caller falls back to Piper.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

BASE = "https://api.elevenlabs.io/v1"

# Half the credit cost of the standard models and the lowest latency, which is
# the only sensible default when the budget is 10,000 characters a month.
DEFAULT_MODEL = "eleven_flash_v2_5"

# Matches Piper's rate, so no resampling and no decoder.
OUTPUT_FORMAT = "pcm_22050"
SAMPLE_RATE = 22_050


@dataclass(frozen=True)
class RemoteVoice:
    """One voice the picker can offer."""

    voice_id: str
    name: str
    blurb: str = ""
    free: bool = True

    @property
    def label(self) -> str:
        suffix = "" if self.free else "  (paid plan)"
        return f"{self.name}{suffix}"


# Verified one at a time against a free key on 2026-08-31 by synthesizing a word
# with each. `free=False` means it answered 402, so the picker can show it as
# unavailable rather than letting you choose something that will never speak.
KNOWN_VOICES: tuple[RemoteVoice, ...] = (
    RemoteVoice("JBFqnCBsd6RMkjVDRZzb", "George", "warm British narrator"),
    RemoteVoice("onwK4e9ZLuTAKqWW03F9", "Daniel", "measured British news reader"),
    RemoteVoice("nPczCjzI2devNBz1zQrb", "Brian", "deep American narrator"),
    RemoteVoice("N2lVS1w4EtoT3dr4eOWO", "Callum", "gravelly, unhurried"),
    RemoteVoice("IKne3meq5aSn9XLyUdCD", "Charlie", "casual Australian"),
    RemoteVoice("TX3LPaxmHKxFdv7VOQHJ", "Liam", "bright American"),
    RemoteVoice("bIHbv24MWmeRgasZH58o", "Will", "easy, conversational"),
    RemoteVoice("cjVigY5qzO86Huf0OWal", "Eric", "smooth American"),
    RemoteVoice("iP95p4xoKVk53GoZ742B", "Chris", "plain and direct"),
    RemoteVoice("CwhRBWXzGAHq8TQ4Fs17", "Roger", "older, steady"),
    RemoteVoice("pqHfZKP75CvOlQylNhV4", "Bill", "gruff documentary"),
    RemoteVoice("SAz9YHcvj6GT2YYXdXww", "River", "calm, neutral"),
    RemoteVoice("EXAVITQu4vr4xnSDxMaL", "Sarah", "soft American"),
    RemoteVoice("FGY2WhTYpPnrIDTdsKH5", "Laura", "bright, quick"),
    RemoteVoice("Xb7hH8MSUJpSbSDYk0k2", "Alice", "clear British"),
    RemoteVoice("XrExE9yKIg1WjnnlVkGX", "Matilda", "friendly American"),
    RemoteVoice("cgSgspJ2msm6clMCkdW9", "Jessica", "young, expressive"),
    RemoteVoice("pFZP5JQG7iQjIQuC4Bku", "Lily", "warm British"),
    RemoteVoice("9BWtsMINqrJLrRacOk9x", "Aria", "expressive American", free=False),
    RemoteVoice("XB0fDUnXU5powFXDhCwa", "Charlotte", "Swedish English", free=False),
)

# George. British, composed, and the closest of the eighteen to what the local
# `jarvis` character was reaching for with a filter.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def voice_named(voice_id: str) -> str:
    """A human name for an id, falling back to the id itself."""
    for voice in KNOWN_VOICES:
        if voice.voice_id == voice_id:
            return voice.name
    return voice_id[:8] if voice_id else "none"


class ElevenError(Exception):
    """A failure the caller falls back from, never surfaces as a crash.

    `reason` is a short stable token so callers can branch on it and the
    dashboard can explain itself. It never contains the API key, and neither
    does the string form.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# A key that came back in a response body is still a key. Validation errors in
# particular echo request fields, and `detail` reaches var/vesper.log through
# `str(exc)` wherever a failure is reported. Found by a review: the test that
# claimed to cover this fed classify() a body that never contained a key, so
# the assertion could not fail.
_KEYLIKE = re.compile(r"\bsk[-_][A-Za-z0-9_-]{8,}")


def redact(text: str) -> str:
    """Strip anything key-shaped out of text that is about to be logged."""
    return _KEYLIKE.sub("[redacted]", text or "")


def classify(status: int, body: bytes) -> ElevenError:
    """Turn an HTTP failure into something the UI can say out loud."""
    text = ""
    try:
        text = redact(body.decode("utf-8", "replace"))[:300]
    except Exception:
        pass
    if status == 401:
        return ElevenError("unauthorized", "the API key was rejected")
    if status == 402:
        return ElevenError("paid-voice", "this voice needs a paid ElevenLabs plan")
    if status == 403:
        return ElevenError("paid-feature", "this needs a paid ElevenLabs plan")
    if status == 422:
        return ElevenError("bad-request", text)
    if status == 429:
        return ElevenError("rate-limited", "too many requests")
    if status >= 500:
        return ElevenError("server-error", f"ElevenLabs returned {status}")
    return ElevenError("http-error", f"{status}: {text}")


class ElevenClient:
    """Talks to the ElevenLabs API. Owns no audio and no state but the key."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 20.0,
        base: str = BASE,
        transport=None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_s = float(timeout_s)
        self.base = base.rstrip("/")
        # Only tests pass this. httpx.MockTransport keeps the suite off the
        # network, which is a rule this repo already holds everywhere else.
        self._transport = transport
        self._shared_client = None
        self._client_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        """One client for the life of this object, so the connection is kept.

        There used to be a fresh `httpx.Client` per call, and `stream_pcm` is
        called once per *sentence*, because Vesper starts speaking before the
        turn has finished. A three sentence answer therefore meant three DNS
        lookups, three TCP handshakes and three TLS handshakes to the same
        host, in front of somebody waiting to hear the first word.

        Callers must not close it. `close()` does that, once.
        """
        import httpx  # late: heavy, and the rest of Vesper never needs it

        with self._client_lock:
            if self._shared_client is None:
                self._shared_client = httpx.Client(
                    timeout=self.timeout_s,
                    headers={"xi-api-key": self.api_key},
                    transport=self._transport,
                )
            return self._shared_client

    def close(self) -> None:
        """Let go of the connection. Safe to call more than once."""
        with self._client_lock:
            client, self._shared_client = self._shared_client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - nothing useful left to do
                pass

    # --- speech -------------------------------------------------------------

    def stream_pcm(self, text: str, voice_id: str, *, model_id: str = DEFAULT_MODEL):
        """Yield raw int16 PCM chunks at 22050Hz. Raises ElevenError only."""
        if not self.configured:
            raise ElevenError("no-key", "no ElevenLabs API key is set")

        import httpx

        url = f"{self.base}/text-to-speech/{voice_id}/stream"
        try:
            client = self._client()
            with client.stream(
                "POST",
                url,
                params={"output_format": OUTPUT_FORMAT},
                json={"text": text, "model_id": model_id},
            ) as response:
                if response.status_code != 200:
                    raise classify(response.status_code, response.read())
                for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk
        except ElevenError:
            raise
        except httpx.TimeoutException:
            raise ElevenError("timeout", f"no response in {self.timeout_s:.0f}s")
        except httpx.HTTPError as exc:
            # Deliberately the exception type rather than its message: httpx
            # messages are long, and this string reaches the log.
            raise ElevenError("network", type(exc).__name__)

    # --- catalogue ----------------------------------------------------------

    def list_voices(self) -> tuple[RemoteVoice, ...]:
        """The account's voices, or the verified fallback list.

        A free key is refused `voices_read` outright, so returning the static
        list is the normal case here rather than an error path.
        """
        if not self.configured:
            return KNOWN_VOICES

        import httpx

        try:
            client = self._client()
            response = client.get(f"{self.base}/voices", params={"page_size": 100})
            if response.status_code != 200:
                return KNOWN_VOICES
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return KNOWN_VOICES

        known = {v.voice_id: v for v in KNOWN_VOICES}
        found: list[RemoteVoice] = []
        for item in payload.get("voices") or []:
            voice_id = str(item.get("voice_id") or "")
            if not voice_id:
                continue
            previous = known.get(voice_id)
            found.append(
                RemoteVoice(
                    voice_id=voice_id,
                    name=str(item.get("name") or voice_id[:8]),
                    blurb=str(item.get("description") or "")[:60],
                    # The API lists voices it will then refuse to speak with, so
                    # anything already measured as 402 stays marked as paid.
                    free=previous.free if previous else True,
                )
            )
        return tuple(found) or KNOWN_VOICES

    # --- designing a voice --------------------------------------------------

    def design(self, description: str, sample_text: str) -> tuple[dict, ...]:
        """Candidate voices from a written description.

        Refused on the free tier with 403, reported as `paid-feature` rather
        than dressed up as a bug.
        """
        if not self.configured:
            raise ElevenError("no-key", "no ElevenLabs API key is set")
        # The endpoint rejects anything shorter, and a 422 landing halfway
        # through a design is a worse experience than being told up front.
        if len(sample_text) < 100:
            raise ElevenError("short-text", "the sample needs at least 100 characters")

        import httpx

        try:
            client = self._client()
            response = client.post(
                f"{self.base}/text-to-voice/design",
                json={
                    "voice_description": description,
                    "model_id": "eleven_ttv_v3",
                    "output_format": OUTPUT_FORMAT,
                    "text": sample_text,
                },
            )
            if response.status_code != 200:
                raise classify(response.status_code, response.content)
            payload = response.json()
        except ElevenError:
            raise
        except httpx.TimeoutException:
            raise ElevenError("timeout", f"no response in {self.timeout_s:.0f}s")
        except (httpx.HTTPError, ValueError) as exc:
            raise ElevenError("network", type(exc).__name__)
        return tuple(payload.get("previews") or ())

    def keep(self, generated_voice_id: str, name: str, description: str) -> str:
        """Save a designed voice to the account. Returns its permanent id."""
        if not self.configured:
            raise ElevenError("no-key", "no ElevenLabs API key is set")

        import httpx

        try:
            client = self._client()
            response = client.post(
                f"{self.base}/text-to-voice/create",
                json={
                    "voice_name": name,
                    "voice_description": description,
                    "generated_voice_id": generated_voice_id,
                },
            )
            if response.status_code not in (200, 201):
                raise classify(response.status_code, response.content)
            return str(response.json().get("voice_id") or "")
        except ElevenError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ElevenError("network", type(exc).__name__)
