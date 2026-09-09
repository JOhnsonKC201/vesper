"""The cloud voice: its cache, its budget, and every way it falls back.

The behaviour worth defending here is not that ElevenLabs works. It is that
when ElevenLabs does not work, nothing else notices. A refused key, an empty
budget, a dropped connection and a timeout must all end with Piper speaking the
same sentence, because `conversation.py` handles "Quiet from now on." and
"Shutting down." locally on purpose so they survive an outage. If a network
failure could make Vesper silent, an outage would take away the ability to shut
him up.

No test here touches the network. Requests go through `httpx.MockTransport` and
playback through a stand-in `sounddevice`, matching how the rest of this suite
already fakes native dependencies.
"""

import json
import sys
import threading
import types

import numpy as np
import pytest

from vesper.tts import eleven, eleven_api
from vesper.tts.budget import Budget
from vesper.tts.cache import AudioCache, key_for
from vesper.tts.eleven import STOCK_PHRASES, ElevenTTS
from vesper.tts.eleven_api import ElevenClient, ElevenError

from conftest import FakeVoice


# --- fakes ------------------------------------------------------------------


class FakeOutputStream:
    """Collects what was written instead of making noise."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written: list[np.ndarray] = []
        self.started = False
        self.aborted = False
        self.closed = False

    def start(self):
        self.started = True

    def write(self, block):
        self.written.append(np.asarray(block).copy())

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True

    @property
    def samples(self) -> np.ndarray:
        if not self.written:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self.written)


@pytest.fixture
def audio(monkeypatch):
    """Install a stand-in sounddevice and hand back the streams it opened."""
    module = types.ModuleType("sounddevice")
    module.streams = []

    def OutputStream(**kwargs):
        stream = FakeOutputStream(**kwargs)
        module.streams.append(stream)
        return stream

    module.OutputStream = OutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def transport_returning(status: int, body: bytes = b"", *, chunks=None):
    """An httpx transport that answers every request the same way."""
    import httpx

    def handle(request: httpx.Request) -> httpx.Response:
        if chunks is not None:
            return httpx.Response(status, stream=httpx.ByteStream(b"".join(chunks)))
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handle)


def transport_raising(exception):
    import httpx

    def handle(request: httpx.Request) -> httpx.Response:
        raise exception

    return httpx.MockTransport(handle)


def pcm(n_samples: int, value: int = 1000) -> bytes:
    return np.full(n_samples, value, dtype="<i2").tobytes()


def build(tmp_path, *, transport=None, key="sk_test", cap=9_000, voice_id="V1"):
    fallback = FakeVoice()
    client = ElevenClient(key, base="https://example.invalid/v1", transport=transport)
    voice = ElevenTTS(
        fallback,
        client=client,
        cache=AudioCache(tmp_path / "cache"),
        budget=Budget(tmp_path / "budget.json", cap=cap),
        voice_id=voice_id,
    )
    return voice, fallback


# --- the cache --------------------------------------------------------------


def test_the_cache_normalises_whitespace_and_case_but_not_punctuation():
    """"One moment." and "one  moment." are the same line. "One moment" is not."""
    assert key_for("One moment.") == key_for("one  moment.")
    assert key_for("One moment.") == key_for("  One Moment.  ")
    assert key_for("One moment.") != key_for("One moment")


def test_a_cached_line_never_reaches_the_network(tmp_path, audio):
    """The whole point. A stock phrase must cost nothing after the first time."""
    voice, fallback = build(tmp_path, transport=transport_raising(RuntimeError("boom")))
    voice.cache.put("V1", "One moment.", pcm(2048))

    voice.speak("One moment.", threading.Event())

    assert voice.last_reason == "cache"
    assert fallback.lines == [], "a cache hit fell through to piper"
    assert voice.budget.spent == 0, "a cache hit was charged for"
    assert audio.streams[0].samples.size == 2048


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path, audio):
    voice, fallback = build(tmp_path, key="")
    path = voice.cache._path("V1", "hello")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    voice.speak("hello", threading.Event())

    assert fallback.lines == ["hello"], "an empty cache file should read as a miss"


def test_the_cache_evicts_least_recently_used_first(tmp_path):
    """Least recently *used*, which is not the same as least recently written.

    The first version only inserted and checked that the oldest insert went.
    That is FIFO, and it passed with `get()`'s `path.touch()` deleted, so a
    regression turning LRU into FIFO would have shipped green. This reads an
    early entry back to promote it, which is the only thing that tells the two
    policies apart.
    """
    import time

    # Each entry is 2048 bytes, so a 4096 cap holds exactly two. Three entries
    # means exactly one eviction, which is what makes the choice observable.
    # An earlier draft wrote six into the same cap, evicting four, so both
    # candidates went whatever the policy was.
    cache = AudioCache(tmp_path / "c", max_bytes=4096)
    cache.put("V1", "oldest but used", pcm(1024))
    time.sleep(0.05)
    cache.put("V1", "oldest and unused", pcm(1024))

    # NTFS mtime resolution is coarse enough to tie two writes in a tight
    # loop, and a tie makes the ordering arbitrary rather than wrong.
    time.sleep(0.05)
    assert cache.get("V1", "oldest but used") is not None
    time.sleep(0.05)

    cache.put("V1", "newest", pcm(1024))

    assert cache.size() <= 4096
    assert cache.has("V1", "newest")
    assert cache.has("V1", "oldest but used"), "reading it did not protect it"
    assert not cache.has("V1", "oldest and unused")


def test_a_month_change_while_running_resets_the_allowance(tmp_path, monkeypatch):
    """The rollover that actually happens to an always-on process.

    `test_a_new_month_resets_the_allowance` writes a stale month and builds a
    fresh Budget, so it exercises `_load`, not `_roll`. Deleting `_roll`'s body
    left it green. This one moves the calendar under a live object.
    """
    from vesper.tts import budget as budget_module

    budget = Budget(tmp_path / "b.json", cap=9_000)
    budget.spend(5_000)
    assert budget.spent == 5_000

    monkeypatch.setattr(budget_module, "_this_month", lambda: "2099-12")

    assert budget.spent == 0, "the allowance did not reset when the month did"
    assert budget.allows(9_000)


def test_a_voice_id_cannot_escape_the_cache_directory(tmp_path):
    """Voice ids come from config and from an API, so they are not trusted."""
    cache = AudioCache(tmp_path / "c")
    cache.put("../../evil", "x", pcm(64))

    written = list((tmp_path / "c").rglob("*.pcm"))
    assert written, "nothing was written at all"
    for path in written:
        assert (tmp_path / "c") in path.parents


# --- the budget -------------------------------------------------------------


def test_the_budget_refuses_a_line_that_would_cross_the_cap(tmp_path):
    budget = Budget(tmp_path / "b.json", cap=100)
    budget.spend(90)

    assert budget.allows(10)
    assert not budget.allows(11), "whole lines only; no switching voice mid-sentence"


def test_the_budget_survives_a_restart(tmp_path):
    path = tmp_path / "b.json"
    Budget(path, cap=500).spend(120)

    assert Budget(path, cap=500).spent == 120


def test_a_new_month_resets_the_allowance(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"month": "1999-01", "characters": 8_000}))

    assert Budget(path, cap=9_000).spent == 0


def test_an_unreadable_budget_reads_as_unspent(tmp_path):
    """Overspending slightly beats going quiet because a file was locked."""
    path = tmp_path / "b.json"
    path.write_text("{ not json")

    assert Budget(path, cap=9_000).spent == 0


# --- every fallback ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("no key", {"key": ""}),
        ("budget spent", {"cap": 0}),
        ("unauthorized", {"transport": transport_returning(401, b'{"detail":"no"}')}),
        ("paid voice", {"transport": transport_returning(402, b'{"detail":"no"}')}),
        ("server error", {"transport": transport_returning(500, b"boom")}),
        ("rate limited", {"transport": transport_returning(429, b"slow down")}),
    ],
)
def test_every_failure_reaches_piper(tmp_path, audio, name, kwargs):
    voice, fallback = build(tmp_path, **kwargs)

    voice.speak("say something", threading.Event())

    assert fallback.lines == ["say something"], f"{name} did not fall back"
    assert voice.budget.spent == 0, f"{name} was charged for"


def test_a_timeout_reaches_piper(tmp_path, audio):
    import httpx

    voice, fallback = build(
        tmp_path, transport=transport_raising(httpx.ReadTimeout("slow"))
    )

    voice.speak("say something", threading.Event())

    assert fallback.lines == ["say something"]
    assert voice.last_reason == "timeout"


def test_a_dropped_connection_reaches_piper(tmp_path, audio):
    import httpx

    voice, fallback = build(
        tmp_path, transport=transport_raising(httpx.ConnectError("refused"))
    )

    voice.speak("say something", threading.Event())

    assert fallback.lines == ["say something"]
    assert voice.last_reason == "network"


def test_speaking_never_raises_whatever_happens(tmp_path, audio):
    """`Speaker._run` routes an exception to on_error and keeps going, but a
    voice that throws on every line would make every line an error event."""
    voice, fallback = build(tmp_path, transport=transport_raising(ValueError("odd")))

    voice.speak("hello", threading.Event())

    assert fallback.lines == ["hello"]


# --- the working path -------------------------------------------------------


def test_a_successful_line_plays_caches_and_is_charged_once(tmp_path, audio):
    voice, fallback = build(
        tmp_path, transport=transport_returning(200, chunks=[pcm(1024), pcm(1024)])
    )

    voice.speak("hello there", threading.Event())

    assert fallback.lines == [], "the cloud path fell back for no reason"
    assert audio.streams[0].samples.size == 2048
    assert voice.budget.spent == len("hello there")
    assert voice.cache.has("V1", "hello there")

    # And the second time is free.
    voice.speak("hello there", threading.Event())
    assert voice.budget.spent == len("hello there")


@pytest.mark.parametrize("split", [1, 3, 1023, 1025, 2047])
def test_a_chunk_boundary_inside_a_sample_does_not_corrupt_the_audio(split):
    """The failure this prevents is silent.

    One odd byte shifts every sample after it, and PCM plays as noise rather
    than raising anything. Driven through `align` directly, because a transport
    fake cannot control where httpx re-chunks the body, so routing this through
    the client would test the fake rather than the code.
    """
    original = np.arange(1, 2049, dtype="<i2")
    raw = original.tobytes()
    chunks = [raw[:split], raw[split:]]

    joined = np.concatenate(list(eleven.align(chunks)))

    assert np.array_equal(joined, original), f"a split at byte {split} corrupted it"


def test_align_holds_a_trailing_odd_byte_rather_than_guessing():
    """A stream that ends mid-sample must drop the half, not invent a sample."""
    assert list(eleven.align([b"\x01"])) == []
    out = list(eleven.align([b"\x01\x02\x03"]))
    assert len(out) == 1 and out[0].size == 1


def test_a_line_interrupted_before_it_starts_never_reaches_the_network(tmp_path, audio):
    """Barge-in has to be checked before the request, not only during playback.

    Without the guard at the top of speak(), an already-cancelled line still
    costs a round trip and a slice of the monthly budget. The inner loop would
    hide that: no audio plays either way, so only counting requests catches it.
    """
    requests = []

    import httpx

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, content=pcm(1024))

    stop = threading.Event()
    stop.set()
    voice, fallback = build(tmp_path, transport=httpx.MockTransport(handle))

    voice.speak("a very long sentence", stop)

    assert requests == [], "an interrupted line was still sent to ElevenLabs"
    assert fallback.lines == [], "an interrupted line was handed to piper instead"
    assert voice.budget.spent == 0, "an interrupted line was charged for"


def test_barge_in_partway_through_stops_the_audio(tmp_path, audio):
    """Cutting Vesper off mid-word is the most important interaction there is."""
    stop = threading.Event()
    voice, _ = build(tmp_path)
    voice.cache.put("V1", "a long line", pcm(64 * 1024))

    original = FakeOutputStream.write

    def write_then_cut(self, block):
        original(self, block)
        if len(self.written) >= 3:
            stop.set()

    FakeOutputStream.write = write_then_cut
    try:
        voice.speak("a long line", stop)
    finally:
        FakeOutputStream.write = original

    played = audio.streams[0].samples.size
    assert played < 64 * 1024, "playback ran to the end despite a barge-in"
    assert played <= 4 * 1024, f"stopped {played} samples in, more than a block late"


def test_an_interrupted_line_is_not_cached(tmp_path, audio):
    """Half a sentence cached would replay as half a sentence forever."""
    stop = threading.Event()
    stop.set()
    voice, _ = build(tmp_path, transport=transport_returning(200, chunks=[pcm(2048)]))

    voice.speak("interrupted", stop)

    assert not voice.cache.has("V1", "interrupted")


def test_the_device_is_aborted_on_interrupt(tmp_path, audio):
    """abort() rather than stop(), so buffered audio is dropped, not drained."""
    stop = threading.Event()
    voice, _ = build(tmp_path, transport=transport_returning(200, chunks=[pcm(8192)]))
    voice.cache.put("V1", "line", pcm(64 * 1024))

    def cut():
        stop.set()

    # Interrupt after the first block has been written.
    original = FakeOutputStream.write

    def write_then_cut(self, block):
        original(self, block)
        if len(self.written) >= 2:
            cut()

    FakeOutputStream.write = write_then_cut
    try:
        voice.speak("line", stop)
    finally:
        FakeOutputStream.write = original

    assert audio.streams[0].aborted, "the device kept its buffer after a barge-in"


# --- switching voice --------------------------------------------------------


def test_changing_voice_takes_effect_on_the_next_line(tmp_path, audio):
    voice, _ = build(tmp_path, transport=transport_returning(200, chunks=[pcm(512)]))
    voice.use(eleven_api.DEFAULT_VOICE_ID)

    assert voice.voice_id == eleven_api.DEFAULT_VOICE_ID
    assert "George" in voice.name

    voice.speak("after the switch", threading.Event())
    assert voice.cache.has(eleven_api.DEFAULT_VOICE_ID, "after the switch")
    assert not voice.cache.has("V1", "after the switch"), "cached under the old voice"


def test_an_empty_voice_id_is_ignored(tmp_path):
    voice, _ = build(tmp_path)
    voice.use("")
    assert voice.voice_id == "V1"


# --- prewarming -------------------------------------------------------------


def test_prewarm_caches_the_stock_phrases(tmp_path, audio):
    voice, _ = build(tmp_path, transport=transport_returning(200, chunks=[pcm(256)]))

    done = voice.prewarm(("One moment.", "Yes?"))

    assert done == 2
    assert voice.cache.has("V1", "One moment.")
    assert voice.budget.spent == len("One moment.") + len("Yes?")


def test_prewarm_skips_what_is_already_cached(tmp_path, audio):
    voice, _ = build(tmp_path, transport=transport_returning(200, chunks=[pcm(256)]))
    voice.cache.put("V1", "One moment.", pcm(256))

    assert voice.prewarm(("One moment.",)) == 0
    assert voice.budget.spent == 0


def test_prewarm_stops_rather_than_draining_the_budget(tmp_path, audio):
    voice, _ = build(
        tmp_path, transport=transport_returning(200, chunks=[pcm(256)]), cap=12
    )

    voice.prewarm(("One moment.", "Checking now.", "Hang on."))

    assert voice.budget.spent <= 12


def test_prewarm_does_nothing_without_a_key(tmp_path):
    voice, _ = build(tmp_path, key="")
    assert voice.prewarm() == 0


def test_the_stock_phrases_are_what_the_assistant_actually_says():
    """A phrase that drifts out of persona.py silently stops being cached."""
    from vesper.brain.persona import STILL_WORKING_FILLERS, THINKING_FILLERS

    for phrase in THINKING_FILLERS + STILL_WORKING_FILLERS:
        assert phrase in STOCK_PHRASES, (
            f"{phrase!r} is said on every tool call but is not prewarmed"
        )


# --- the client itself ------------------------------------------------------


def test_a_missing_key_is_reported_rather_than_sent(tmp_path):
    client = ElevenClient("")
    with pytest.raises(ElevenError) as caught:
        list(client.stream_pcm("hello", "V1"))
    assert caught.value.reason == "no-key"


def test_listing_voices_falls_back_to_the_verified_list():
    """A free key is refused voices_read, so this is the normal case."""
    client = ElevenClient("sk_test", transport=transport_returning(401, b"nope"))
    assert client.list_voices() == eleven_api.KNOWN_VOICES


def test_the_known_voices_record_which_ones_a_free_key_cannot_use():
    paid = {v.name for v in eleven_api.KNOWN_VOICES if not v.free}
    assert paid == {"Aria", "Charlotte"}, (
        "measured against a real free key on 2026-08-31; update the measurement, "
        "not the expectation"
    )
    assert all(v.voice_id for v in eleven_api.KNOWN_VOICES)
    assert len({v.voice_id for v in eleven_api.KNOWN_VOICES}) == len(
        eleven_api.KNOWN_VOICES
    )


def test_the_default_voice_is_one_a_free_key_can_actually_speak_with():
    chosen = [v for v in eleven_api.KNOWN_VOICES
              if v.voice_id == eleven_api.DEFAULT_VOICE_ID]
    assert chosen and chosen[0].free


def test_designing_a_voice_reports_the_paid_plan_wall_plainly(tmp_path):
    """403 on free, whatever the key's permissions say. Not dressed up."""
    client = ElevenClient("sk_test", transport=transport_returning(403, b"nope"))
    with pytest.raises(ElevenError) as caught:
        client.design("a calm British butler", "x" * 120)
    assert caught.value.reason == "paid-feature"


def test_designing_rejects_a_short_sample_before_spending_a_request(tmp_path):
    client = ElevenClient("sk_test", transport=transport_returning(200, b"{}"))
    with pytest.raises(ElevenError) as caught:
        client.design("a calm British butler", "too short")
    assert caught.value.reason == "short-text"


# --- the remembered choice --------------------------------------------------


def test_the_picked_voice_survives_a_restart(tmp_path):
    from vesper import voicechoice

    path = tmp_path / "choice.json"
    assert voicechoice.save(path, "ABC123", "George")

    loaded = voicechoice.load(path)
    assert loaded.voice_id == "ABC123"
    assert loaded.name == "George"
    assert loaded.chosen


def test_a_missing_or_broken_choice_is_no_choice(tmp_path):
    from vesper import voicechoice

    assert not voicechoice.load(tmp_path / "nope.json").chosen
    broken = tmp_path / "broken.json"
    broken.write_text("[]")
    assert not voicechoice.load(broken).chosen


def test_clearing_the_choice_hands_control_back_to_config(tmp_path):
    from vesper import voicechoice

    path = tmp_path / "choice.json"
    voicechoice.save(path, "ABC123", "George")
    voicechoice.clear(path)
    assert not voicechoice.load(path).chosen


# --- one connection, not one per sentence -----------------------------------


def test_one_http_client_serves_every_sentence():
    """`stream_pcm` is called once per sentence, not once per answer.

    Vesper starts speaking before the turn has finished, so a three sentence
    answer used to mean three DNS lookups, three TCP handshakes and three TLS
    handshakes to the same host, in front of somebody waiting to hear the
    first word.
    """
    import httpx

    built = []
    real = httpx.Client

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"\x00\x01" * 200)
    )
    client = ElevenClient("sk-test-key", transport=transport)
    httpx.Client = counting
    try:
        for sentence in ("One.", "Two.", "Three."):
            list(client.stream_pcm(sentence, "voice-id"))
    finally:
        httpx.Client = real

    assert len(built) == 1, f"opened {len(built)} clients for three sentences"
    client.close()


def test_closing_twice_is_not_an_error():
    import httpx

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b""))
    client = ElevenClient("sk-test-key", transport=transport)
    client._client()
    client.close()
    client.close()  # must not raise
    assert client._shared_client is None


def test_a_closed_client_can_still_be_used_again():
    """`close` is called at shutdown, but nothing promises no later call."""
    import httpx

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"\x00\x01" * 100)
    )
    client = ElevenClient("sk-test-key", transport=transport)
    list(client.stream_pcm("One.", "voice-id"))
    client.close()
    assert list(client.stream_pcm("Two.", "voice-id")), "it did not come back"
    client.close()
