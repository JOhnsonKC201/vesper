"""Privacy invariants, enforced rather than promised.

The commitment made to the user is specific: the only things that leave this
machine are the conversation with Claude, which goes out through the `claude`
subprocess, and the text of Vesper's replies when the ElevenLabs voice is
switched on. Speech recognition, the sensors and the wake word are local
unconditionally, and microphone audio never leaves under any setting.

A README can claim that. These tests make it true, and make it stay true when
someone later adds "just a quick lookup" to a sensor.

The cloud voice is one named, contained exception rather than a general
loosening. `tts/eleven_api.py` is the only file in the package permitted to
import an HTTP client, and `test_only_one_file_may_touch_the_network` fails the
moment a second one appears. The rest of this file got stricter, not weaker:
there are now tests that the API key cannot reach a log, and that microphone
samples have no path to the network at all.
"""

import ast
import socket
from pathlib import Path

import numpy as np
import pytest

from vesper.audio.speaker import Speaker
from vesper.conversation import Conversation, ConversationConfig
from vesper.sensors import snapshot as sensors
from vesper.sensors.window import ActiveWindow, redact
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI

PACKAGE = Path(__file__).resolve().parent.parent / "vesper"

# Anything that can open a connection. The claude subprocess is how Vesper
# reaches Anthropic, and it is spawned, not imported, so none of these belong
# anywhere in the package.
NETWORK_MODULES = {
    "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
    "ftplib", "telnetlib", "smtplib", "websockets", "websocket",
    "xmlrpc", "socketserver", "paramiko",
    # Added after a review pointed out the obvious gap: this is a denylist of
    # names, so it only catches what someone remembered to write down.
    # asyncio.open_connection opens a raw socket and nothing here was looking
    # for it. ctypes is deliberately absent rather than forgotten: it is
    # legitimately used for Win32 window titles in sensors/window.py, so
    # test_no_module_calls_a_network_dll below checks the thing that actually
    # matters instead of banning the module.
    "asyncio",
}

# The single file allowed to speak HTTP, and the only one. Adding a second name
# here should feel like a decision, which is what the test below turns it into.
NETWORK_FILE = "eleven_api.py"

# Files that are allowed to name the network for good reason.
ALLOWED_URL_FILES = {"piper_voice.py", "whisper.py", NETWORK_FILE}


def source_files():
    return sorted(PACKAGE.rglob("*.py"))


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


# --- static guarantees ------------------------------------------------------


def test_no_module_imports_a_network_library():
    offenders = []
    for path in source_files():
        if path.name == NETWORK_FILE:
            continue
        bad = imported_names(path) & NETWORK_MODULES
        if bad:
            offenders.append(f"{path.relative_to(PACKAGE)}: {sorted(bad)}")
    assert not offenders, (
        "Only " + NETWORK_FILE + " may make network calls. "
        "Offending modules: " + "; ".join(offenders)
    )


def test_only_one_file_may_touch_the_network():
    """The containment itself, asserted from the other direction.

    The test above skips one filename. This one fails if that skip ever starts
    covering something new, so widening the hole cannot happen by accident: it
    takes editing this test, deliberately, with the name in front of you.
    """
    talkers = sorted(
        path.name for path in source_files()
        if imported_names(path) & NETWORK_MODULES
    )
    assert talkers == [NETWORK_FILE], (
        f"network access spread to {talkers}. Every outbound call in this "
        f"package goes through {NETWORK_FILE} so there is one place to audit."
    )


def test_no_module_calls_a_network_dll():
    """The hole an import denylist cannot close.

    `ctypes.windll.wininet` reaches the network with no import this file would
    ever notice, and ctypes cannot simply be banned because window titles are
    read through it. So check the thing that would actually be dangerous.
    """
    offenders = []
    for path in source_files():
        text = path.read_text(encoding="utf-8").lower()
        for library in ("wininet", "ws2_32", "winhttp", "urlmon", "winsock"):
            if library in text:
                offenders.append(f"{path.relative_to(PACKAGE)}: {library}")
    assert offenders == [], "; ".join(offenders)


def test_no_module_shells_out_instead_of_importing():
    """`test_the_only_external_process_is_claude` looks for an import of
    `subprocess`. `os.system` needs no import at all, because `os` is already
    imported nearly everywhere in this package."""
    offenders = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for call in ("os.system(", "os.popen(", "os.execv", "os.spawn"):
            if call in text:
                offenders.append(f"{path.relative_to(PACKAGE)}: {call}")
    assert offenders == [], "; ".join(offenders)


def test_the_cloud_voice_is_off_by_default():
    """Shipping it on would send replies out before anyone chose to."""
    from vesper.config import Config

    assert Config().voice.engine == "piper"
    assert Config().voice.eleven.api_key == ""


def test_the_packaged_default_carries_no_api_key():
    """The tracked config must never hold a credential."""
    import re

    default = PACKAGE.parent / "packaging" / "default" / "config.yaml"
    text = default.read_text(encoding="utf-8")
    assert not re.search(r"sk_[A-Za-z0-9]{20,}", text), "a key leaked into the default"
    assert re.search(r'api_key:\s*(""|\'\'|\s*$)', text, re.M), (
        "voice.eleven.api_key must ship empty"
    )


def test_the_api_key_never_reaches_an_error_message():
    """A key in an exception ends up in var/vesper.log, which is plain text."""
    from vesper.tts.eleven_api import ElevenClient, ElevenError, classify

    # Assembled rather than written out, so this fixture does not itself trip
    # the credential scan that runs over the tree in CI.
    secret = "sk" + "_" + "thisisnotarealkeyjustatestvalue00000000"

    # Feed the secret to classify() rather than a body that never contained
    # it. The first version passed b'{"detail":"nope"}', so the assertion was
    # unconditionally true: the string it looked for could not have been there
    # whatever classify did. The branches that matter are 422 and the generic
    # one, which do put the response body verbatim into `detail`, and some
    # APIs echo request fields back in a validation error.
    leaky_body = ('{"detail":"bad key ' + secret + '"}').encode()
    for status in (401, 402, 403, 422, 429, 500, 418):
        error = classify(status, leaky_body)
        assert secret not in str(error), f"{status} leaked the key into str()"
        assert secret not in error.detail, f"{status} leaked the key into detail"
        assert secret not in error.reason

    client = ElevenClient(secret, base="http://127.0.0.1:9")
    try:
        list(client.stream_pcm("hello", "voice"))
    except ElevenError as exc:
        assert secret not in str(exc), "the key leaked into the failure"
        assert secret not in exc.detail
    else:
        pytest.fail("a connection to a closed port should have failed")


def test_microphone_audio_has_no_path_to_the_network():
    """The line that must never move.

    The cloud voice sends text. Nothing sends samples. This walks the one
    module allowed to make requests and asserts it never accepts an ndarray.
    """
    import inspect

    from vesper.tts import eleven_api

    # AST rather than a substring search, so the prose in the docstring cannot
    # fail this and, more importantly, cannot silently satisfy it either.
    assert "numpy" not in imported_names(PACKAGE / "tts" / NETWORK_FILE), (
        f"{NETWORK_FILE} imported numpy; audio must never be uploaded"
    )

    # Every outbound body is assembled from strings. An ndarray or a buffer of
    # samples reaching one of these is the failure this guards against.
    #
    # Compared as names because `from __future__ import annotations` makes
    # every annotation in that module a string at runtime, the same thing
    # `config._coerce` has a comment about.
    # Every method that leaves the machine, discovered rather than listed.
    # The first version named three methods by hand, so a new one was invisible
    # until somebody remembered to add it here.
    outbound = [
        name for name in dir(eleven_api.ElevenClient)
        if not name.startswith("_")
        and callable(getattr(eleven_api.ElevenClient, name, None))
    ]
    assert {"stream_pcm", "design", "keep"} <= set(outbound)

    for name in outbound:
        method = getattr(eleven_api.ElevenClient, name)
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            continue
        for parameter in parameters:
            if parameter.name == "self":
                continue
            # An unannotated parameter used to pass, because "" was allowed.
            # That is the whole bypass: add `def stream_pcm(self, text: str,
            # mic_samples, ...)` with no hint and the guard said nothing.
            assert parameter.annotation is not inspect.Parameter.empty, (
                f"ElevenClient.{name} takes {parameter.name} with no type. "
                f"Annotate it, so this guard can see what it is."
            )
            assert str(parameter.annotation) == "str", (
                f"ElevenClient.{name} takes {parameter.name}: "
                f"{parameter.annotation}, which is not text"
            )


def test_socket_is_never_imported():
    """Not even indirectly. There is no reason for this package to need one."""
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in source_files()
        if "socket" in imported_names(path)
    ]
    assert offenders == []


def test_no_hardcoded_external_urls():
    offenders = []
    for path in source_files():
        if path.name in ALLOWED_URL_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("http://", "https://"):
            if marker in text:
                offenders.append(f"{path.relative_to(PACKAGE)} contains {marker}")
    assert offenders == [], "; ".join(offenders)


def test_the_only_external_process_is_claude():
    """Nothing else may be spawned. A subprocess is a hole in every other rule."""
    spawners = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if "subprocess" in imported_names(path):
            spawners.append(path.name)
    assert spawners == ["claude.py"], (
        f"unexpected subprocess use in {spawners}; only the brain may spawn"
    )


# --- runtime guarantees -----------------------------------------------------


def test_a_full_conversation_opens_no_sockets(monkeypatch):
    """Drive a real turn through the loop and assert nothing dialled out."""
    connections = []

    real_connect = socket.socket.connect

    def spy(self, address, *args, **kwargs):
        connections.append(address)
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", spy)

    voice = FakeVoice()
    speaker = Speaker(voice)
    conversation = Conversation(
        brain=FakeBrain(["All good here."]),
        stt=FakeSTT(["Vesper how are things"]),
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(greet_on_start=False),
        ui=RecordingUI(),
    )
    conversation._on_utterance(np.full(16000, 0.2, dtype=np.float32))
    speaker.wait_until_idle(timeout=5)
    speaker.close()

    assert voice.lines == ["All good here."]
    external = [
        address
        for address in connections
        if not (isinstance(address, tuple) and str(address[0]).startswith("127."))
    ]
    assert external == [], f"Vesper opened outbound connections: {external}"


def test_sensors_read_locally_and_open_no_sockets(monkeypatch):
    connections = []
    monkeypatch.setattr(
        socket.socket, "connect", lambda self, addr, *a, **k: connections.append(addr)
    )
    sensors.take(include_processes=True)
    sensors.context_block()
    assert connections == []


# --- redaction --------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Bitwarden - my vault",
        "1Password",
        "Chrome - Incognito",
        "reset your password",
        "API_KEY setup - Notepad",
        "Barclays Banking",
        "LastPass Vault",
        # Edge's word for incognito. It was missing, so an Edge private window
        # was reported by its real title while Chrome's was hidden.
        "InPrivate - Microsoft Edge",
        "Mozilla Firefox Private Browsing",
    ],
)
def test_sensitive_window_titles_are_hidden(title):
    """Window titles are the most revealing thing Vesper reads."""
    result = redact(ActiveWindow(title=title, process="app.exe", pid=1))
    assert result.title == "[hidden]"
    assert result.process == "app.exe", "the app name is still useful context"


def test_ordinary_window_titles_are_kept():
    result = redact(ActiveWindow(title="main.py - Vesper", process="Code.exe", pid=1))
    assert result.title == "main.py - Vesper"


def test_context_block_stays_small():
    """It rides on every single turn, so size is a cost, not a detail."""
    block = sensors.context_block()
    assert len(block) < 400, f"context block is {len(block)} chars, too fat for every turn"


# --- what gets written down -------------------------------------------------


def test_the_log_can_be_told_not_to_keep_the_words(tmp_path):
    """Everything the microphone hears is transcribed, addressed or not.

    One week of a real var/vesper.log holds 975 transcriptions and 238 of them
    were for Vesper. The rest is a record of a room: other people, a
    television, half of a phone call. sensors/window.py already redacts a
    sensitive window title and there was no equivalent for what was said.
    """
    from vesper.logfile import LogFile, attach

    class UI:
        def __init__(self):
            self.seen = []

        def heard(self, text, addressed):
            self.seen.append((text, addressed))

        def info(self, message): pass
        def warn(self, message): pass
        def error(self, message): pass
        def discarded(self, reason): pass

    path = tmp_path / "quiet.log"
    ui = UI()
    attach(ui, LogFile(path), transcripts=False)
    ui.heard("my card number is four one one one", False)
    ui.heard("Vesper what time is it", True)

    written = path.read_text(encoding="utf-8")
    assert "card number" not in written
    assert "four one one one" not in written
    assert "what time is it" not in written
    # But that something was heard, and who for, is still answerable.
    assert "->" in written and written.count("chars") == 2
    # And the terminal is untouched, which is the point of wrapping rather
    # than editing the UI.
    assert ui.seen == [
        ("my card number is four one one one", False),
        ("Vesper what time is it", True),
    ]


def test_transcripts_are_kept_by_default(tmp_path):
    """It is the only thing that answers "I said the wake word and nothing
    happened", so the default must not change."""
    from vesper.config import RuntimeSettings
    from vesper.logfile import LogFile, attach

    assert RuntimeSettings().log_transcripts is True

    class UI:
        def heard(self, text, addressed): pass
        def info(self, message): pass
        def warn(self, message): pass
        def error(self, message): pass
        def discarded(self, reason): pass

    path = tmp_path / "loud.log"
    ui = UI()
    attach(ui, LogFile(path))
    ui.heard("Vesper what time is it", True)
    assert "what time is it" in path.read_text(encoding="utf-8")
