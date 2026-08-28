"""Privacy invariants, enforced rather than promised.

The commitment made to the user is specific: the only thing that leaves this
machine is the conversation with Claude, which goes out through the `claude`
subprocess. Speech recognition, speech synthesis, the sensors and the wake word
are all local, and no part of Vesper phones anywhere.

A README can claim that. These tests make it true, and make it stay true when
someone later adds "just a quick lookup" to a sensor.
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
}

# Files that are allowed to name the network for good reason.
ALLOWED_URL_FILES = {"piper_voice.py", "whisper.py"}


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
        bad = imported_names(path) & NETWORK_MODULES
        if bad:
            offenders.append(f"{path.relative_to(PACKAGE)}: {sorted(bad)}")
    assert not offenders, (
        "Vesper must not make network calls of its own. "
        "Offending modules: " + "; ".join(offenders)
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
