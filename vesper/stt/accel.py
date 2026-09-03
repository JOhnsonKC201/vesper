"""Reaching the GPU, and proving it works before trusting it with your voice.

Three separate things stood between Vesper and the graphics card, and only the
first one is obvious.

**faster-whisper does not use torch.** It runs on CTranslate2, which is its own
runtime with its own CUDA build. Asking `torch.cuda.is_available()` whether
Whisper can use the GPU is asking the wrong library entirely. torch here is
deliberately the CPU build, installed for Silero VAD and the voiceprint model,
so that question answered "no" on every machine forever, whatever hardware was
in it. CTranslate2 is the one that has to be asked.

**The CUDA libraries are not on the search path.** They ship as pip packages
that drop their DLLs in `site-packages/nvidia/*/bin`, a directory nobody's PATH
has ever heard of. `os.add_dll_directory` does not fix it: CTranslate2 loads
them from native code with a plain LoadLibrary, which walks the ordinary search
order and never sees directories added for Python's own extension loading. They
have to be on PATH, and on PATH before the model is built.

**A device being present is not the same as it working.** CTranslate2 reports
one CUDA device on a machine with no CUDA runtime installed at all. The model
then builds without complaint and every transcription afterwards raises
`Library cublas64_12.dll is not found`. Success at load with failure at
inference is the worst shape this could take, because it turns "the GPU is not
set up" into "the assistant dies the first time you speak to it", at login, with
no console for the traceback to land in. So the check below is a real
inference, run once, and it is the only answer trusted.
"""

from __future__ import annotations

import os
import site
import sys
import threading
from pathlib import Path

import numpy as np

# Long enough for the encoder to do real work, short enough that paying for it
# at startup is not felt. This is also the warm-up: the first inference of a
# process compiles kernels, and it is better spent here than on your first
# sentence.
_PROBE_SECONDS = 1.0
_PROBE_RATE = 16_000
# Generous: the first inference in a process compiles kernels, and on a cold
# kernel cache that was measured at 26s once before settling to 0.26s. Long
# enough never to fail a working card, short enough to be survivable.
_PROBE_TIMEOUT_S = 60.0

_search_paths_added: list[str] = []


def _site_dirs() -> list[Path]:
    """Everywhere a wheel could have put its DLLs, without guessing."""
    candidates = list(site.getsitepackages())
    user = site.getusersitepackages()
    if isinstance(user, str):
        candidates.append(user)
    else:  # some builds return a list
        candidates.extend(user)
    # The running interpreter's own prefix last, for a venv laid out unusually.
    candidates.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    seen: list[Path] = []
    for entry in candidates:
        path = Path(entry)
        if path.is_dir() and path not in seen:
            seen.append(path)
    return seen


def dll_directories() -> list[str]:
    """The `nvidia/*/bin` folders that actually contain DLLs, deduplicated."""
    found: list[str] = []
    for root in _site_dirs():
        for dll in sorted(root.glob("nvidia/*/bin/*.dll")):
            parent = str(dll.parent)
            if parent not in found:
                found.append(parent)
    return found


def enable_dll_search() -> list[str]:
    """Put the CUDA DLLs where a native LoadLibrary will find them.

    Idempotent, because it is called from the model load and the self check and
    a Listener may be built more than once in a process. Returns what was added
    the first time, so the caller can say what it did.
    """
    global _search_paths_added
    if _search_paths_added:
        return _search_paths_added

    directories = dll_directories()
    if not directories:
        return []

    # Prepended, not appended: a half installed CUDA toolkit elsewhere on PATH
    # with a mismatched cuBLAS is a worse answer than the one pinned right here
    # next to the runtime that was built against it.
    os.environ["PATH"] = os.pathsep.join(directories + [os.environ.get("PATH", "")])
    for directory in directories:
        try:
            os.add_dll_directory(directory)
        except (AttributeError, OSError):
            # Not Windows, or the directory vanished between the glob and here.
            # PATH alone is what CTranslate2 actually reads, so this is a bonus.
            pass
    _search_paths_added = directories
    return directories


def device_count() -> int:
    """How many CUDA devices CTranslate2 can see. Never raises."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def compute_types(device: str = "cuda") -> set[str]:
    """Which precisions this device supports. Empty if it cannot be asked."""
    try:
        import ctranslate2

        return set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return set()


def best_compute_type(device: str) -> str:
    """The precision to use when the config says `auto`.

    float16 halves the memory and is what every current card is fastest at.
    int8_float16 is the fallback for a card without full half precision, and
    int8 is what CPU has always used here.
    """
    if device != "cuda":
        return "int8"
    supported = compute_types("cuda")
    for candidate in ("float16", "int8_float16", "float32"):
        if candidate in supported:
            return candidate
    return "float32"


def probe(model, timeout_s: float = _PROBE_TIMEOUT_S) -> str:
    """Run one real inference. Returns "" if the GPU works, else why not.

    A loaded model proves nothing. This is the only check that does.

    Bounded in time, and on its own thread, because a broken CUDA install does
    not reliably raise. A half uninstalled one was observed to sit in `encode`
    instead of failing, and a startup that hangs is worse than the failure this
    whole function exists to catch: it is an assistant that never comes up at
    all, from a login shortcut, with nothing on screen to say why. The thread
    is a daemon, so one that never returns cannot hold the process open either.
    """
    silence = np.zeros(int(_PROBE_SECONDS * _PROBE_RATE), dtype=np.float32)
    outcome: list[str] = []

    def run() -> None:
        try:
            segments, _info = model.transcribe(silence, language="en", beam_size=1)
            # `transcribe` returns a generator and does no work until it is
            # drawn from. Consuming it here is the entire point of this.
            list(segments)
        except Exception as exc:
            outcome.append(f"{type(exc).__name__}: {exc}")
        else:
            outcome.append("")

    worker = threading.Thread(target=run, daemon=True, name="whisper-gpu-probe")
    worker.start()
    worker.join(timeout_s)
    if not outcome:
        return f"the gpu did not answer within {timeout_s:.0f}s"
    return outcome[0]


def missing_runtime_hint() -> str:
    """What to tell someone who has the card but not the libraries.

    Only worth saying when there is a device to be disappointed about.
    """
    if not device_count():
        return ""
    if dll_directories():
        return (
            "cuda libraries are installed but did not load, so transcription "
            "is on the cpu"
        )
    return (
        "gpu found but the cuda runtime is not installed: "
        "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
    )
