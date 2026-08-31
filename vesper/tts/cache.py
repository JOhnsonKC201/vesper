"""Synthesized audio kept on disk, because Vesper repeats himself constantly.

Seventeen lines are hardcoded into the assistant: the six thinking fillers and
three still-working fillers in `brain/persona.py`, plus "Yes?", "Doing it.",
"Leaving it.", "Quiet from now on.", "Back on.", "Shutting down.", the greeting
and the undo line. Roughly 350 characters in total. Before this they were
re-synthesized every single time, which was free with a local model and is not
free with a metered one.

Caching them changes two things, and the second matters more than the money.

The money: on a 10,000 character monthly budget, the stock phrases would have
been a meaningful slice of it. Now they cost 350 characters once, ever.

The latency: a cached phrase plays from disk with no network round trip. So
when Claude starts a tool call and Vesper says "One moment.", that lands as
fast as it did with Piper, and the real sentence synthesizes behind it. Cloud
speech would otherwise have made the most latency-sensitive line in the whole
system the slowest one.

Raw headerless int16 PCM, not wav and not mp3. The rate is fixed by the request
(`pcm_22050`) so there is nothing to record, and a container would mean a
parser. `np.frombuffer` is the whole read path.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

# Two bytes per sample at 22050Hz is about 44KB a second, so this holds roughly
# four minutes of speech. The fixed phrases are a few seconds of that; the rest
# is whatever you happened to say twice.
DEFAULT_MAX_BYTES = 12 * 1024 * 1024

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def key_for(text: str) -> str:
    """A stable filename for a line of speech.

    Normalised on whitespace and case so "One moment." and "one  moment."
    share an entry. Not on punctuation, because punctuation changes delivery.
    """
    normal = " ".join((text or "").split()).casefold()
    return hashlib.sha1(normal.encode("utf-8")).hexdigest()[:20]


class AudioCache:
    """PCM keyed by voice and text. Never raises; a broken cache is a miss."""

    def __init__(self, root: Path | str | None, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root) if root else None
        self.max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _dir(self, voice_id: str) -> Path | None:
        if self.root is None:
            return None
        # A voice id comes from config or the API, so it is not trusted to be a
        # safe path component.
        return self.root / (_UNSAFE.sub("_", voice_id or "default")[:64])

    def _path(self, voice_id: str, text: str) -> Path | None:
        folder = self._dir(voice_id)
        return None if folder is None else folder / f"{key_for(text)}.pcm"

    # --- reading ------------------------------------------------------------

    def get(self, voice_id: str, text: str) -> bytes | None:
        path = self._path(voice_id, text)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except (OSError, ValueError):
            self.misses += 1
            return None
        if not data:
            self.misses += 1
            return None
        self.hits += 1
        # Touch, so eviction can be least-recently-used rather than oldest.
        # Failing to touch is not worth losing a cache hit over.
        try:
            path.touch()
        except OSError:
            pass
        return data

    def has(self, voice_id: str, text: str) -> bool:
        path = self._path(voice_id, text)
        try:
            return path is not None and path.exists() and path.stat().st_size > 0
        except OSError:
            return False

    # --- writing ------------------------------------------------------------

    def put(self, voice_id: str, text: str, pcm: bytes) -> None:
        path = self._path(voice_id, text)
        if path is None or not pcm:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write beside and rename, so a crash mid-write cannot leave a
            # truncated file that later plays as a burst of noise.
            temporary = path.with_suffix(".part")
            temporary.write_bytes(pcm)
            temporary.replace(path)
            self._note(voice_id, text)
        except (OSError, ValueError):
            return
        self._evict()

    def _note(self, voice_id: str, text: str) -> None:
        """Keep a readable index, so the cache directory can be inspected."""
        folder = self._dir(voice_id)
        if folder is None:
            return
        manifest = folder / "manifest.json"
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, ValueError):
            existing = {}
        existing[key_for(text)] = {"text": text[:200], "at": int(time.time())}
        try:
            manifest.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass

    # --- housekeeping -------------------------------------------------------

    def size(self) -> int:
        if self.root is None or not self.root.exists():
            return 0
        total = 0
        try:
            for path in self.root.rglob("*.pcm"):
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return 0
        return total

    def _evict(self) -> None:
        """Drop the least recently used entries until back under the cap."""
        if self.root is None:
            return
        try:
            files = [(p.stat().st_mtime, p.stat().st_size, p)
                     for p in self.root.rglob("*.pcm")]
        except OSError:
            return
        total = sum(size for _, size, _ in files)
        if total <= self.max_bytes:
            return
        for _, size, path in sorted(files):
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue

    def clear(self) -> None:
        if self.root is None:
            return
        try:
            for path in self.root.rglob("*"):
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        continue
        except OSError:
            pass
