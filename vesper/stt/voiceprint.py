"""Telling your voice from everyone else's.

An assistant that runs from login hears everything: a video, a meeting on
speakers, someone else in the room saying the wrong word. Without this, any of
them can wake it, and one of them could answer a permission question.

**This never runs at idle.** It is only reached for an utterance that already
contains the wake word, a handful of times an hour. That is what keeps an
always-on assistant inside a CPU budget: the expensive thing happens after
something has happened, not thirty times a second.

## Why a real model, after trying not to use one

The first version of this file used the mean and standard deviation of MFCCs
compared by cosine similarity. No download, no new dependency, half a
millisecond per utterance. It did not work, and the measurement is worth keeping
because it is the reason this file looks the way it does.

Enrolled on one voice and tested against two others, all real synthesized
speech:

    owner        0.989
    other male   0.887 to 0.917
    other female 0.896 to 0.931   <- scored higher than the other male

Separation of 0.057, with a female voice indistinguishable from a male one. The
cause is structural rather than a tuning problem: every speech MFCC-statistic
vector points in roughly the same direction, so cosine similarity between any
two of them is high by construction. Dropping the energy coefficient, z-scoring,
correlation instead of cosine, and adding pitch statistics were all tried. The
best of them scored +0.057 and adding pitch made it actively worse, because
appending dimensions to a unit vector drives every comparison toward 1.0.

WeSpeaker ResNet34, trained on VoxCeleb, on the same three voices:

    owner        0.856 to 0.869
    strangers   -0.010 to 0.094

Separation of 0.761. That is a working feature rather than a decorative one.

## What it costs

57ms per utterance and 26MB on disk, once. It runs on onnxruntime, which is
already installed as part of faster-whisper, and the features come from
torchaudio, which is already installed for the VAD. So this adds a model file
and **no new Python packages**, which matters on a machine that has had wheels
blocked by Windows Application Control before.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
EMBEDDING_DIM = 256

# Below this an utterance is too short to characterise a voice. Shorter clips
# are still answered rather than rejected: "Vesper" on its own is under a
# second, and refusing to wake for your own wake word would be absurd.
MIN_SECONDS = 0.6

# And below this, a mismatch is never trusted enough to act on. A short clip
# produces an unstable embedding, and the failure is not theoretical: a real
# "Sure." scored 0.22 against its own owner's profile and was thrown away.
# "Sure" is how you approve things, so that reading turned a safety feature into
# a way of ignoring consent. Anything shorter than this is answered, whatever it
# scores.
MIN_REJECT_SECONDS = 1.6

# Cosine thresholds. Measured above: the owner sits near 0.86 in clean
# conditions and strangers below 0.10. These sit far from both, because real
# use is messier than a test: a different mic distance, a cold, a noisy room.
#
# The wide middle band is the design. Only the clearly-different region rejects;
# everything else is heard normally and written to the log. That mirrors the
# call already made for the wake word, where "whisper" and "jasper" were removed
# because being occasionally deaf is worse than occasionally waking in error.
CONFIDENT_MATCH = 0.55
CLEARLY_DIFFERENT = 0.35

MATCH, UNSURE, DIFFERENT = "match", "unsure", "different"

_SESSION = None
_SESSION_LOCK = threading.Lock()
_SESSION_FAILED = False


@dataclass
class Profile:
    """An enrolled voice: the mean embedding, and how much it varied."""

    embedding: tuple[float, ...] = ()
    samples: int = 0
    spread: float = 0.0
    created: str = ""
    model: str = ""

    @property
    def enrolled(self) -> bool:
        # The dimension check retires profiles made by the MFCC version rather
        # than comparing against them, which would silently score zero.
        return (
            bool(self.embedding)
            and self.samples > 0
            and len(self.embedding) == EMBEDDING_DIM
        )

    def to_dict(self) -> dict:
        return {
            "embedding": list(self.embedding),
            "samples": self.samples,
            "spread": self.spread,
            "created": self.created,
            "model": self.model,
        }

    @staticmethod
    def from_dict(data: dict) -> "Profile":
        raw = data.get("embedding") or []
        return Profile(
            embedding=tuple(float(x) for x in raw),
            samples=int(data.get("samples") or 0),
            spread=float(data.get("spread") or 0.0),
            created=str(data.get("created") or ""),
            model=str(data.get("model") or ""),
        )


# --- the model --------------------------------------------------------------


def model_path(download: bool = False) -> Path | None:
    """Where the model is, or None if it is not here.

    Downloading only when asked, so a missing model at conversation time is a
    shrug rather than a network call in the middle of someone talking.
    """
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            MODEL_REPO, MODEL_FILE, local_files_only=not download
        ))
    except Exception:
        return None


def _session():
    """The ONNX session, built once. None if the model is unavailable."""
    global _SESSION, _SESSION_FAILED
    if _SESSION is not None or _SESSION_FAILED:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None or _SESSION_FAILED:
            return _SESSION
        try:
            import onnxruntime as ort

            path = model_path()
            if path is None:
                _SESSION_FAILED = True
                return None
            options = ort.SessionOptions()
            # Two threads. This runs while somebody is waiting for an answer,
            # but it is 57ms of work and taking every core for it would be the
            # kind of thing that makes an always-on assistant felt.
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            _SESSION = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
        except Exception:
            _SESSION_FAILED = True
    return _SESSION


def available() -> bool:
    return _session() is not None


def embed(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """One 256-dimension vector describing how a voice sounds.

    Returns None whenever it cannot form an opinion, which callers must treat
    as "no opinion" rather than as a rejection.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < int(MIN_SECONDS * sample_rate):
        return None

    session = _session()
    if session is None:
        return None

    try:
        import torch
        import torchaudio

        peak = float(np.abs(audio).max())
        if peak < 1e-6:
            return None

        # Kaldi-style fbank at the scale WeSpeaker was trained on, then mean
        # normalised over time, which removes the microphone and the room and
        # leaves the speaker.
        wav = torch.from_numpy(audio / peak).unsqueeze(0) * 32768.0
        feats = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        feats = feats - feats.mean(dim=0, keepdim=True)

        vector = session.run(None, {"feats": feats.unsqueeze(0).numpy()})[0][0]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return None
        return (vector / norm).astype(np.float32)
    except Exception:
        # A voiceprint failure must never stop Vesper answering. No opinion.
        return None


def warm(sample_rate: int = SAMPLE_RATE) -> None:
    """Pay the model load at startup rather than on the first thing you say."""
    try:
        embed(np.zeros(int(sample_rate * (MIN_SECONDS + 0.2)), dtype=np.float32) + 1e-3,
              sample_rate)
    except Exception:
        pass


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two unit vectors."""
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


# --- the profile on disk ----------------------------------------------------


class VoicePrint:
    """Enrolment, storage and comparison. Never raises at the call site."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._profile: Profile | None = None

    @property
    def enrolled(self) -> bool:
        return self.profile.enrolled

    @property
    def profile(self) -> Profile:
        if self._profile is None:
            self._profile = self._load()
        return self._profile

    def _load(self) -> Profile:
        if self.path is None or not self.path.exists():
            return Profile()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return Profile()
            return Profile.from_dict(raw)
        except (OSError, ValueError, TypeError, AttributeError):
            # A corrupt profile means no opinion, never a locked-out assistant.
            return Profile()

    def enrol(self, clips: list[np.ndarray], sample_rate: int = SAMPLE_RATE) -> Profile:
        """Learn a voice from several clips. Returns the stored profile."""
        from datetime import datetime

        vectors = [v for v in (embed(c, sample_rate) for c in clips) if v is not None]
        if not vectors:
            return Profile()

        mean = np.mean(np.vstack(vectors), axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-9)
        # How much your own samples agreed with each other. Read back by
        # --enroll: if your own clips only score 0.5 against their own average,
        # nothing downstream can separate you from anybody else.
        spread = float(np.mean([similarity(mean, v) for v in vectors]))

        profile = Profile(
            embedding=tuple(float(x) for x in mean),
            samples=len(vectors),
            spread=spread,
            created=datetime.now().isoformat(timespec="seconds"),
            model=MODEL_REPO,
        )
        self._profile = profile
        self._save(profile)
        return profile

    def _save(self, profile: Profile) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, ValueError):
            pass

    def compare(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[str, float]:
        """Is this the enrolled speaker? Returns a verdict and a score.

        Answers UNSURE whenever it cannot tell: nothing enrolled, a clip too
        short, a missing model, any failure at all. Only a confident mismatch
        returns DIFFERENT, because only DIFFERENT causes anything to be ignored.
        """
        profile = self.profile
        if not profile.enrolled:
            return UNSURE, 0.0

        vector = embed(audio, sample_rate)
        if vector is None:
            return UNSURE, 0.0

        stored = np.asarray(profile.embedding, dtype=np.float32)
        score = similarity(vector, stored)
        if score >= CONFIDENT_MATCH:
            return MATCH, score
        if score <= CLEARLY_DIFFERENT:
            # A confident-looking mismatch on a short clip is not confident, it
            # is just short. Only long enough utterances may be rejected.
            long_enough = audio.size >= int(MIN_REJECT_SECONDS * sample_rate)
            return (DIFFERENT if long_enough else UNSURE), score
        return UNSURE, score
