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

# Those two constants were wrong for the one user who mattered, and stayed wrong
# for eleven days in silence. `var/vesper.log` holds 279 scored utterances whose
# maximum was 0.54 on every single day, against a bar of 0.55: not one match,
# ever, while 266 utterances were thrown away. The model was never the problem.
# Enrolled on two real fixture clips it scores the third at 0.80 and a different
# speaker at 0.05, so the separation was always there to be used.
#
# Two things caused it. The bar was a guess that nobody checked against this
# user's own numbers, and a profile was a single averaged centroid, which sits
# between unlike clips and therefore near none of them. So the bar is now
# measured by `calibrate` and stored in the profile, and a clip is scored against
# every sample you gave. The constants above remain the fallback for a profile
# recorded before this change, which must keep behaving exactly as it did rather
# than silently acquiring new thresholds.
#
# How far below your own worst agreement the bar sits. Enrolment is a handful of
# clips in one sitting; live use is every mood, mic distance and head cold, so
# the live spread is wider than the enrolled one and the bar must allow for it.
CALIBRATION_MARGIN = 0.12
# Never demand more than this, however tightly your clips agreed. Two clips that
# score 0.99 against each other are usually the same sentence said the same way,
# which says nothing about tomorrow.
MATCH_CEILING = 0.62
# Never demand less than this, however badly they agreed. Below it the profile is
# not separating anybody from anybody, which `--voicecheck` says out loud rather
# than leaving to be inferred from being ignored.
MATCH_FLOOR = 0.35
# A rejection sits clear of the owner's own band, because the costs are not
# symmetric: answering a stranger once is a curiosity, and ignoring your owner is
# the whole feature failing. This is the gap kept below the match bar.
REJECT_GAP = 0.18
REJECT_FLOOR = 0.08

# How many clips a profile may hold. Enrolment contributes a handful and `adapt`
# adds confirmed utterances as they arrive, so without a cap the file grows for
# as long as Vesper is used.
MAX_VECTORS = 12

MATCH, UNSURE, DIFFERENT = "match", "unsure", "different"

_SESSION = None
_SESSION_LOCK = threading.Lock()
_SESSION_FAILED = False


@dataclass
class Profile:
    """An enrolled voice: every clip of it, their mean, and where the bar sits.

    `vectors` is the part that was missing. A mean alone is one point, and a
    voice is a region: scored against the mean, an utterance at the edge of your
    own range loses to the averaging, which is how a real owner came to sit at
    0.54 for eleven days. Keeping the clips costs 1KB each and scores every
    utterance against the nearest thing you actually said.
    """

    embedding: tuple[float, ...] = ()
    vectors: tuple[tuple[float, ...], ...] = ()
    # How many leading entries of `vectors` came from `--enroll` rather than from
    # `adapt`. They are the anchors: never evicted, and the only clips the
    # thresholds are ever measured from.
    #
    # Without that separation, learning as it goes compounds a mistake instead of
    # correcting one. An utterance that cleared the bar joins the profile, the
    # bar is recalculated from the wider set, and an outlier drags the worst
    # leave-one-out score DOWN, which lowers the bar for everyone who comes next.
    # One lucky accept would quietly make the next one easier.
    anchors: int = 0
    samples: int = 0
    spread: float = 0.0
    created: str = ""
    model: str = ""
    # Zero means "recorded before calibration existed", which is not the same as
    # zero difficulty. Both properties below read them that way.
    match_at: float = 0.0
    reject_at: float = 0.0

    @property
    def enrolled(self) -> bool:
        # The dimension check retires profiles made by the MFCC version rather
        # than comparing against them, which would silently score zero.
        return (
            bool(self.embedding)
            and self.samples > 0
            and len(self.embedding) == EMBEDDING_DIM
        )

    @property
    def calibrated(self) -> bool:
        """Were these thresholds measured from this voice, or inherited?"""
        return self.match_at > 0.0 and self.reject_at > 0.0

    @property
    def match_threshold(self) -> float:
        return self.match_at if self.calibrated else CONFIDENT_MATCH

    @property
    def reject_threshold(self) -> float:
        return self.reject_at if self.calibrated else CLEARLY_DIFFERENT

    def to_dict(self) -> dict:
        return {
            "embedding": list(self.embedding),
            "vectors": [list(v) for v in self.vectors],
            "anchors": self.anchors,
            "samples": self.samples,
            "spread": self.spread,
            "created": self.created,
            "model": self.model,
            "match_at": self.match_at,
            "reject_at": self.reject_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "Profile":
        raw = data.get("embedding") or []
        embedding = tuple(float(x) for x in raw)
        stored = data.get("vectors") or []
        vectors = tuple(
            tuple(float(x) for x in v)
            for v in stored
            if isinstance(v, (list, tuple)) and len(v) == EMBEDDING_DIM
        )
        # A profile written before `vectors` existed has exactly one point in it,
        # its mean, so it is read as a one-clip profile and keeps scoring the way
        # it always did. Nothing about an old file starts behaving differently.
        backfilled = not vectors and len(embedding) == EMBEDDING_DIM
        if backfilled:
            vectors = (embedding,)
        # A profile written before anchors existed has only enrolment clips in
        # it, because adapt did not exist either, so all of them are anchors.
        anchors = int(data.get("anchors") or 0) or len(vectors)
        # `samples` on such a file counts the clips that were recorded, not the
        # clips that survived into it: four were averaged into one mean and the
        # other three are gone. Reporting the recorded count made --check and
        # --voicecheck say "4 clips" about a profile with one point in it, which
        # is the sort of confident wrong number those two exist to eliminate.
        samples = len(vectors) if backfilled else int(data.get("samples") or 0)
        return Profile(
            embedding=embedding,
            vectors=vectors,
            anchors=min(anchors, len(vectors)),
            samples=samples,
            spread=float(data.get("spread") or 0.0),
            created=str(data.get("created") or ""),
            model=str(data.get("model") or ""),
            match_at=float(data.get("match_at") or 0.0),
            reject_at=float(data.get("reject_at") or 0.0),
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


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def leave_one_out(vectors: list[np.ndarray]) -> list[float]:
    """For each clip, how well the rest of the profile recognises it.

    This is the only honest dress rehearsal available at enrolment time. Hold one
    clip back, build the profile from the others, and score the held-out clip
    exactly the way a live utterance will be scored. What comes back is the range
    of scores your own voice produces against your own profile, which is the
    number the bar has to sit below.
    """
    usable = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
    if len(usable) < 2:
        # One clip cannot be held out from itself. A single-clip profile scores
        # its own clip at 1.0, so there is nothing to measure and the caller
        # falls back to the floor.
        return [1.0] * len(usable)

    scores = []
    for index, held in enumerate(usable):
        others = usable[:index] + usable[index + 1 :]
        reference = [_unit(np.mean(np.vstack(others), axis=0))] + others
        scores.append(max(similarity(held, candidate) for candidate in reference))
    return scores


def calibrate(vectors: list[np.ndarray]) -> tuple[float, float]:
    """Where the match and reject bars belong for this particular voice.

    Derived, not guessed. The match bar sits a margin below the worst score your
    own clips produce against the rest of your own profile, clamped so that
    neither unusually tight nor unusually loose enrolment can produce a bar that
    is impossible or meaningless. The reject bar sits a further gap below that,
    because being ignored by your own assistant is a worse failure than
    answering somebody else once.
    """
    usable = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None]
    if not usable:
        return CONFIDENT_MATCH, CLEARLY_DIFFERENT

    worst = min(leave_one_out(usable))
    match = min(max(worst - CALIBRATION_MARGIN, MATCH_FLOOR), MATCH_CEILING)
    reject = max(match - REJECT_GAP, REJECT_FLOOR)
    return round(match, 4), round(reject, 4)


def score_against(profile: "Profile", vector: np.ndarray) -> float:
    """How much this utterance looks like the enrolled voice.

    The best of the stored clips and their mean, rather than the mean alone. A
    voice is a region and an average is a point: scored against the point, an
    utterance at the edge of your own range is penalised for variation you
    cannot help, which is exactly the failure this replaces.
    """
    if vector is None:
        return 0.0
    candidates = [np.asarray(v, dtype=np.float32) for v in profile.vectors]
    mean = np.asarray(profile.embedding, dtype=np.float32)
    if mean.size:
        candidates.append(mean)
    if not candidates:
        return 0.0
    return max(similarity(vector, candidate) for candidate in candidates)


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

        # More clips than the cap is not a problem worth solving cleverly: the
        # later ones are the ones said after the microphone settled.
        vectors = vectors[-MAX_VECTORS:]
        mean = _unit(np.mean(np.vstack(vectors), axis=0))
        # How much your own samples agreed with each other. Read back by
        # --enroll: if your own clips only score 0.5 against their own average,
        # nothing downstream can separate you from anybody else.
        spread = float(np.mean([similarity(mean, v) for v in vectors]))
        match_at, reject_at = calibrate(vectors)

        profile = Profile(
            embedding=tuple(float(x) for x in mean),
            vectors=tuple(tuple(float(x) for x in v) for v in vectors),
            # Everything from a deliberate enrolment is an anchor.
            anchors=len(vectors),
            samples=len(vectors),
            spread=spread,
            created=datetime.now().isoformat(timespec="seconds"),
            model=MODEL_REPO,
            match_at=match_at,
            reject_at=reject_at,
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

        score = score_against(profile, vector)
        if score >= profile.match_threshold:
            return MATCH, score
        if score <= profile.reject_threshold:
            # A confident-looking mismatch on a short clip is not confident, it
            # is just short. Only long enough utterances may be rejected.
            long_enough = audio.size >= int(MIN_REJECT_SECONDS * sample_rate)
            return (DIFFERENT if long_enough else UNSURE), score
        return UNSURE, score

    def adapt(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        """Add an utterance that is known to be yours. Returns whether it did.

        The root cause of the eleven silent days was a profile recorded once, in
        one sitting, on a microphone path that later changed underneath it, with
        no way to notice or catch up. A profile that never learns is a profile
        that can only drift. So an utterance confirmed as yours by evidence
        stronger than this model's own opinion, a typed line or a clear match,
        joins the profile and the bar is recalculated from the wider set.

        It refuses anything the profile already rejects, which is what stops a
        stranger or a television talking its way in one utterance at a time. That
        guard is the whole reason this is safe: the caller decides the utterance
        is yours, and this still declines if the voice says otherwise.
        """
        profile = self.profile
        if not profile.enrolled:
            return False
        if audio is None or audio.size < int(MIN_SECONDS * sample_rate):
            return False

        vector = embed(audio, sample_rate)
        if vector is None:
            return False
        if score_against(profile, vector) <= profile.reject_threshold:
            return False

        # The anchors stay. Only the learned tail rotates, oldest out first, so
        # the profile tracks the microphone as it is now without ever letting go
        # of the clips you actually sat down and recorded.
        anchors = list(profile.vectors[: profile.anchors])
        learned = list(profile.vectors[profile.anchors :])
        learned.append(tuple(float(x) for x in vector))
        room = max(MAX_VECTORS - len(anchors), 0)
        learned = learned[-room:] if room else []
        kept = anchors + learned

        arrays = [np.asarray(v, dtype=np.float32) for v in kept]
        mean = _unit(np.mean(np.vstack(arrays), axis=0))
        # Measured from the anchors alone. A learned clip can widen what counts
        # as a match, because scoring takes the best of every clip, but it must
        # never move the bar: an outlier scores badly against its peers, which
        # would drag the worst leave-one-out score down and quietly make the
        # NEXT acceptance easier. One lucky accept compounding into a standing
        # invitation is the failure that separation prevents.
        match_at, reject_at = calibrate(
            [np.asarray(v, dtype=np.float32) for v in anchors] or arrays
        )

        updated = Profile(
            embedding=tuple(float(x) for x in mean),
            vectors=tuple(kept),
            anchors=len(anchors),
            samples=len(kept),
            spread=float(np.mean([similarity(mean, v) for v in arrays])),
            created=profile.created,
            model=profile.model or MODEL_REPO,
            match_at=match_at,
            reject_at=reject_at,
        )
        self._profile = updated
        self._save(updated)
        return True
