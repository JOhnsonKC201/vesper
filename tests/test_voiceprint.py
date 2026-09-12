"""The voice profile: does it recognise its owner, and only its owner.

Written after eleven days of real logs showed the shipped profile had never once
produced a match. 279 scored utterances in `var/vesper.log`, median 0.32, and a
maximum of exactly 0.54 on every single day, against a hardcoded
`CONFIDENT_MATCH = 0.55`. The model was never the problem: enrolled on two of
the real fixture clips it scores the third at 0.80 and a different real speaker
at 0.05. What failed was a single averaged centroid plus a constant that was
never checked against this user's own numbers.

So these tests pin the two things that stop it happening again: the bar is
derived from your own clips rather than guessed, and a clip is scored against
every sample you gave rather than against their average alone.
"""

import json
import wave

import numpy as np
import pytest

from vesper.stt import voiceprint as vp


def _load(path):
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


@pytest.fixture(scope="module")
def speaker_a(request):
    root = request.path.parent / "fixtures"
    return [_load(root / f"speaker_a_{i}.wav") for i in (1, 2, 3)]


@pytest.fixture(scope="module")
def speaker_b(request):
    root = request.path.parent / "fixtures"
    return [_load(root / f"speaker_b_{i}.wav") for i in (1, 2, 3)]


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _vector(seed):
    """A deterministic unit vector of the right width."""
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=vp.EMBEDDING_DIM))


# The speaker model is a 26MB download that `--enroll` fetches once, so CI does
# not have it and `embed` returns None there. The same guard test_unattended.py
# uses, for the same reason: these tests are about real audio, and without the
# model there is no audio to be right or wrong about.
needs_model = pytest.mark.skipif(
    not vp.available(),
    reason="the speaker model is not downloaded; run --enroll once",
)


def _mixed(seed, other, weight):
    """A vector like `seed` but pulled `weight` of the way toward `other`."""
    return _unit(_vector(seed) * (1.0 - weight) + _vector(other) * weight)


# --- calibration ------------------------------------------------------------


def test_the_bar_sits_below_your_own_worst_agreement():
    """The number that failed was a constant. This one is a measurement."""
    clips = [_vector(1), _mixed(1, 2, 0.10), _mixed(1, 3, 0.15)]
    match, reject = vp.calibrate(clips)
    assert match < min(vp.leave_one_out(clips))
    assert reject < match


def test_noisy_enrolment_cannot_demand_perfection():
    """Clips that barely agree must not produce an unreachable bar.

    This is the shipped failure in one line: self agreement of 0.83 produced a
    profile whose owner never once cleared 0.55.
    """
    match, _ = vp.calibrate([_vector(3), _vector(4), _vector(5)])
    assert vp.MATCH_FLOOR <= match <= vp.MATCH_CEILING


def test_two_near_identical_clips_cannot_demand_everything():
    """Self agreement near 1.0 is a sign of two copies, not of a great profile."""
    one = _vector(7)
    match, _ = vp.calibrate([one, one])
    assert match <= vp.MATCH_CEILING


def test_rejection_stays_clear_of_your_own_band():
    """Whatever else happens, the owner's own worst clip is never a rejection."""
    clips = [_vector(11), _mixed(11, 12, 0.20), _mixed(11, 13, 0.30)]
    match, reject = vp.calibrate(clips)
    assert reject <= match - 0.15
    assert reject < min(vp.leave_one_out(clips))


def test_calibration_survives_a_single_clip():
    match, reject = vp.calibrate([_vector(17)])
    assert vp.MATCH_FLOOR <= match <= vp.MATCH_CEILING
    assert 0.0 < reject < match


def test_calibration_of_nothing_falls_back_to_the_defaults():
    assert vp.calibrate([]) == (vp.CONFIDENT_MATCH, vp.CLEARLY_DIFFERENT)


# --- the profile keeps every clip ------------------------------------------


@needs_model
def test_a_profile_keeps_every_clip_not_just_their_average(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    profile = store.enrol(
        [speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1]
    )
    assert len(profile.vectors) == 2
    assert all(len(v) == vp.EMBEDDING_DIM for v in profile.vectors)
    # The mean is still there, because it is what a stranger is furthest from.
    assert len(profile.embedding) == vp.EMBEDDING_DIM
    assert profile.calibrated is True


def test_scoring_takes_your_best_clip_not_their_average():
    """Why the average failed: it sits between two unlike clips, near neither.

    A profile built from a close voice and a distant one has a centroid that
    matches nothing. Scored against the clips themselves, the close one wins.
    """
    near, far = _vector(21), _vector(22)
    mean = _unit(near + far)
    profile = vp.Profile(
        embedding=tuple(float(x) for x in mean),
        vectors=(tuple(float(x) for x in near), tuple(float(x) for x in far)),
        samples=2,
        match_at=0.6,
        reject_at=0.3,
    )
    spoken = _unit(near * 0.97 + far * 0.03)
    assert vp.score_against(profile, spoken) > 0.9
    assert float(np.dot(spoken, mean)) < 0.9


def test_an_old_single_mean_profile_still_loads(tmp_path):
    """The file on disk today has no `vectors` key. It must keep working."""
    legacy = {
        "embedding": [float(x) for x in _vector(31)],
        "samples": 4,
        "spread": 0.8321347087621689,
        "created": "2026-08-31T00:20:28",
        "model": vp.MODEL_REPO,
    }
    path = tmp_path / "vp.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    profile = vp.VoicePrint(path).profile
    assert profile.enrolled
    assert profile.samples == 4
    # Treated as one clip, so scoring behaves exactly as it used to.
    assert len(profile.vectors) == 1


def test_a_legacy_profile_is_reported_as_uncalibrated(tmp_path):
    """It is the uncalibrated ones that ignore their owner, so say so."""
    legacy = {"embedding": [float(x) for x in _vector(33)], "samples": 4}
    path = tmp_path / "vp.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert vp.VoicePrint(path).profile.calibrated is False


def test_a_legacy_profile_still_uses_the_old_constants(tmp_path):
    """No calibration means no invented thresholds: behave exactly as before."""
    legacy = {"embedding": [float(x) for x in _vector(35)], "samples": 4}
    path = tmp_path / "vp.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    profile = vp.VoicePrint(path).profile
    assert profile.match_threshold == vp.CONFIDENT_MATCH
    assert profile.reject_threshold == vp.CLEARLY_DIFFERENT


# --- real voices ------------------------------------------------------------


@needs_model
def test_the_same_real_speaker_matches(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1])
    verdict, score = store.compare(speaker_a[2][0], speaker_a[2][1])
    assert verdict == vp.MATCH, f"own voice scored {score:.3f}"


@needs_model
def test_a_different_real_speaker_is_rejected(tmp_path, speaker_a, speaker_b):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1])
    for index, (clip, rate) in enumerate(speaker_b, start=1):
        verdict, score = store.compare(clip, rate)
        assert verdict == vp.DIFFERENT, f"stranger {index} scored {score:.3f}"


# --- learning as it goes ----------------------------------------------------


@needs_model
def test_a_confirmed_utterance_joins_the_profile(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0]], sample_rate=speaker_a[0][1])
    before = len(store.profile.vectors)
    assert store.adapt(speaker_a[1][0], speaker_a[1][1]) is True
    assert len(store.profile.vectors) == before + 1
    # And it is on disk, not only in memory.
    assert len(vp.VoicePrint(tmp_path / "vp.json").profile.vectors) == before + 1


@needs_model
def test_learning_refuses_a_voice_that_is_not_yours(tmp_path, speaker_a, speaker_b):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1])
    before = len(store.profile.vectors)
    assert store.adapt(speaker_b[0][0], speaker_b[0][1]) is False
    assert len(store.profile.vectors) == before


@needs_model
def test_learning_is_capped_so_the_file_cannot_grow_forever(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0]], sample_rate=speaker_a[0][1])
    for _ in range(vp.MAX_VECTORS + 5):
        store.adapt(speaker_a[1][0], speaker_a[1][1])
    assert len(store.profile.vectors) <= vp.MAX_VECTORS


def test_learning_does_nothing_without_a_profile(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    assert store.adapt(speaker_a[0][0], speaker_a[0][1]) is False


# --- learning must not make the next acceptance easier -----------------------
#
# From a security review of the adaptive profile. Scoring takes the best of
# every stored clip, so one accepted utterance becomes a lasting exemplar. That
# is the point when it is really you. It is a problem if the same clip also
# moves the bar: an outlier scores badly against its peers, so recalculating
# from the widened set drags the worst leave-one-out score down and lowers the
# bar for whoever comes next. One lucky accept would buy a standing invitation.


@needs_model
def test_enrolment_clips_are_marked_as_anchors(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    profile = store.enrol(
        [speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1]
    )
    assert profile.anchors == 2
    assert profile.anchors == len(profile.vectors)


@needs_model
def test_learning_never_moves_the_bar(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1])
    before = (store.profile.match_at, store.profile.reject_at)

    assert store.adapt(speaker_a[2][0], speaker_a[2][1]) is True
    assert (store.profile.match_at, store.profile.reject_at) == before


@needs_model
def test_learning_does_not_promote_itself_to_an_anchor(tmp_path, speaker_a):
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0]], sample_rate=speaker_a[0][1])
    store.adapt(speaker_a[1][0], speaker_a[1][1])

    assert store.profile.anchors == 1
    assert len(store.profile.vectors) == 2
    # And it survives a round trip, or the next run would treat it as enrolled.
    assert vp.VoicePrint(tmp_path / "vp.json").profile.anchors == 1


@needs_model
def test_the_clips_you_recorded_are_never_evicted(tmp_path, speaker_a):
    """The tail rotates. What you sat down and recorded does not."""
    store = vp.VoicePrint(tmp_path / "vp.json")
    store.enrol([speaker_a[0][0], speaker_a[1][0]], sample_rate=speaker_a[0][1])
    anchors = store.profile.vectors[:2]

    for _ in range(vp.MAX_VECTORS + 6):
        store.adapt(speaker_a[2][0], speaker_a[2][1])

    assert store.profile.anchors == 2
    assert store.profile.vectors[:2] == anchors
    assert len(store.profile.vectors) <= vp.MAX_VECTORS


def test_a_profile_from_before_anchors_treats_its_clips_as_anchors(tmp_path):
    """Those clips came from --enroll, because adapt did not exist yet."""
    legacy = {
        "embedding": [float(x) for x in _vector(41)],
        "vectors": [[float(x) for x in _vector(41)]],
        "samples": 1,
        "match_at": 0.5,
        "reject_at": 0.3,
    }
    path = tmp_path / "vp.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert vp.VoicePrint(path).profile.anchors == 1
