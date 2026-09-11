"""Does the stored profile actually recognise you? Measured, not assumed.

This exists because the opposite went unnoticed for eleven days. The shipped
profile scored 279 real utterances with a maximum of 0.54 against a bar of 0.55,
so it matched nothing, ever, and rejected 266 of them. Every part of that was
visible in `var/vesper.log` and none of it was visible to the user, who simply
experienced an assistant that ignored them.

So: say a few things, see the numbers, and be told plainly whether to re-enrol.
It reuses the real microphone and the real endpointer, so what is scored here is
what the live loop would have scored.
"""

from __future__ import annotations

import time

import numpy as np

from .audio.mic import Microphone
from .audio.vad import EndpointConfig, Endpointer, VoiceActivity
from .stt.voiceprint import (
    MIN_SECONDS,
    VoicePrint,
    available,
    calibrate,
    embed,
    model_path,
    score_against,
)

PROMPTS = [
    "Vesper, what is using the most memory right now?",
    "Vesper, how much disk space is left?",
    "Vesper, open my calendar.",
]


def run(config, *, prompts: list[str] | None = None) -> int:
    """Score a few live utterances against the stored profile."""
    prompts = prompts or PROMPTS
    store = VoicePrint(config.voiceprint_path())
    profile = store.profile

    print("\nchecking whether Vesper knows your voice\n")

    if not profile.enrolled:
        print("  No profile. Every voice is accepted, including other people's.")
        print("  Run:  run.bat --enroll\n")
        return 1

    print(f"  recorded       {profile.created or 'unknown'}")
    print(f"  clips stored   {profile.samples}")
    print(f"  self agreement {profile.spread:.3f}")
    if profile.calibrated:
        print(f"  match at       {profile.match_threshold:.3f}  "
              f"(measured from your own clips)")
        print(f"  reject below   {profile.reject_threshold:.3f}")
    else:
        print(f"  match at       {profile.match_threshold:.3f}  "
              f"(INHERITED DEFAULT, never checked against you)")
        print(f"  reject below   {profile.reject_threshold:.3f}")
        print("\n  This profile predates calibration. That is the exact shape of")
        print("  the bug that ignored 266 utterances: a bar nobody measured.")

    if not available():
        print("\n  The speaker model is not here, so nothing can be scored.")
        print("  Run:  run.bat --enroll\n")
        return 1
    if model_path() is None:
        print("\n  The speaker model is missing.\n")
        return 1

    print("\n  Say each line normally, at your usual distance.\n")
    scores = _collect(config, prompts)
    if not scores:
        print("\n  Heard nothing usable. Check the microphone with --devices\n")
        return 1

    return _report(profile, scores)


def _collect(config, prompts: list[str]) -> list[float]:
    """Record each prompt and score it the way the live loop would."""
    store = VoicePrint(config.voiceprint_path())
    profile = store.profile
    mic = Microphone(device=config.listening.device, on_error=lambda e: print(f"  {e}"))
    endpointer = Endpointer(
        VoiceActivity(),
        EndpointConfig(
            end_silence_ms=config.listening.end_silence_ms,
            max_utterance_s=config.listening.max_utterance_s,
        ),
    )

    scores: list[float] = []
    try:
        mic.start()
        time.sleep(0.4)  # let the stream settle before the first prompt
        for index, prompt in enumerate(prompts, start=1):
            print(f"  {index} of {len(prompts)}   say:  {prompt}")
            clip = _record_one(mic, endpointer)
            if clip is None:
                print("     did not catch that, moving on\n")
                continue
            vector = embed(clip)
            if vector is None:
                print("     could not score that one\n")
                continue
            score = score_against(profile, vector)
            verdict, _ = store.compare(clip)
            print(f"     {score:.3f}   {verdict}\n")
            scores.append(score)
    finally:
        mic.stop()
    return scores


def _record_one(mic: Microphone, endpointer: Endpointer, timeout: float = 20.0):
    """One utterance, or None if nothing usable arrived in time."""
    endpointer.reset()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        block = mic.read(timeout=0.5)
        if block is None:
            continue
        utterance = endpointer.feed(block)
        if utterance is not None:
            if utterance.size < int(MIN_SECONDS * 16000):
                return None
            return utterance
    return None


def _report(profile, scores: list[float]) -> int:
    """Say plainly whether this profile works, and what to do if it does not."""
    best, worst = max(scores), min(scores)
    matched = sum(1 for s in scores if s >= profile.match_threshold)
    rejected = sum(1 for s in scores if s <= profile.reject_threshold)

    print("  ---")
    print(f"  your voice scored {worst:.3f} to {best:.3f}")
    print(f"  matched {matched} of {len(scores)}, "
          f"would have been ignored {rejected} of {len(scores)}")

    if rejected:
        print("\n  IGNORED YOU. That is the failure, not a tuning preference.")
        print("  Re-record the profile:  run.bat --enroll\n")
        return 1

    if not matched:
        # The eleven-day failure exactly: never rejected, never confirmed, so
        # every spoken instruction was demoted and had to be said twice.
        print("\n  Never confidently matched you. Vesper will answer, but it")
        print("  treats you as unidentified, which means anything you teach it")
        print("  has to be said twice before it sticks.")
        would_be, _ = calibrate([np.asarray(v, dtype=np.float32) for v in profile.vectors])
        print(f"  Re-enrolling would set the bar near {would_be:.2f} "
              f"instead of {profile.match_threshold:.2f}.")
        print("  Re-record the profile:  run.bat --enroll\n")
        return 1

    print("\n  Working. It knows you, and a clearly different voice is ignored.\n")
    return 0
