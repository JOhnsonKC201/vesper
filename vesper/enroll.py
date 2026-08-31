"""Teaching Vesper what you sound like.

A short, deliberately unglamorous flow: it prints a sentence, you read it, it
records until you stop talking, four times over. The sentences are the ones you
will actually say to it, because a profile built from you reading a paragraph in
a careful voice is a profile of a voice you never use.

It reuses the real endpointer rather than recording for a fixed number of
seconds. That matters more than it sounds: the enrolled audio then has the same
leading pre-roll and the same trailing cut as every utterance it will later be
compared against, so the comparison is between like and like.
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
    embed,
    model_path,
    similarity,
)

PROMPTS = [
    "Vesper, how much disk space is left?",
    "Vesper, what am I looking at?",
    "Vesper, commit that and push it.",
    "Vesper, what is using the most memory right now?",
]

# Below this, your own four samples disagree with each other so much that no
# threshold could separate you from anyone else. Better to say so than to ship a
# profile that will quietly ignore you.
USABLE_SELF_AGREEMENT = 0.75


def run(config, *, prompts: list[str] | None = None) -> int:
    """Record a few utterances and save a voice profile. Returns an exit code."""
    prompts = prompts or PROMPTS
    store = VoicePrint(config.voiceprint_path())

    print("\nteaching Vesper your voice\n")
    if not available():
        # Fetched here rather than mid-conversation, so a missing model is a
        # one-time wait you asked for, not a network call while somebody is
        # halfway through a sentence.
        print("  fetching the speaker model, about 26MB, once...")
        if model_path(download=True) is None or not available():
            print("  could not fetch it. Without it Vesper answers every voice,")
            print("  exactly as it did before. Check your connection and retry.\n")
            return 1
        print("  got it.\n")

    print("It will show a sentence. Say it normally, at the distance and volume")
    print("you would actually use. It stops recording when you stop talking.\n")

    mic = Microphone(device=config.listening.device, on_error=lambda e: print(f"  {e}"))
    endpointer = Endpointer(
        VoiceActivity(),
        EndpointConfig(
            end_silence_ms=config.listening.end_silence_ms,
            max_utterance_s=config.listening.max_utterance_s,
        ),
    )

    clips: list[np.ndarray] = []
    try:
        mic.start()
        time.sleep(0.4)  # let the stream settle before the first prompt
        for index, prompt in enumerate(prompts, start=1):
            print(f"  {index} of {len(prompts)}   say:  {prompt}")
            clip = _record_one(mic, endpointer)
            if clip is None:
                print("     did not catch that, moving on\n")
                continue
            seconds = clip.size / 16000
            print(f"     got it, {seconds:.1f}s\n")
            clips.append(clip)
    finally:
        mic.stop()

    if len(clips) < 2:
        print("Not enough usable recordings. Check the microphone with")
        print("  run.bat --devices\n")
        return 1

    profile = store.enrol(clips)
    if not profile.enrolled:
        print("Could not build a profile from those recordings.\n")
        return 1

    print(f"saved {config.voiceprint_path()}")
    print(f"  samples          {profile.samples}")
    print(f"  self agreement   {profile.spread:.3f}")

    if profile.spread < USABLE_SELF_AGREEMENT:
        # Worth being blunt. A profile this loose will not reliably recognise
        # its owner, and quietly shipping it produces an assistant that ignores
        # you, which is the single worst outcome for this feature.
        print("\n  That is low: your own samples disagree with each other, which")
        print("  usually means background noise or a moving microphone. Vesper")
        print("  will still answer you, because an unsure match answers anyway,")
        print("  but it will not reliably tell you from anyone else.")
        print("  Run this again somewhere quieter to improve it.")
    else:
        print("\n  Vesper will now ignore voices clearly different from yours,")
        print("  and still answer whenever it is unsure.")

    _report_separation(clips, profile)
    return 0


def _record_one(mic: Microphone, endpointer: Endpointer, timeout: float = 20.0):
    """One utterance, or None if nothing usable arrived in time."""
    endpointer.reset()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        block = mic.read(timeout=0.5)
        if block is None:
            continue
        utterance = endpointer.feed(block, preroll=mic.preroll)
        if utterance is not None:
            if utterance.size < int(MIN_SECONDS * 16000):
                return None
            return utterance
    return None


def _report_separation(clips: list[np.ndarray], profile) -> None:
    """How far apart your own samples sit, as a sanity figure.

    Not a validation of whether it can reject somebody else, which cannot be
    measured without somebody else. It does catch the case where the recordings
    were so inconsistent that the profile is meaningless.
    """
    vectors = [v for v in (embed(c) for c in clips) if v is not None]
    if len(vectors) < 2:
        return
    stored = np.asarray(profile.embedding, dtype=np.float32)
    scores = sorted(similarity(stored, v) for v in vectors)
    print(f"  your own clips   {scores[0]:.3f} to {scores[-1]:.3f} against the profile")
