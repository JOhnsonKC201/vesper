# How Vesper works

Notes on the decisions that were not obvious, and the measurements behind them.
For usage, see the README.

---

## The brain is a subprocess, not an API client

The requirement was "use Claude Code, not the API". That is not a limitation to
work around; it is the whole design. `claude` is a native binary at
`C:\Users\johns\.local\bin\claude.exe`, and it speaks a bidirectional
newline-delimited JSON protocol:

```
claude -p --input-format stream-json --output-format stream-json --verbose
```

One process is spawned at startup and lives for the session. Each spoken turn is
one JSON line written to its stdin; the reply streams back as frames on stdout.
Because it is one process with one session id, it remembers the entire
conversation with no work on our side.

`vesper/brain/protocol.py` is pure: bytes in, events out, no subprocess. That
split is what makes the protocol testable against a recorded transcript, so a
change in the CLI's frame shape fails a unit test rather than silently making
Vesper mute.

### The flag that decides whether this is viable at all

Asking Claude to reply with one word, measured on this machine:

| Setup | Prompt tokens | Cost | Latency |
|---|---|---|---|
| Default environment | 284,681 | $1.14 | 7.1s |
| `--strict-mcp-config` + custom system prompt | 38,195 | $0.15 | 1.9s |
| `--safe-mode` + custom system prompt + `--tools` | 600 to 5,000 | $0.008 | 2.0s |

284,681 tokens is what the global `CLAUDE.md` plus every configured MCP server
(chrome-devtools, ruflo's ~300 tools, Canva, Gamma, Gmail, Notion, Supabase,
Playwright, Vercel) costs to load. A session pays it once, which is fine for a
coding session and fatal for something you talk to all day.

`--safe-mode` disables CLAUDE.md, MCP, hooks, plugins and skills **while keeping
OAuth subscription auth**. That last part is why it, specifically, is the right
flag. `--bare` disables the same things but forces `ANTHROPIC_API_KEY`, which
would break the one requirement that started this project.

There is a test asserting `--safe-mode` is always in the argv, because losing it
would not break anything visibly. It would just quietly cost a hundred times
more.

---

## Latency, and where it actually goes

From the moment you stop speaking:

| Stage | Measured |
|---|---|
| Endpointer confirms silence | 700ms (`end_silence_ms`) |
| Whisper `base.en` on CPU, 2.5s clip | 270 to 550ms |
| Claude time to first token | ~1,500ms |
| Piper starts playing | immediate, synthesis runs at 7.3x realtime |
| **First audible word** | **~2.3 to 2.6s** |

Three things keep that number down.

**Sentence-level flushing.** `--include-partial-messages` gives text deltas as
they generate. `brain/sentences.py` assembles them into sentences and hands each
one to the voice as it completes, so the first sentence plays while the third is
still being written.

**Greedy decoding for short clips.** Below three seconds, beam search buys
nothing and costs 150 to 300ms, so `beam_size` drops to 1.

**Holding phrases.** The hardest case is a question needing several shell
commands, where Claude generates no text until it has finished working. A
question that took four tool calls produced sixteen seconds of dead silence,
which reads as a crash. Now the first `ToolStarted` event with nothing spoken
yet triggers a local filler ("let me look", "one moment", never the same one
twice running). The same question now speaks at 3.2 seconds.

---

## Turn-taking

### Endpointing

Silero VAD, wrapped in `audio/vad.py`. Two details are load-bearing:

Silero v5+ accepts **exactly 512 samples at 16kHz**, and raises on anything
else. Echo Flow fed it 2048-sample slices inside a try/except for months, so the
model loaded, never once ran, and every decision silently came from the RMS
fallback. `VoiceActivity` therefore counts `neural_frames`, and a test asserts
it is non-zero. Microphone blocks are 480 samples, which never align to 512, so
a residual buffer carries the remainder between calls.

`end_silence_ms` defaults to 700, against a dictation tool's 1500. It is the
most-felt number in the app: too short cuts you off mid-thought, too long puts a
dead beat in every exchange.

### Pre-roll

By the time voice activity is confirmed, the first syllable is already in the
past. `Microphone` keeps a 400ms ring buffer, and the endpointer prepends it
when it opens an utterance.

This is not theoretical. The voice smoke test initially failed with "Vesper, how
many CPU cores" transcribed as "But how many CPU cores", because the test's mic
stand-in returned an empty pre-roll. The onset of the name was being eaten
before the endpointer confirmed speech.

### Barge-in

The mic stays live while Vesper speaks. Three consecutive blocks (90ms) of
speech, scored by a separate stricter VAD, cut playback off mid-word.

`Speaker` uses a generation counter rather than a bare stop flag. The race a
flag cannot close: a barge-in landing between the worker dequeuing a line and
starting to speak it. The worker would clear the flag and then speak exactly the
line the user just interrupted. So it clears the flag *before* re-reading the
generation, never after.

Playback is written in 1024-frame blocks with a stop check between them, giving
a worst-case interruption latency of ~45ms at 22.05kHz. You cannot interrupt
`sounddevice.play(); wait()`.

### Hearing itself

On laptop speakers the mic picks up Vesper's own voice, which without handling
makes it interrupt itself forever.

Rather than acoustic echo cancellation, any utterance is transcribed and its
words compared against the last few things Vesper said. Sixty percent overlap
means it heard itself, and the audio is discarded. Cheap, needs no extra model,
self-corrects, and degrades sensibly: utterances under two words are never
treated as echo, so "yes" and "stop" always get through.

`half_duplex: true` is the alternative for a loud room: deaf while speaking, no
barge-in, no chance of self-hearing.

---

## Wake word

Matched on the transcript, not on raw audio. The utterance is being transcribed
anyway, so this costs nothing and needs no second model.

Three layers, in order:

1. **Decoder biasing.** `initial_prompt="Talking to an assistant named Vesper,
   also called Jarvis."` makes Whisper expect the name. Measured on synthesized
   speech: one correct out of three without it, three out of three with it.
2. **Homophone list** for known substitutions (vespa, vester, vespr).
3. **Fuzzy matching** over the first three words and adjacent word pairs, at a
   0.72 similarity threshold, catching mishearings nobody enumerated.

The list originally contained "whisper" and "jasper". Both had to come out. They
are ordinary English words, and "whisper it to me quietly" waking the assistant
during a meeting is a far worse failure than missing one wake, which you recover
from by saying the name again. The threshold and the list are both tested
against a set of innocent sentences that must never trigger.

After it answers, a 25 second follow-up window accepts plain speech, so a
conversation does not require saying the name before every sentence.

---

## Two channels

Long output must never be read aloud. Claude wraps it in `<screen>` tags; the
tags and their contents are stripped from the speech stream and printed instead.

`brain/channels.py` does this on the *stream*, which is harder than on a finished
reply: a tag arrives split across deltas, so at any moment the buffer may end
with `<scr`, which could be the start of a tag or someone talking about scripts.
The rule is to emit everything that definitely is not part of a tag and hold
back any trailing text that could still become one. Worst case is eight
characters held for one more delta.

`clean_for_speech` is the backstop for when the model drifts and emits markdown
anyway. A stray asterisk read aloud as "asterisk" breaks the illusion instantly.

---

## Ambient awareness

`sensors/` reads the foreground window, idle time and machine vitals. The
`context_block()` that rides on every turn is about 34 tokens:

```
time: Friday 28 August, 01:45
focused window: Code.exe: whisper.py
user: at the keyboard
cpu 6%, ram 60%, battery 100% on ac, disk 597gb free
```

That is what makes "what am I looking at" and "why is this failing" work without
explanation. It is labelled `[machine context, not spoken by the user]` so the
model treats it as ambient awareness rather than as instructions.

Window titles are the most revealing thing Vesper reads, so titles matching
password managers, banking or incognito are replaced with `[hidden]`, keeping
only the app name.

### The proactive loop, and why it is nearly all restraint

`changes_since()` returns `Change` objects carrying a `notable` flag. App
switches and idle transitions are recorded as context but are **never** notable.
Without that gate, a normal working hour spends a Claude turn every two minutes
at roughly two cents each.

When something notable does happen, the loop does not decide what to say. It
describes what changed and asks Claude whether it is worth interrupting for,
with instructions to reply `SILENT` otherwise. Observed:

| Signal | Verdict |
|---|---|
| switched from Chrome to Code.exe | not even asked (not notable) |
| cpu jumped 12% to 97% | asked, answered SILENT |
| unplugged at 12%, battery to 8% | "Battery's at eight percent and unplugged, plug in now or you'll lose that ffmpeg job." |
| only 4gb left on C | "C drive's down to four gigabytes free, worth clearing something out soon." |

Then: at most one remark per fifteen minutes, silent in quiet hours, silent
while a conversation is in progress (`brain.busy`), silent when the user has
been idle fifteen minutes, and it is told what it already said so it cannot
repeat itself. A rambling reply that ignores the format is treated as a refusal.

Mute is handled locally in `conversation.py`, never sent to Claude. Telling
something to be quiet should not require a network round trip, and must work
while it is mid-sentence.

---

## Testing

260 tests, 84% coverage, 13 seconds, no microphone or speakers required.

The interesting part is what is faked and what is not.

**Not faked.** `tests/fake_claude.py` is a real subprocess speaking the real
stream-json protocol, so pipes, reader threads, framing, interrupts and crash
recovery are all genuinely exercised at zero cost.
`tests/fixtures/turn_with_tool.jsonl` is a recorded transcript from the actual
CLI, covering the one thing the fake cannot: that the flags are still valid.
`tests/fixtures/speech_*.wav` is real speech synthesized locally by Piper, because
Silero scores a sine wave near zero and a tone-based test would pass with the
model completely broken.

**Faked.** Audio devices, via a stand-in `sounddevice` module. The two most
valuable behaviours in `mic.py` are both failure paths (a start that raises must
still close the stream, a stop that raises must still clear the running flag)
and both are awkward to trigger on working hardware.

`tests/test_privacy.py` enforces the privacy claim by parsing every module's AST:
no network imports, no sockets, no hardcoded URLs, and `claude.py` is the only
file permitted to spawn a subprocess.

`scripts/voice_smoke.py` is the end-to-end proof: Piper renders a question, the
audio goes through the mic seam, and Silero, Whisper, the wake gate, Claude and
the voice all run for real.

---

## Things deliberately not done

**No overlay window.** A subagent traced pixelpets as a possible host and the
verdict was clear: `renderer.js` is 2,890 lines in one global scope with the pet
state machine inseparable from the window, `sandbox: true` with no mic
permission handling, and a CSP with no `connect-src`. A voice orb is a separate
~400 line job, and the ~200 lines genuinely worth taking are the cross-platform
always-on-top and click-through knowledge, not the code.

**No GPU.** `torch` here is the CPU build, so the RTX 5060 sits idle.
`base.en` at 0.4s is fast enough that a cu128 install was not worth the risk.

**No second model for proactive checks.** Haiku would be cheaper per check, but
it would need a second process and would lose the shared conversation context
that let the battery warning reference the ffmpeg job. The notability gate makes
checks rare enough that this does not matter.

---

## Dependency pins

```
ctranslate2==4.7.2
av==17.0.1
```

Newer wheels of both are blocked on this machine by Windows Application Control:

```
ImportError: DLL load failed while importing link:
An Application Control policy has blocked this file.
```

The actual culprit is PyAV's native libraries; faster-whisper imports `av`
eagerly even when it is only ever handed numpy arrays. Both pinned versions were
already trusted here. Do not float either without confirming a model loads.
