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

## Consent

Vesper reads without asking and changes nothing without a spoken yes. The
interesting part is where that boundary is enforced, because three of the four
obvious places to put it are worthless.

**Not in the prompt.** "Ask before writing files" is a request, and a model that
drifts, or that has just read a file telling it otherwise, can fail to honour
it. The prompt shapes how the refusal *sounds*; it must not be the thing doing
the refusing.

**Not in the tool list alone.** Leaving `Write` out of `--tools` stops the Write
tool and stops nothing else. Measured against the real CLI: asked to create a
file with `Write` unavailable, Claude ran `mkdir -p … && printf 'hello' > file`
and reported success.

**Not in an allowlist containing bare `Bash`.** The same hole, one step further
in. `Bash(python*)` has it too, since `python -c` writes files, deletes them and
opens sockets. The original read allowlist had exactly that entry.

**In the CLI's permission layer.** `--permission-mode manual` plus an allowlist
of specific read-only verbs refuses everything else before it runs, and emits a
`system/permission_denied` frame naming the tool. That frame is the whole
mechanism. With a narrow list, every bypass route above is refused, including
output redirection inside an otherwise-allowed command.

The flow:

```
  claude tries Write --> CLI refuses --> permission_denied frame
                                               |
                      ActionRequest <-----------+   (input recovered from the
                             |                       preceding tool_use frame)
                             v
               "I want to create notes dot txt. Do I do this for you?"
                             |
         +-------------------+--------------------+
       "yes"             "no", something else, or nothing
         |                   |
    grant one verb      log it, tell Claude to drop it
    respawn --resume
    run the action
    revoke, respawn
```

Four decisions inside that are load-bearing:

**The denial frame does not carry the tool's input**, only its name. "I want to
run a command, shall I?" is not a question anyone can answer, so `StreamParser`
remembers tool calls by id for the length of a turn and joins the input back on.

**A grant is a respawn.** The CLI fixes its allowlist at spawn, so a running
process cannot be widened. That is affordable precisely because it only happens
when someone has just said yes out loud, and `--resume` carries the conversation
across it.

**A grant is one verb, handed back immediately.** `git commit` produces
`Bash(git commit:*)`, never bare `Bash`, so an approval cannot be spent on
something else. Revocation runs behind the answer being spoken and takes the
busy lock, so the next turn waits for read-only rather than racing it. Deferring
it to the next question would have been simpler and wrong: the ambient loop
shares this brain, and an unattended proactive turn could inherit permission
given for something else entirely.

**Only a yes approves.** Silence, an unrelated question and a "yes but" all
leave the request unapproved, and it expires after `consent.window_s` so a yes
meant for something else cannot land on a stale request. Every outcome is
appended to `var/actions.log`, since spoken consent leaves no other trace.

One consequence worth knowing: the persona has to tell Claude to *attempt* the
action rather than announce it. An earlier wording ("say what you want to do")
produced a turn where it described the write in prose without calling the tool,
so no denial frame was emitted, so nothing was asked, and a spoken "yes" would
have landed on nothing. The gate only fires if the attempt is real.

### Two prompts that were not enough

Both of these were written as instructions first, and both had to become code.

**Claude narrating the refusal.** After being stopped it would add its own
sentence in front of the question: "Waiting on you.", or worse, "That was
blocked, not by the approval layer, but by the sandbox's allowed directories",
which was simply wrong. It does not know why it was stopped and guesses badly.
The prompt was changed to forbid this by name, and the very next run said
"Waiting on you for that one." So `is_refusal_noise` in `persona.py` drops
sentences that only restate a refusal, and only within a turn that was actually
refused, so asking "why was that blocked?" later still gets an answer. Same
backstop, same reasoning, as `clean_for_speech` for markdown.

**Undo.** Telling the model to be careful is not a recovery mechanism.
`undo.py` copies the target aside before the grant is issued, never after: once
the process is respawned with the permission, the action can happen at any
moment. "Undo that" is handled locally alongside the mute phrases, because it is
what you say when something has gone wrong and it must not depend on the network
or on Claude's memory of what it changed.

Undo refuses to guess. A removal command containing a glob, a variable, a
redirect or a chain returns no targets at all rather than a partial list, since
a wrong answer means silently missing a file. Anything it cannot reverse gets a
specific sentence explaining why, because "no" on its own is not an answer.

### Checking the gate from outside

`vesper/gatecheck.py` asks the real CLI to write a file it must not be allowed
to write, and fails the self check if it succeeds, if it is refused without a
report, or if the child errors. It runs last in `--check` because it is the only
step that costs a turn, and it is the only step whose failure is dangerous.

A self test that cannot fail is worse than none, so it was verified in both
directions against claude 2.1.250: `manual` gives `refused and reported`, and
`auto` gives `the write went through`.

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

435 tests, 86% coverage, 27 seconds, no microphone or speakers required.

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
