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

### When the CLI cannot run the turn at all

The CLI reports a turn it could not run as an ordinary `result` frame with
`is_error` set and its own explanation as the text. Until 2026-09-04 that text
took the same path as an answer. That night the laptop woke from a 21 hour
sleep with the saved login expired, every turn came back `authentication_failed`,
and Vesper said "Failed to authenticate: OAuth session expired and could not be
refreshed" three times in its own voice, counted each as a success, and left
nothing in the log.

`brain/failures.py` names the failure from the result's `is_error`, from the
error code the CLI puts on its message (`authentication_failed`) and, as a
fallback, from the wording. The ladder in `Conversation._failed_turn` is: log
the CLI's text as an ERROR and count a failure; for anything but a dead login,
say one plain sentence and stop; for a dead login, respawn the brain once and
ask the same thing again, because a fresh child re-reads the credentials file
and that is the only way a login renewed in some other terminal reaches this
process; if the retry fails too, lock out. Locked out, every utterance first
runs `claude auth status` (exit 0 logged in, 1 not), a subprocess rather than
a model call, and the moment it says logged in the brain is respawned and the
turn goes through. The same check runs before the brain is spawned at start,
so a Vesper launched onto a dead login says so instead of failing quietly all
evening.

One mechanical point. `ClaudeBrain.ask` holds its busy lock for as long as its
generator is open, so the conversation closes the generator before it calls
`restart()`, which takes the same lock. A respawn from inside the loop would
deadlock, and the fake brain in the tests has no lock to show it.

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

A 25 second window then accepts plain speech, so a conversation does not
require saying the name before every sentence. Two details decide whether it
feels right, and this document described neither for a while because the window
was switched off in the shipped config and nobody was living with it.

It is measured from **the room going quiet**, not from the end of the turn.
`half_duplex` makes the microphone deaf while Vesper speaks, and `TurnComplete`
fires before playback finishes, so a clock started when the answer was ready is
spent on Vesper's own voice: a fifteen second reply left you seven seconds to
respond in. It restarts when the speaker actually drains, noticed on the audio
loop that is already running thirty times a second rather than on a timer. And
every utterance the gate accepts pushes it out again, so it tracks the
conversation rather than the last reply.

The window decides whether Vesper **listens**, never whether he may **act**.
`WakeResult.reason` is what separates the two: only `wake-word` may approve a
change to the machine or end the process. A "yes" said to someone else in the
room lands inside the window all the time, and "quit" is ordinary English.

`WakeGate` holds one float and reads no clock at all. `now` is a parameter on
every method, which is what makes the whole thing testable without sleeping.

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

A "yes but" is its own answer, `QUALIFIED`, and it is not a refusal. On
2026-09-05 "Sure, do it, but in front of me" was read as a no, the refusal note
told Claude never to try again, and four more "do it"s were turned down with
"I don't relitigate a refusal". Now the question stays open, Vesper says it only
acts on a plain answer (in words that avoid "yes", "no" and its own name, or
the echo filter would discard the reply it is asking for), and when the plain
yes comes the condition rides along in the approved note as an instruction
about how. The refusal note itself now says what it always meant: no going
around a refusal on Vesper's own initiative, but a user who asks again later
gets asked again.

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

## Speech out, when it is not local

`tts/base.py` defines a three member protocol: `name`, `speak(text, stop)`,
`close()`. Backends own their own playback rather than returning samples,
because SAPI speaks through COM and never hands you a buffer, and "make this
audible, and stop the moment you are told" is the honest contract.

`ElevenTTS` is not a fourth backend alongside piper, sapi and none. It is a
router holding a `PiperTTS` as a member:

```
speak(text, stop):
    cache hit                      -> play from disk, free, works offline
    no key, no budget, any failure -> self.fallback.speak(text, stop)
    otherwise                      -> stream, play, cache, count
```

Two reasons for that shape, and the second is the important one.

`Speaker.voice` is read inside the worker thread, so changing voice on a cloud
backend costs nothing: `use()` assigns a string that `speak` reads once at the
top, and the worst case is a sentence already in flight finishing in the old
voice, which is correct anyway.

A local backend cannot do that. It holds a loaded ONNX model and an open output
stream, so switching means building a second one and handing it over, which is
what `Speaker.use_voice` is for. The handover this originally avoided turned
out to be three lines in a fixed order: barge in first, because Piper holds the
whole utterance inside its own lock and `close()` would otherwise block the
caller for the length of the sentence; swap under the Speaker's lock, so a
`say()` landing at the same moment queues against one voice or the other rather
than half of each; close the old one last, because two output streams fighting
over the device is a real symptom rather than a tidiness point.

Avoiding it for as long as we did had a cost. The dashboard only got a voice
picker when the cloud voice was running, and the default install is Piper, so
the setting was not there at all.

More importantly, the local commands must survive an outage. Mute, undo and
shutdown are handled without touching Claude precisely so they work with the
network down. If they went through a cloud voice with no local fallback, an
outage would take away the ability to shut the assistant up.

### The cache is not an optimisation

Seventeen lines are hardcoded into the assistant: the fillers, "Yes?", "Doing
it.", "Shutting down.". They cost about 350 characters to synthesize once.

The money matters on a 10,000 character monthly allowance, but the latency
matters more. A cached line plays with no round trip, so the holding phrase
covering a tool call, the most latency-sensitive line in the system, becomes
the fastest rather than the slowest. Measured: 519ms cold, 0ms cached,
identical audio.

### What the free tier actually refuses

Measured against a real key, because none of this is in the docs and the free
tier behaves like a different product wearing the same API:

| | |
|---|---|
| `pcm_22050` | works, which is why there is no audio decoder in this project |
| Library voices | 402, all of them |
| Aria and Charlotte | 402, despite being listed as premade |
| The other 18 premade voices | work |
| Creating a voice | 403, whatever permissions the key carries |
| Listing voices | 401, so the voice list is a measured constant |

---

## Lessons

`session.json` carries the conversation id across restarts, but a conversation
gets compacted and eventually dropped, and "stop reading me file paths" should
outlive that. So an instruction given out loud is written to `var/lessons.json`
and put into the system prompt at the start of every future session.

It is not machine learning, and `learning.py` says so in its first paragraph.
Nothing trains. The value is entirely in two rules.

**A correction has to be repeated.** "Remember that", "from now on", "always",
"never" and "don't" count immediately. "No, I meant..." is stored but kept out
of the prompt until it has happened twice, because a single "no" in a noisy
transcript must not become a permanent rule.

**The negation has to survive.** "Don't read me file paths" captures as "read
me file paths" if you strip the trigger and stop thinking, and the stored rule
then causes the exact behaviour that was complained about, forever, with
nothing about the stored line looking wrong. The prefix is per pattern rather
than one shared "do not", because "stop reading" has to stay "stop reading".

The costs are asymmetric: a missed lesson means saying it again, a wrong one
changes behaviour on every turn in a way that is very hard to trace back. That
asymmetry is why the length bounds, the question check and the repetition
threshold exist, and why most of `test_learning.py` is about what must *not* be
learned.

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
