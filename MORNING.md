# What I built while you were asleep

**Vesper** is at `C:\Vesper`, with a shortcut on your Desktop.

To try it right now:

```
run.bat --text          type to it, no microphone needed
run.bat                 talk to it out loud
run.bat --check         confirm every dependency (all green as of 04:00)
```

It runs on the `claude` CLI and your subscription. No API key exists anywhere in
the project.

---

## Try these first

The fastest way to see whether it is any good:

1. `run.bat --text`, then type **"what am I looking at right now?"**
   It reads your foreground window from the ambient context and answers without
   you explaining anything.

2. Then **"why might that be slow?"**
   It has a real shell and will go and measure rather than guess.

3. Then `run.bat` and say **"Vesper, what did I ask you a minute ago?"**
   It remembers across process restarts.

4. Talk over it mid-sentence. It stops in about 90 milliseconds.

---

## The decision that mattered most

I measured `claude -p` before writing anything else. Asking it to say one word,
in your normal environment: **284,681 prompt tokens, $1.14, 7.1 seconds.**

That is your global `CLAUDE.md` plus the tool schemas of every MCP server you
have configured (ruflo alone is ~300 tools). Fine for a coding session, fatal
for something you talk to all day.

With `--safe-mode` plus an explicit system prompt and tool list: **376 to 5,000
tokens, $0.008 to $0.015, 2.0 seconds.** Roughly a hundredfold reduction.

`--safe-mode` is specifically right because it strips all that **while keeping
OAuth subscription auth**. `--bare` looks equivalent but forces
`ANTHROPIC_API_KEY`, which would have broken your one hard requirement. There is
a test asserting `--safe-mode` is always in the argv, because losing it would
not break anything visible, it would just quietly cost a hundred times more.

---

## What works, verified end to end

`scripts/voice_smoke.py` renders spoken questions with Piper, pushes the audio
through the microphone seam, and runs Silero, Whisper, the wake gate, Claude and
the voice for real. Last run:

```
spoken aloud: "Vesper, how many CPU cores does this machine have?"
  heard [addressed]: Vesper, how many CPU cores does this machine have?
  tool  Bash: wmic cpu get NumberOfCores,NumberOfLogicalProcessors
  spoke: "Checking now."
  spoke: "Eight physical cores, sixteen logical processors, on a Ryzen 7 8745HX."

spoken aloud: "And how much free space is on the C drive?"      <- no wake word
  spoke: "About five hundred ninety seven gigabytes free on C, out of roughly
          a terabyte total."

spoken aloud: "What was the first thing I asked you?"
  spoke: "How many CPU cores this machine has."

PASS   3 turns, $0.0489
```

Latency, from `scripts/latency.py`: **3.2 seconds from end of speech to first
word** on questions that need a shell command. 1.4s when they do not.

---

## The five things I am most pleased with

**It says "let me look" while it works.** A question needing four commands
produced sixteen seconds of dead silence, which reads as a crash. Now the first
tool call with nothing spoken yet triggers a spoken holding phrase, varied and
never repeated back to back. Same question now speaks at 3.2 seconds.

**It can tell its own voice from yours.** On laptop speakers the mic hears
Vesper and would interrupt itself forever. Instead of echo cancellation, any
interrupting utterance is transcribed and compared against what it just said.
Sixty percent word overlap means it heard itself. Costs nothing, self-corrects,
and utterances under two words are exempt so "yes" and "stop" always land.

**It knows when not to speak.** The ambient loop asks Claude "is this worth
interrupting for?" and gets `SILENT` almost every time. A CPU spike from 12% to
97%: silent. Battery at 8% and unplugged: *"plug in now or you'll lose that
ffmpeg job."* App switches never even cost a turn.

**Long output never gets read aloud.** Claude wraps it in `<screen>` tags,
stripped from the speech stream and printed instead. The splitter works on the
live stream, so a tag arriving split across deltas as `<scr` still routes
correctly.

**It remembers yesterday.** Session id is persisted and resumed, bounded at
twelve hours so the context cannot grow forever.

---

## Three real bugs the tests caught, that I would otherwise have shipped

**`re.match` instead of `re.search`** in the sentence splitter. It anchors at
position 0, so the abbreviation guard never ran for any sentence longer than one
word. "Ask Dr. Lee about it." would have been spoken as two utterances with a
gap in the middle.

**The pre-roll buffer is load-bearing.** The voice test failed with "Vesper, how
many CPU cores" transcribed as "But how many CPU cores". The onset of the name
was being eaten before the endpointer confirmed speech. Without 400ms of
retained history, the wake word simply never fires.

**Quiet hours worked too well.** Sixteen proactive tests failed because it was
2:30am and the loop was correctly refusing to speak. The tests now control the
clock; the feature was right.

---

## Two things on this machine specifically

**Windows Application Control blocks the current wheels.** `ctranslate2 4.8.1`
and `av 18.1.0` both fail with `DLL load failed ... An Application Control
policy has blocked this file`. The real culprit is PyAV's native libraries, which
faster-whisper imports eagerly even though we only ever hand it numpy arrays.
Pinned to `ctranslate2==4.7.2` and `av==17.0.1`, both already trusted here.
Do not float either without confirming a model loads.

**Whisper is on CPU.** Your `torch` is the CPU build, so the RTX 5060 is unused.
`base.en` at 450ms is fast enough that fixing this was not worth the overnight
risk of a cu128 install.

---

## Choices I made without asking

- **Named it Vesper**, with "Jarvis" as an accepted wake word. Change
  `identity.wake_words` in `config.yaml` if you want.
- **Put it at `C:\Vesper`, not on the Desktop.** Your Desktop is redirected into
  OneDrive and OneDrive is running, so anything there syncs to Microsoft. That
  cuts against "nothing leaves this machine". You approved this before you left;
  the Desktop shortcut points at it.
- **Removed "whisper" and "jasper" from the wake word homophones.** They were in
  as known mishearings, but they are ordinary English words and "whisper it to
  me quietly" waking the assistant mid-meeting is worse than missing one wake.
  Biasing Whisper's decoder toward the name made them unnecessary anyway: one
  correct out of three without the bias, three out of three with it.
- **British male voice** (`en_GB-alan-medium`, 63MB, downloaded). Swap
  `voice.model` and run `python -m piper.download_voices <name> --data-dir
  var/voices` for another.

---

## What I did not build

- **No visual overlay.** I had a subagent trace pixelpets as a possible host.
  Its `renderer.js` is 2,890 lines in one global scope with the pet state
  machine inseparable from the window, `sandbox: true` with no mic permission
  handling, and a CSP with no `connect-src`. A voice orb is a clean ~400 line
  job separately; the ~200 lines worth taking from there are the always-on-top
  and click-through knowledge, not the code.
- **No always-running background service.** `run_silent.vbs` exists if you want
  to put a shortcut in your Startup folder, but I did not enable autostart
  without asking.
- **No separate long-term facts store.** Session resume covers continuity within
  a day. A durable "remember that I prefer X" store is the obvious next thing.

---

## State

| | |
|---|---|
| Tests | 296 passing, 14 seconds |
| Coverage | 85% |
| Source | 3,823 lines |
| Tests + scripts | 3,609 lines |
| Self check | all six green |
| Cost of everything I ran tonight | about $1.60 |

Nothing was pushed, published, posted or sent anywhere. There is no git repo
yet; I did not want to make that call for you. `tests/test_privacy.py` parses
every module's AST and fails if anything imports a network library, opens a
socket, hardcodes a URL, or spawns a process other than `claude`.

Your machine was kept awake with `SetThreadExecutionState`, which is a
per-process request Windows drops the moment that process exits. No power
settings were changed.

Read `README.md` for usage and `docs/ARCHITECTURE.md` for why things are the
way they are.
