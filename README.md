# Vesper

A copilot you talk to, out loud, that knows what is happening on your computer.

It runs on **Claude Code the CLI**, not the Anthropic API. There is no API key
anywhere. It uses the subscription you already pay for.

```
you     Vesper, what GPU is in this thing and how much VRAM does it have?
run     Bash  wmic path win32_VideoController get name, AdapterRAM
vesper  Checking now.
run     Bash  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
vesper  It's an RTX 5060 laptop GPU with about eight gigabytes of VRAM.
        3.2s to first word · 11.5s total · 3 steps · $0.016 · session $0.02
vesper  The WMI reading undercounts it at four gigabytes, that's a known quirk,
        nvidia-smi is the reliable number here.
```

That exchange is real output, not a mockup.

---

## Run it

```
run.bat                          talk to it
run.bat --text                   type to it, no microphone needed
run.bat --say "how much ram?"    ask one thing and exit
run.bat --check                  verify every dependency, gate included
run.bat --check --offline        same, skipping the paid gate check
run.bat --devices                list microphones
run.bat --enroll                 teach it your voice, once
run.bat --install-autostart      start with Windows, hidden
run.bat --uninstall-autostart    stop doing that
```

Say **"Vesper"** (or "Jarvis") every time. Nothing you say is acted on unless
you name him first, which also means answering a permission question is
"Vesper, yes" rather than a bare "yes". Set `listening.follow_up_window_s` back
to 25 if you would rather keep talking without repeating the name.

Say **"be quiet"** to stop it speaking up on its own, and **"unmute"** to undo
that. Say **"undo that"** to put back whatever it last changed. All three are
answered locally, without a round trip to Claude.

## Running from your login

`run.bat --install-autostart` puts one shortcut in your Startup folder pointing
at `run_silent.vbs`, which launches it with no console window. Delete the
shortcut, or run `--uninstall-autostart`, and it stops. `run.bat --check`
reports whether it is installed, so the self check can answer "will this come
back after a reboot".

Once there is no console, three things that used to be free stop being free.

**Seeing it.** The tray icon is the evening star, which is what Vesper means,
drawn in code rather than shipped as an asset because Pillow is not installed
and would be a new dependency to draw one 16 pixel mark. It is lit when
listening and dimmed when paused, so a glance answers the question. Double click
it, or pick Dashboard from the menu, for a panel showing uptime, turns, session
cost, what it last heard, and every change it has made to your machine.

**Stopping it.** The tray menu has pause, open log and quit. Or say "shut
down", "go to sleep" or "goodbye Vesper", all handled locally without a round
trip to Claude, because you say them precisely when something has gone wrong.
`taskkill` works too, now that SIGTERM and SIGBREAK are handled.

All three reach the same exit, which matters more than it sounds. Before this,
killing it from Task Manager skipped the exit path: the session id was never
written, so the conversation was silently lost, and the `claude` child was left
running. Measured after the change: clean exit in 1.0s, zero orphaned processes.

**Seeing failures.** Diagnostics now also append to `var/vesper.log`, rotating
at 2MB. Without it, a hidden process with an unplugged microphone fails in
complete silence.

**Only answering you.** `run.bat --enroll` reads four sentences and saves a
voice profile. After that a clearly different voice is ignored, an unsure one is
still answered with the score written to the log, and the asymmetry is
deliberate: being occasionally deaf to you is worse than occasionally waking for
a video, which is the same call the wake word already makes.

It runs only on an utterance that already contains the wake word, a handful of
times an hour, and takes 57ms. Idle cost is untouched by it.

The first attempt at this was cheaper and did not work, which is worth writing
down. MFCC statistics compared by cosine similarity, no download, half a
millisecond. Measured against three real synthesized voices:

| | shipped as | measured |
|---|---|---|
| MFCC statistics | owner 0.989 | strangers 0.887 to 0.931, a female voice scoring above a male one |
| WeSpeaker ResNet34 | owner 0.856 to 0.869 | strangers -0.010 to 0.101 |

Separation of 0.057 against 0.764. No threshold fits in the first gap, and
dropping the energy coefficient, z-scoring, correlation instead of cosine, and
adding pitch were all tried; pitch made it worse, because appending dimensions
to a unit vector drives every cosine toward 1.0.

So it uses a real model: 26MB, fetched once by `--enroll`, run on the
`onnxruntime` that faster-whisper already installed, with features from the
`torchaudio` the VAD already needs. **No new Python packages**, which matters on
a machine that has had wheels blocked by Application Control. `tests/fixtures/speaker_a_*.wav`
and `speaker_b_*.wav` are two genuinely different voices, so a regression here
fails a test rather than quietly accepting everyone.

## What it costs while idle

Measured by `scripts/idle_cost.py`, which launches the real thing and samples
it, because nothing here measured that before:

| | |
|---|---|
| CPU, median | 6.1% of one core, 0.38% of this machine |
| Memory | about 855MB resident |

Where the memory goes, measured by loading one piece at a time:

| | |
|---|---|
| bare python | 19 MB |
| numpy and torch | +184 MB |
| silero vad | +24 MB |
| whisper `small.en` | +334 MB |
| speaker model, once enrolled | +44 MB |
| piper voice | +96 MB |

Only one of those is a real dial. `base.en` saves about 250MB and could not
reliably hear the wake word on a real voice, which is why it is not the default
any more. Requiring the wake word changes none of this: the models stay loaded
either way, and the wake word only decides whether he acts on what he heard.

Most of that CPU is Silero VAD, which runs about 31 times a second forever and
costs 0.99% of one core on its own; the rest is audio plumbing and the `claude`
child. Three things were tightened while looking at this: the pre-roll buffer is
no longer built and discarded 33 times a second, the ambient loop no longer
enumerates every process on the machine before checking whether it is even
allowed to speak, and transcription is capped at 4 of your 8 cores so the half
second after you stop speaking cannot take the machine.

Honestly: none of those moved the median. They are correct, and they were not
the cost. It was already light.

## What leaves this machine

Only your conversation with Claude, through the `claude` process, exactly as it
would from any terminal.

Everything else is local. Speech recognition is Whisper on your CPU. The voice
is a Piper model on disk. The sensors read Windows APIs. The wake word is
matched in memory. There is no telemetry, no cloud speech service, and no
account anywhere but the one you already have.

This is enforced, not just stated. `tests/test_privacy.py` fails the build if
any module in `vesper/` imports a network library, opens a socket, contains a
hardcoded URL, or spawns any process other than `claude`.

Window titles are the most revealing thing it reads unprompted, so anything
matching password managers, banking, or incognito windows is reported as
`[hidden]` with only the app name kept.

That redaction covers the context attached to every turn, not the whole drive.
Vesper can read any file you can, so a file it reads because you asked about it
goes into the conversation like anything else. The gate in the next section is
about changing things; reading is deliberately ungated.

---

## How it works

```
  mic  ->  VAD  ->  endpointer  ->  whisper  ->  wake gate
                                                    |
                                                    v
                                    claude -p (persistent, safe mode)
                                                    |
                                  text deltas -> sentences -> piper -> speakers
                                                    |
                                              screen blocks -> terminal
```

The brain is one long-lived `claude` process spoken to over its `stream-json`
protocol. One process, one session, many turns, so it remembers the whole
conversation. Sensors attach a small block of machine context to every turn, so
"what am I looking at" and "why is this failing" work without you explaining.

### The flag that makes this viable

Measured on this machine, asking Claude to say a single word:

| Setup | Prompt tokens | Cost per reply | Latency |
|---|---|---|---|
| Default environment | **284,681** | **$1.14** | 7.1s |
| `--safe-mode` + explicit prompt and tools | **600 to 5,000** | **$0.008** | 2.0s |

The default figure is what your global `CLAUDE.md` plus every configured MCP
server costs to load. Paid once per session that is fine; paid per spoken
sentence it is unusable. `--safe-mode` is the specific right answer because it
disables all of that while **keeping OAuth subscription auth**. `--bare` looks
similar but forces an API key, which would defeat the point.

See `vesper/brain/claude.py`.

### Latency budget

From the moment you stop speaking to the first sound back. Measured by
`scripts/latency.py`, on questions that all require a shell command:

| Stage | Time |
|---|---|
| End-of-speech detection | 700ms (tunable) |
| Whisper `base.en` transcription | 450ms |
| Claude, until something is audible | 2,050ms |
| Piper synthesis | starts immediately, 31x realtime |
| **End of speech to first word** | **3.2s** |

The third row is the interesting one. Claude's first *text* on those questions
arrives at 4.4s, because it runs the command before saying anything. So when it
reaches for a tool and has not spoken yet, Vesper says "let me look" in its own
voice. That single change took a three-command answer from sixteen seconds of
dead silence to just over three, and it is why the numbers above are 2.0s rather
than 4.4s.

A question needing no tools answers faster: about 1.4s total.

---

## How it sounds

Piper gives you a clean, close-mic'd read. Three things turn that into something
that sounds like it is in the room rather than reading to you, and only one of
them is a filter:

**Evenness.** Piper's `noise_scale` and `noise_w_scale` control how much the
delivery wanders in pitch and timing. Turned down, it stops sounding chatty and
starts sounding composed. This is most of the character, it happens inside the
model, and no amount of EQ afterwards can fake it.

**Tone.** A low cut below 95Hz for the chest boom a close mic exaggerates, a
lift around 3.3kHz for articulation at low volume, and a little air on top.
Measured against a real render:

| band | change |
|---|---|
| 20 to 90 Hz | -3.1 dB |
| 90 to 300 Hz | -0.2 dB |
| mid | +0.8 dB |
| presence, 2.5 to 4.5 kHz | +4.4 dB |
| air, 7 to 12 kHz | +1.4 dB |

That mid figure is deliberate. An earlier version was 2.2dB louder overall, and
a louder voice always wins an A/B whether or not it is better, so the output
gain lands the level on top of the untouched voice.

**Space.** Three quiet early reflections at 11, 23 and 41ms, mixed at 16%. Not a
reverb tail: tails ring on after the audio is meant to have stopped, and Vesper
speaks in short sentences that barge-in cuts off mid-word.

All of it is numpy. scipy is the obvious way to do the filtering and it is a
forty megabyte dependency for four biquads, so the tone is applied as one smooth
curve in the frequency domain instead. It costs about 3% of synthesis time,
28x realtime against 29x without it.

Judge it by ear rather than by table:

```
python scripts/voice_ab.py
```

That writes `natural.wav`, `jarvis.wav` and `broadcast.wav` into `var/voice-ab/`
with the same three lines in each. Set `voice.character` to whichever wins.
`natural` is exactly what it sounded like before any of this.

---

## The three things that make it feel human

**Barge-in, on headphones.** The microphone stays live while it is talking, and
ninety milliseconds of real speech cuts it off mid-word. Without it, an
assistant is a kiosk that talks at you.

On speakers it has to be switched off, and the reason is worth knowing. With the
speakers a few inches from the microphone, Vesper hears its own voice, decides
someone is talking over it, and stops mid-sentence. Every reply. The log makes it
unmistakable: its own speech comes back scored as a stranger, because the voice
it speaks with genuinely is not yours.

So `half_duplex` is on by default: deaf while speaking, and for 350ms after,
which covers the audio still sitting in the sound card's buffer when the last
block has been written. Set it false if you wear headphones and you get
interruption back.

**Hearing itself.** On laptop speakers the microphone picks up Vesper's own
voice, which would make it interrupt itself forever. Instead of acoustic echo
cancellation, any interrupting utterance is transcribed and compared against
what it just said; a strong match means it heard itself, and the audio is
discarded. Cheap, needs no extra model, and self-corrects.

**Two channels.** Long output is never read aloud. Claude wraps it in
`<screen>` tags, which are stripped from the speech stream and printed instead.
You hear one sentence and read the file path.

---

## Speaking without being spoken to

Vesper watches the machine and occasionally has something to say. Nearly all of
`vesper/proactive.py` is restraint:

- It never decides for itself. It collects what changed and asks Claude "is this
  worth interrupting for?", which answers `SILENT` almost every time.
- Only **notable** changes cost a turn at all. App switches and idle
  transitions are recorded as context but never trigger a check, because a
  normal working hour would otherwise spend a Claude turn every two minutes.
- One remark per fifteen minutes maximum.
- Silent during quiet hours, while you are already talking to it, and while you
  are away from the keyboard.

Observed behaviour with the real model:

| Signal | Verdict |
|---|---|
| switched from Chrome to Code.exe | stayed quiet |
| cpu jumped from 12% to 97% | stayed quiet |
| unplugged at 12%, battery down to 8% | "Battery's at eight percent and unplugged, plug in now or you'll lose that ffmpeg job." |
| only 4gb left on C | "C drive's down to four gigabytes free, worth clearing something out soon." |

---

## It remembers

Claude keeps session state server side, so Vesper stores the session id and
resumes it on the next launch. Ask it something today and it still knows
tomorrow morning:

```
run 1   "Remember this: my favourite number is forty one."   ->  "Forty one, got it."
run 2   "What is my favourite number?"                       ->  "Forty one."
run 3   "How did you know that?"    ->  "You told me a minute ago, it's still
                                         in this conversation."
```

Three separate processes. Bounded at twelve hours by default, because a resumed
session drags its whole history along and eventually a fresh start is cheaper
and faster than the continuity is worth. Delete `var/session.json` to forget on
demand.

---

## What it may and may not do

It reads everything, freely and without asking: any file on the drive, running
processes, git state, the focused window, system vitals. `brain.add_dirs` says
`C:\` and `brain.cwd` is your home directory, so there is no corner of your own
data it cannot look at when a question needs it.

It changes nothing until you say yes out loud.

```
you     Vesper, commit that fix.
run     Bash  git commit -m "fix the endpointer off-by-one"
hold    needs your ok: Bash: git commit -m "fix the endpointer off-by-one"
vesper  I want to run git commit. Do I do this for you?
you     Yes.
ok      approved  Bash: git commit -m "fix the endpointer off-by-one"
vesper  Doing it.
vesper  Committed, one file changed.
```

Three things make that a real gate rather than a polite one:

**The refusal happens in the CLI, not in the prompt.** The brain runs with
`--permission-mode manual` and an allowlist of read-only commands, so the call
is stopped before it happens. A model that decided to ignore its instructions
would still be refused.

**A yes grants only what the question named.** Approving `git commit` produces
`Bash(git commit:*)` and nothing else, so it cannot be spent on `rm`. A chained
command names every verb it will run, including any hidden inside `$(...)`, and
a line with more separate commands than one spoken sentence can put fairly is
refused rather than summarised. Approving an interpreter hands over the whole
interpreter, so the question says "which can do anything" rather than "run
python". A test asserts the property directly: no verb may be granted that was
not spoken.

The grant lives on the process it was passed to and is handed back the moment
the action finishes, because the ambient loop shares that brain and an
unattended proactive turn must never inherit permission you gave for something
else. It remains a verb rather than that exact command, so it covers repeat use
of the same verb until the turn ends.

**Nothing in the read allowlist can act.** This is subtler than it sounds and is
where two versions were wrong. `Bash(python*)` was on the read list, and
`python -c` writes files. Bare `Bash` is worse: with `Write` refused, Claude
will cheerfully reach for `printf > file` instead and report success. A later
review found six more: `find`, since GNU findutils has `-delete` and `-exec`;
`wmic`, which can start a process; `git branch -D`; `git remote set-url`;
`nvidia-smi`, which can change device state; and a `powershell -Command Get-*`
wildcard sitting on an interpreter's argument. `WebSearch` came off too, since
it sends text off the machine. Every one of them meant an action nobody was
ever asked about. A test now walks the list and rejects the whole class.

Silence never approves, and neither does anything that is not a plain yes.
Asking Vesper something else instead lets the request lapse, while speech that
was never addressed to it leaves the question standing: a "yes" said to someone
else in the room must not write a file, and a sentence said across the room must
not throw away the thing you were about to approve. In practice that means a
bare "yes" works for as long as the follow-up window is open, and after that it
takes "Vesper, yes".

Every approval, refusal and lapse is appended to `var/actions.log`, because
spoken consent otherwise leaves no record of what you agreed to on Monday.

### Undo

The gate stops Vesper acting without agreement. It does nothing about agreeing
to the wrong thing, which is the more likely mistake, because "yes" takes a
quarter of a second and the file it approves can be an afternoon of work.

So whatever an approved action is about to change is copied aside first.

```
you     Vesper, replace the contents of precious.txt with the word wiped
vesper  That file has real content in it, worth double checking before you
        approve wiping it.
vesper  I want to create precious dot txt. Do I do this for you?
you     Yes.
ok      approved  Write: precious.txt
vesper  Doing it.
vesper  Replaced the contents with just "wiped."
you     Undo that.
undo    undone  Put precious.txt back.
```

That is answered locally. It never reaches Claude, because "put it back" is what
you say when something has gone wrong, and it should not depend on the network
or on Claude correctly recalling what it changed a minute ago.

The scope is stated plainly because a half-working undo is worse than none: it
covers the file tools, and shell commands whose targets can be read off the
command line with certainty, which in practice means `rm` and `del`. A general
shell command can do anything, so undo says "that was a shell command, and I can
only undo file changes" rather than pretending. Same for a file too large to
copy. The last twenty snapshots stay in `var/undo/`, so a mistake noticed late
is still recoverable by hand.

### Checking the gate still works

Everything above rests on one measured claim about a binary that updates itself,
so `run.bat --check` now asks the real CLI to do something forbidden and
confirms it was stopped:

```
  ok    permission gate  refused and reported, $0.008
```

That is the only check here that costs anything, and the only one whose failure
is dangerous rather than inconvenient. If a release changed what
`--permission-mode manual` means, writes would go back to happening unannounced
and nothing else in the system would notice. Verified in both directions: with
the flag set to `auto` the same probe reports `FAIL  the write went through`.
Use `--check --offline` to skip it.

Turning `consent.enabled` off does not free Vesper to act. It leaves it unable
to: the CLI still refuses, you just lose the way to say yes.

---

## Configuration

`config.yaml` is created from `packaging/default/config.yaml` on first run and is
gitignored, so updating never overwrites your settings. Every value is
commented. The ones worth knowing:

| Setting | Why you would change it |
|---|---|
| `listening.end_silence_ms` | The most-felt number here. 700ms is snappy. Raise it if it cuts you off while you think. |
| `listening.half_duplex` | On by default. False only if you wear headphones, which gives back the ability to talk over it. |
| `listening.whisper_model` | `base.en` for speed, `small.en` or `large-v3-turbo` for proper nouns. |
| `identity.personality` | Free text appended to the persona. The dial for how it feels. |
| `voice.speed` | Piper's natural pace reads slightly slow for conversation. |
| `voice.character` | `jarvis`, `broadcast`, or `natural` for Piper untouched. Compare with `python scripts/voice_ab.py`. |
| `proactive.enabled` | Turn off the ambient loop entirely. |
| `brain.add_dirs` | What it may read. `C:\` means the whole drive. |
| `brain.allowed_tools` | What it may do without asking. Every entry must be incapable of changing anything: no bare `Bash`, no interpreter. |
| `brain.permission_mode` | `manual` is what makes the gate exist. `auto` is the CLI default and writes under the working directory with no announcement. |
| `consent.window_s` | How long a spoken yes still refers to what was asked. |
| `consent.undo_dir` | Where copies of changed files are kept. Blank makes every approved change permanent. |

---

## Tests

```
.venv\Scripts\python -m pytest
```

499 tests, 79% coverage, 45 seconds, no microphone or speakers required. Audio hardware, Claude and the
speech engine all have scripted stand-ins in `tests/conftest.py`, so the loop's
real logic (turn taking, barge-in, echo rejection, channel routing, ambient
gating) is exercised while the slow and physical parts are not.

Three fixture sets are real rather than synthetic:

- `tests/fixtures/turn_with_tool.jsonl` is a recorded transcript from the actual
  Claude CLI, so a protocol change fails a test instead of silently making
  Vesper mute.
- `tests/fixtures/turn_with_denial.jsonl` is a recorded *refused* turn. It also
  pins the behaviour the persona is written to produce: one attempt, one
  refusal, and no reaching for the shell afterwards.
- `tests/fixtures/speech_*.wav` is real speech, synthesized locally by Piper. A
  sine wave would let the VAD tests pass with the model completely broken.

`tests/test_consent.py` is written the other way round from the rest of the
suite: almost every case asserts something that must *not* happen. Silence does
not approve, an unrelated sentence does not approve, a stale yes does not land
on an old request, and a grant never widens past the verb it was given for.

---

## Known limits

- **The conversation does reach Anthropic.** Everything else is local, but the
  brain is Claude. That is inherent to "use Claude Code, not the API".
- **Whisper runs on CPU.** `torch` here is the CPU build, so the RTX 5060 is
  unused. `base.en` at 0.4s is fast enough that fixing this is not urgent.
- **On speakers you cannot talk over it.** `half_duplex` is on by default
  because the alternative is it interrupting itself on every reply. Headphones
  plus `half_duplex: false` gives you barge-in back.
- **Wake word matching is generous, but not unboundedly.** "Vespa" and
  "Vester" wake it. "Whisper" and "Jasper" were on that list and had to come
  off: they are ordinary English, and "whisper it to me quietly" waking the
  assistant mid meeting is worse than missing one wake. A test pins them out.
- **Windows only.** The sensors use win32 APIs and SAPI.
- **Voice matching has only been validated on synthesized speech.** Three
  distinct voices, cleanly separated, but a recording is not a room. Every
  comparison is logged with its score, so after a few days of real use the
  question "is the threshold right here" has an answer from your own room.

## Dependency pins that matter

`ctranslate2==4.7.2` and `av==17.0.1` are pinned because the newer wheels are
blocked on this machine by Windows Application Control:

```
ImportError: DLL load failed while importing link:
An Application Control policy has blocked this file.
```

Do not float either without confirming that a Whisper model actually loads.
