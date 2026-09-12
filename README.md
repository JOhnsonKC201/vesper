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
run.bat --desktop-icon           put a launcher on the desktop
run.bat --remove-desktop-icon    take it off again
```

Say **"Vesper"** (or "Jarvis") to wake him. He stays awake for 25 seconds, and
every thing you say pushes that back out, so a real back and forth does not
mean repeating the name before every sentence. He also stays awake for 25
seconds after he finishes answering, measured from when his own voice stops
rather than from when the answer was ready, because the microphone is deaf
while he talks. Then he goes back to sleep and needs the name again.

Two things always cost you his name, awake or not: **approving a change to your
machine**, and **ending him**. A bare "yes" inside the window approves nothing,
because a yes said to someone else in the room lands inside that window all the
time, and "quit" and "exit" are ordinary English. Both get "I'll need my name on
that one" and the question stays open for a proper answer. Saying no needs no
name: stopping something should never be the harder half. A question you never
answer expires when he goes back to sleep, rather than standing open waiting for
a stray yes. Say **"go to sleep"** to put him back early, and set
`listening.follow_up_window_s` to 0 if you would rather say his name every
single time.

Say **"be quiet"** to stop it speaking up on its own, and **"unmute"** to undo
that. Say **"undo that"** to put back whatever it last changed. All three are
answered locally, without a round trip to Claude.

## Running from your login

`run.bat --install-autostart` puts one shortcut in your Startup folder pointing
at `run_silent.vbs`, which launches it with no console window. Delete the
shortcut, or run `--uninstall-autostart`, and it stops. `run.bat --check`
reports whether it is installed, so the self check can answer "will this come
back after a reboot".

`run.bat --desktop-icon` puts the other half on your desktop: the same silent
launcher, wearing the same evening star the tray shows, so quitting Vesper for
a call is not a decision you have to think about. Two details in it are not
obvious. The icon goes where Windows actually draws the desktop, which with
OneDrive's desktop backup turned on is `~/OneDrive/Desktop` and not `~/Desktop`,
and the old folder is usually still there to be written to by mistake. And
clicking it while Vesper is already running puts a message on screen rather
than only printing one, because started this way there is no console for a
printed line to land in, and a double click that does nothing at all is a
double click you make twice.

Once there is no console, three things that used to be free stop being free.

**Seeing it.** The tray icon is the evening star, which is what Vesper means,
drawn in code rather than shipped as an asset because Pillow is not installed
and would be a new dependency to draw one 16 pixel mark. It is a warm star
asleep, a cool one while he is awake and acting on plain speech, and drained of
colour when paused, so a glance answers the question. Double click
it, or pick Dashboard from the menu, for a panel showing uptime, turns, session
cost, what it last heard, and every change it has made to your machine.

**Changing the voice.** The same panel lists every voice this machine can
speak in and switches between them while it is running. Each Piper model
appears once per character, since the character is half the sound, and every
voice Windows has is there too. Preview speaks a sample line in the one you are
pointing at before you commit to it, and the pick outlasts a restart. With
`voice.engine: elevenlabs` the list is the eighteen usable cloud voices
instead, with the month's remaining allowance drawn underneath.

**Stopping it.** The tray menu has pause, open log and quit. Or say "Vesper,
shut down" or "Vesper, goodbye", handled locally without a round trip to
Claude, because you say them precisely when something has gone wrong. Not "go
to sleep", which used to end it and now means what it says.
`taskkill` works too, now that SIGTERM and SIGBREAK are handled.

All three reach the same exit, which matters more than it sounds. Before this,
killing it from Task Manager skipped the exit path: the session id was never
written, so the conversation was silently lost, and the `claude` child was left
running. Measured after the change: clean exit in 1.0s, zero orphaned processes.

**Seeing failures.** Diagnostics now also append to `var/vesper.log`, rotating
at 2MB. Without it, a hidden process with an unplugged microphone fails in
complete silence. The brain's own lines are in there too, tagged `BRAIN`: when
the `claude` child was spawned, when it stopped, and whatever it wrote to
stderr. A turn the CLI reports as failed lands as an `ERROR` line carrying the
CLI's text, and is spoken as one plain sentence, never as that text.

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

## How long before it can hear you

Not the same question as how fast it answers, and for a while the answer was
worse. `start()` built five things in a row and opened the microphone last, so
Vesper was deaf for the whole of starting up, which is also the moment you are
most likely to say something, having just started it.

The microphone opens first now, on its own, and Whisper and the voiceprint warm
on threads behind it. Measured with the real config, real microphone, real
models and a real `claude` child, on a warm cache:

| | |
|---|---|
| `start()` blocked for, before | 2.54s |
| `start()` blocked for, after | 0.73s |

Warm numbers, and the cold ones are worse: the pieces measured one at a time
come to 4.5s, and the log has a cold start where Whisper alone took 7.37s
against the 1.2s it takes once the model file is in the page cache.

`Listener.load()` takes a lock and re-checks inside it, which is what makes
warming in the background safe. Without that, the warming thread and the first
utterance can both find the model unset and both build one, and two copies of
`small.en` is how a card with room for one runs out of memory a second after
starting up.

A model that will not build no longer stops Vesper starting. It used to raise
straight out of `start()`.

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

Out of the box, only your conversation with Claude, through the `claude`
process, exactly as it would from any terminal.

Everything else is local. Speech recognition is Whisper on your CPU. The voice
is a Piper model on disk. The sensors read Windows APIs. The wake word is
matched in memory. There is no telemetry and no account anywhere but the one
you already have.

**There is exactly one setting that changes this, and it is off by default.**
Setting `voice.engine` to `elevenlabs` sends the text of Vesper's replies to
ElevenLabs so they come back as speech. Those replies can quote the contents of
your files, because Vesper can read the whole drive, so this is a real change
rather than a technicality. Microphone audio never leaves under any setting,
and putting `engine` back to `piper` stops all of it immediately.

This is enforced, not just stated. `tests/test_privacy.py` fails the build if
any module in `vesper/` opens a socket, contains a hardcoded URL, or spawns any
process other than `claude`. Exactly one file, `tts/eleven_api.py`, may import
an HTTP client, and a test asserts that list has exactly one name on it, so a
second cannot appear by accident.

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
| Whisper `small.en` transcription | 130ms on the gpu, 1,516ms on the cpu |
| Claude, until something is audible | 2,050ms |
| Piper synthesis | starts immediately, 31x realtime |
| **End of speech to first word** | **2.9s on the gpu, 4.3s on the cpu** |

Transcription later got about 7% cheaper again, by not asking the decoder for
word timings. `word_timestamps=True` makes faster-whisper run a second
alignment pass over every segment, and the only thing it ever produced here was
a list of low-confidence words that nothing read. Same model, same clips, same
decode settings, seven passes over the four wav fixtures, one flag changed:

| | with word timings | without |
|---|---|---|
| gpu, float16 | 344.0ms | 319.8ms |
| cpu, int8 | 1,529.0ms | 1,414.7ms |

Small, and paid on everything the microphone hears rather than only on what was
said to Vesper. One week of a real log holds 975 transcriptions, and 238 of
them were addressed to him.

The third row is the interesting one. Claude's first *text* on those questions
arrives at 4.4s, because it runs the command before saying anything. So when it
reaches for a tool and has not spoken yet, Vesper says "let me look" in its own
voice. That single change took a three-command answer from sixteen seconds of
dead silence to just over three, and it is why the numbers above are 2.0s rather
than 4.4s.

A question needing no tools answers faster: about 1.4s total.

The second row used to say 450ms and `base.en`, from before the model had to be
made bigger to hear the wake word reliably. `small.en` on the cpu costs 1.5s,
which is the single largest thing between you and an answer, and it was being
paid on a machine with an idle graphics card. See the next section.

### Whisper on the gpu

faster-whisper does not use torch. It runs on CTranslate2, which is its own
runtime with its own CUDA build. The device check asked
`torch.cuda.is_available()`, and torch here is deliberately the cpu build,
installed for Silero VAD and the voiceprint model. So the answer was "no gpu" on
every machine, forever, whatever card was in it. CTranslate2 on this machine
reports the card perfectly well.

Measured on an RTX 5060, `small.en`, four spoken questions of about three
seconds each, identical transcripts from both:

| | Median | Min | Max |
|---|---|---|---|
| cpu, int8, 4 threads | 1,516ms | 1,462ms | 1,649ms |
| gpu, float16 | **130ms** | 128ms | 137ms |

Two packages are needed and are not installed by default, because they are
1.3GB and most machines will not use them:

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Nothing else changes. Without them `whisper_device: auto` stays on the cpu
exactly as before, and `run.bat --check` prints which one is in use.

Two details in it were not obvious. Those wheels drop their DLLs in
`site-packages/nvidia/*/bin`, which is on nobody's PATH, and
`os.add_dll_directory` does not help: CTranslate2 loads them from native code
with a plain LoadLibrary, which never sees directories added for Python's own
extension loading. They have to be on PATH before the model is built.

And a device being present is not the same as it working. CTranslate2 reports
one CUDA device on a machine with no CUDA runtime installed at all; the model
then builds without complaint and every transcription afterwards raises
`Library cublas64_12.dll is not found`. Load time success with inference time
failure is the worst available shape, because it turns "the gpu is not set up"
into "Vesper dies the first time you speak to it", at login, with no console for
the traceback to land in. So the check is one real inference, run once at load,
and anything that fails it falls back to the cpu and says so in the log.

### The cloud voice, measured

`voice.engine: elevenlabs` trades latency for a better voice, and the trade is
smaller than it looks. Measured by `scripts/cloud_voice_smoke.py` on a
70 character line:

| | First audio | Charged |
|---|---|---|
| Piper, local | starts immediately | free |
| ElevenLabs, cold | 519ms | 70 characters |
| ElevenLabs, from cache | 0ms | free |

The third row is why this is viable at all. The seventeen lines Vesper repeats,
every holding phrase and every stock reply, are synthesized once and then play
from disk. So the most latency-sensitive line in the whole system, the "one
moment" that covers a tool call, is the one that costs nothing and arrives
instantly. Only novel sentences pay the 519ms.

The free ElevenLabs tier is 10,000 characters a month, which at roughly 200
characters a reply is about 45 replies once the stock phrases are paid for.
`voice.eleven.monthly_characters` caps it at 9,000 deliberately, under the free
allowance rather than on it, so the ceiling is found here rather than by
ElevenLabs. Past the cap Vesper speaks through Piper: the voice changes, nothing
breaks, and no bill arrives.

Two things on the free tier are refused whatever your key's permissions say:
library voices answer 402, and creating a voice answers 403. Eighteen of the
twenty premade voices work, and the dashboard marks the two that do not.

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
with the same three lines in each. `natural` is exactly what it sounded like
before any of this.

Or skip the wav files: the dashboard lists the model at each of the three
characters, and Preview plays them one click apart. Either way the winner is
remembered in `var/voice-choice.json`, which beats `voice.character` in the
config file.

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

### And it keeps corrections past that

The session above expires. "Stop reading me file paths" should not.

When you tell Vesper how you want something done, that instruction is written
to `var/lessons.json` and put in front of Claude at the start of every future
session. The behaviour changes permanently because of something you said once.

```
"Vesper, from now on keep replies under two sentences"
    ->  "I'll remember that."   and the reply is short from then on, forever
```

Say "forget that" to drop the last one, or "forget everything" to clear it.
Both are handled locally and never reach Claude, for the same reason mute and
undo are: the moment you want something forgotten is not the moment to depend
on a network call.

This is not machine learning and calling it that would be a lie. Nothing trains
and no weight moves. It is a durable instruction list with two rules that stop
it going wrong:

**A correction has to be repeated.** "Remember that", "from now on", "always",
"never" and "don't" are unambiguous and count immediately. "No, I meant..." is
a much weaker signal, so it is stored but kept out of the prompt until it has
happened twice. A single "no" in a noisy transcript should not become a
permanent rule.

**The negation has to survive.** "Don't read me file paths" would capture as
"read me file paths" if you strip the trigger word and stop thinking, and the
stored rule would then cause the exact behaviour that was complained about,
with nothing about the stored line looking wrong. There is a test for it.

Twelve lessons reach the prompt, most-repeated first, because a lesson you have
had to give three times is the one least able to afford being buried in a long
prompt. Everything is one readable JSON file you can open and edit.

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

Only a plain yes counts. "Sure, do it, but in front of me" is a yes with a
condition, and Vesper will not act on half an agreement: it keeps the question
open, says it only acts on a plain answer, and when the plain yes comes it
passes your condition along as an instruction about how. A no is a no for that
request, not forever: ask again later and you will be asked again.

**Hands.** It has a mouse and a keyboard, through the same `vasper` command that
gives it eyes, and it uses them in front of you. `vasper look` lists every
control on the front window with its position and is free, like a screenshot.
`vasper click "Search"`, `vasper type "weather baltimore" --enter`, `vasper key
ctrl+l`, `vasper scroll down` and `vasper move` are the hands, and the first one
in a turn is refused and asked about: "I want to use the mouse and keyboard to
click Search. Do I do this for you?" One yes covers all five for that turn and
is handed back when the turn ends. Every click glides the cursor across the
screen first and every character is typed one at a time, so you see what is
about to happen and can say stop. When the click is your own request ("click
on the LinkedIn tab", "type hello in it", "press enter"), there is no second
question: your words are the yes, the hands are granted for that turn and the
actions log records them as asked for. Vesper still asks before anything it
thought of on its own. And a "no" that carries a new instruction ("no, no,
click on the LinkedIn tab") declines the question and then does what you said. Asked to look something up, it will open the
browser and type the search where you can watch rather than call a background
tool. Two things it will not do: click inside a window whose title it hides
(password managers, banking, private browsing), and act on a name that matches
two controls; it names both and asks you to pick.

Three things make that a real gate rather than a polite one:

**The refusal happens in the CLI, not in the prompt.** The brain runs with
`--permission-mode manual` and an allowlist of read-only commands, so the call
is stopped before it happens. A model that decided to ignore its instructions
would still be refused.

**A yes grants only what the question named, for shell commands.** Approving
`git commit` produces `Bash(git commit:*)` and nothing else, so it cannot be
spent on `rm`. A chained command names every verb it will run, including any
hidden inside `$(...)`, and a line with more separate commands than one spoken
sentence can put fairly is refused rather than summarised. Approving an
interpreter hands over the whole interpreter, so the question says "which can do
anything" rather than "run python". A test asserts the property directly: no
verb may be granted that was not spoken.

**For files it does not, and the question now says so.** This paragraph used to
claim the property held everywhere. It does not. Measured against claude
2.1.259: a path scoped `Write(C:/Users/you/notes.txt)` is refused even for the
file it names, through `--allowedTools` and through `--settings` alike, while a
bare `Write` allows writing to a file that was never mentioned to the user. The
CLI offers no finer grain than the tool, so a yes to one file is a yes to
`Write` until it is taken back, and "I want to create notes dot txt" was quietly
a wider promise than it sounded.

Three things now hold that line, since the grant itself cannot be narrowed:

- The question carries the width it actually has. "I want to create notes dot
  txt, **which lets me write to other files too**. Do I do this for you?" This
  is the same move the interpreter case already made, for the same reason.
- Every tool call made while that grant is open is checked against the one that
  was approved. Anything else is counted, written to `var/actions.log` as
  `unasked`, and said out loud at the end of the turn, so a grant spent on
  something you were never asked about stops being invisible.
- The grant is handed back the moment the action finishes, as it always was.

`WebFetch` gained a name for the same reason. It used to ask "fetch a page from
the web", which named nothing at all, on the one action that sends something off
this machine. It now says which host.

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
| `listening.follow_up_window_s` | How long he stays awake after anyone speaks. 0 requires his name on every single utterance. Approving a change and ending him take the name either way. |
| `identity.personality` | Free text appended to the persona. The dial for how it feels. |
| `voice.speed` | Piper's natural pace reads slightly slow for conversation. |
| `voice.choice` | Where the dashboard writes the voice you picked. It beats `model`, `character` and `eleven.voice_id`; delete the file to go back to the config. |
| `voice.character` | `jarvis`, `broadcast`, or `natural` for Piper untouched. Compare with `python scripts/voice_ab.py`. |
| `proactive.enabled` | Turn off the ambient loop entirely. |
| `brain.add_dirs` | What it may read. `C:\` means the whole drive. |
| `brain.allowed_tools` | What it may do without asking. Every entry must be incapable of changing anything: no bare `Bash`, no interpreter. |
| `brain.permission_mode` | `manual` is what makes the gate exist. `auto` is the CLI default and writes under the working directory with no announcement. |
| `consent.window_s` | The backstop on how long a spoken yes still refers to what was asked. In voice mode the awake window is shorter and closes first; this is what governs typed input, which has no microphone loop to notice. |
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
- **The login can expire while it sleeps.** On 2026-09-04 the laptop woke from
  a 21 hour sleep with the CLI's saved login gone, and every turn came back
  "Failed to authenticate: OAuth session expired and could not be refreshed",
  which Vesper read aloud three times as if it were its own answer. Now that
  turn is logged as an error, the brain is respawned once and the question
  asked again, and if the login is still dead Vesper says so in one sentence
  and waits. Run `claude login` in any terminal; the next thing you say gets a
  fresh brain. `claude auth status` is also asked before the brain is spawned
  at all, so a Vesper started onto a dead login says so at once.
- **Whisper uses the gpu when the gpu is usable.** It asked torch about a
  CTranslate2 runtime, which is the wrong library, so it never did. It now asks
  CTranslate2 and proves the answer with a real inference. Without the two
  CUDA packages named above it stays on the cpu, which costs 1.4s per turn.
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
