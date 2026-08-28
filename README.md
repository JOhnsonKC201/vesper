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
        3.2s to first word · 11.5s total · 3 steps · $0.016
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
run.bat --check                  verify every dependency
run.bat --devices                list microphones
```

Say **"Vesper"** (or "Jarvis") to get its attention. After it answers you have
twenty five seconds to keep talking without repeating the name. Talk over it at
any point and it stops mid-word.

Say **"be quiet"** to stop it speaking up on its own. Say **"unmute"** to
undo that.

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

Window titles are the most revealing thing it reads, so anything matching
password managers, banking, or incognito windows is reported as `[hidden]` with
only the app name kept.

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

## The three things that make it feel human

**Barge-in.** The microphone stays live while it is talking. Ninety milliseconds
of real speech cuts it off mid-word and starts listening. Without this it is a
kiosk that talks at you.

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

It reads freely: files, processes, git state, the focused window, system vitals.
It runs read-only commands without asking.

It cannot change anything on its own. Writing files, installing software and git
operations are absent from the allowlist in `config.yaml`, so they are refused,
and the refusal is surfaced as a spoken request for permission. Widen
`brain.allowed_tools` if you want to loosen that.

---

## Configuration

`config.yaml` is created from `packaging/default/config.yaml` on first run and is
gitignored, so updating never overwrites your settings. Every value is
commented. The ones worth knowing:

| Setting | Why you would change it |
|---|---|
| `listening.end_silence_ms` | The most-felt number here. 700ms is snappy. Raise it if it cuts you off while you think. |
| `listening.half_duplex` | Set true on laptop speakers in a loud room. Loses barge-in, removes any chance of it hearing itself. |
| `listening.whisper_model` | `base.en` for speed, `small.en` or `large-v3-turbo` for proper nouns. |
| `identity.personality` | Free text appended to the persona. The dial for how it feels. |
| `voice.speed` | Piper's natural pace reads slightly slow for conversation. |
| `proactive.enabled` | Turn off the ambient loop entirely. |

---

## Tests

```
.venv\Scripts\python -m pytest
```

296 tests, 85% coverage, 14 seconds, no microphone or speakers required. Audio hardware, Claude and the
speech engine all have scripted stand-ins in `tests/conftest.py`, so the loop's
real logic (turn taking, barge-in, echo rejection, channel routing, ambient
gating) is exercised while the slow and physical parts are not.

Two fixture sets are real rather than synthetic:

- `tests/fixtures/turn_with_tool.jsonl` is a recorded transcript from the actual
  Claude CLI, so a protocol change fails a test instead of silently making
  Vesper mute.
- `tests/fixtures/speech_*.wav` is real speech, synthesized locally by Piper. A
  sine wave would let the VAD tests pass with the model completely broken.

---

## Known limits

- **The conversation does reach Anthropic.** Everything else is local, but the
  brain is Claude. That is inherent to "use Claude Code, not the API".
- **Whisper runs on CPU.** `torch` here is the CPU build, so the RTX 5060 is
  unused. `base.en` at 0.4s is fast enough that fixing this is not urgent.
- **Speakers cause more interruptions than headphones.** Echo rejection catches
  it after the fact; headphones avoid it entirely.
- **Wake word matching is generous.** "Vespa", "Whisper" and "Jasper" all wake
  it, because being slightly deaf is a much worse failure than occasionally
  waking when it should not have.
- **Windows only.** The sensors use win32 APIs and SAPI.

## Dependency pins that matter

`ctranslate2==4.7.2` and `av==17.0.1` are pinned because the newer wheels are
blocked on this machine by Windows Application Control:

```
ImportError: DLL load failed while importing link:
An Application Control policy has blocked this file.
```

Do not float either without confirming that a Whisper model actually loads.
