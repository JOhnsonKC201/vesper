# research/ — desktop control experiments

Experiment 1 from the brief: does a UFO-style control loop do better reading the
UIA tree, a parsed screenshot, or both?

This lives outside the `vesper/` package deliberately. `tests/test_privacy.py`
enforces two invariants on the shipping package that this work would break
wholesale: only `brain/claude.py` may import `subprocess`, and no file may
contain a url. Driving pywinauto and pulling OmniParser weights violates both.
Keeping the experiment at the repo root means the guards stay intact and
whatever survives can be promoted deliberately rather than by accident.

## Running it

```bash
# wiring check: observes and plans, performs nothing
python runner.py --dry-run --scripted

# one arm, real actions
python runner.py --modes uia

# the whole comparison
python runner.py
```

Everything lands in `var/`: a JSONL trace per run and a markdown report with
the tables.

Tests are separate from Vesper's suite (`pytest.ini` only picks up `tests/`):

```bash
python -m pytest research/tests -q
```

## The three modes

| mode | what it reads | strength | blind to |
|---|---|---|---|
| `uia` | UI Automation tree | names, control types, enabled state, ~0.2s | custom-drawn surfaces |
| `vision` | screenshot, OmniParser boxes | anything actually drawn | names, state, what a box is for |
| `both` | UIA, plus vision where UIA saw nothing | UFO2's hybrid detection | pays for both passes |

The fusion in `both` is asymmetric on purpose: UIA wins every overlap above
0.30 IoU, and vision only contributes where UIA has nothing. Merging the other
way would let a box labelled `icon` at 0.31 confidence displace a control that
knows its own name, and `both` would score worse than `uia` for a reason that is
an artifact of the merge rather than a fact about the world.

This is not hypothetical. The first live observation, taken against Windows
Terminal, returned **nine** UIA elements: the tabs, the scrollbar and the close
button. None of the text on screen. That is precisely the gap UFO2 introduced
hybrid detection to close.

## What is real and what is not

**Working and tested:** the element model, the JSONL trace, the destructive
guard, the UIA backend, the three-mode dispatch and fusion, the action space
(click/type/hotkey/launch/wait), the control loop, world-verified checks, and
the runner with its report. 21 tests, all passing.

**Seam only:** the OmniParser detector. The adapter is written and the protocol
is settled, but the weights are ~2GB and are not on this machine. With
`vision.weights: null` the `vision` and `both` arms **observe nothing and say
so**, in a warning before the run starts and in the trace notes. They do not
fall back to something cheaper and score it as though the mode had run.

To make those two arms real:

```bash
pip install ultralytics
huggingface-cli download microsoft/OmniParser-v2.0 --local-dir var/omniparser
# then set vision.weights in config.yaml to var/omniparser/icon_detect/model.pt
```

There is no caption model wired, so a detected box is labelled `icon` and
nothing more. A planner told "the Save button" behaves differently from one told
"icon", and pretending otherwise would flatter mode (b). Adding a captioner is
the next honest step for that arm.

**Not started:** Experiment 3 (voice latency).

## Experiment 2: the skill library

```bash
python runner.py --experiment 2 --modes uia
```

Two passes over the same tasks. Pass 1 starts with an empty library and learns
from wins; pass 2 repeats with the library available. Pass 1 *is* the without
arm, which is why it has to start empty.

**What gates a skill.** Voyager admits a skill when self-verification says the
task worked, because Minecraft gave it no better oracle. Here there is one, so
a skill is only admitted when the task's `check` passed **against the world**.
Letting a model's own verdict decide what enters a library that then teaches
future runs is how a library fills with confident nonsense. A run that claimed
success and failed its check teaches nothing, and there is a test for exactly
that.

**What a skill is.** Not a macro. The obvious approach, recording the winning
actions and replaying them, does not survive contact: a click is stored as a
coordinate and an element id, and both are meaningless next time. Element ids
are per-observation and coordinates move when a window opens 40 pixels lower.
So a stored click carries a *description* of what it pressed and is
re-resolved against a fresh observation at replay time, re-observing before
every click. That costs an observation per click and it is the whole difference
between a skill and a macro.

**Two step counts, not one.** A skill collapses six actions into one planner
decision, so `decisions` falls the moment the library is used at all, purely
because the word changed meaning. The report shows `decisions` and `actions`
side by side. If actions does not fall too, the library has not made the agent
more efficient, it has only moved the bookkeeping. Do not quote the decisions
number on its own.

**Resets are mandatory for this to mean anything.** Pass 1 creates
`np-01.txt`, so pass 2's check would pass the instant it started whether or not
the agent did anything. Tasks that touch the filesystem carry a `reset`, the
runner applies it before both passes, and Experiment 1 gets it too because each
task there runs once per mode. Resets are confined to `var/scratch` and a path
outside it is refused rather than redirected.

### The honest limit

Parameterisation is near-literal. The distiller lifts a value into a parameter
only when it appears verbatim in the goal, which catches paths and quoted
phrases and misses everything a model would generalise properly. Distilling
"Open Notepad, type report and press Save" yields a two-parameter recipe
`(target, text)`, and it does that by string matching, not by understanding.
These are recipes with the obvious arguments pulled out, not the general
programs Voyager writes.

Skill descriptions are generated deterministically rather than by asking a
model, and retrieval is IDF-weighted lexical overlap rather than embeddings.
Both would retrieve better as an LLM call. Both would also make retrieval vary
run to run, sitting directly on top of the thing being measured. That is a
deliberate trade and the right one for an experiment, and the wrong one for a
product.

A skill that is retrieved and keeps failing is **demoted, not dropped**, so it
can still be chosen and seen to fail rather than vanishing and leaving the
failure unexplained.

## Two design decisions worth arguing with

**Success is checked against the world, never self-reported.** Every task
carries a `check` and the runner refuses to load one without it. A loop that
stops when the model says "done, success true" measures the model's confidence,
and WindowsWorld (2604.27776) shows agents are confidently wrong in exactly the
multi-app cases they fail at. The report has an `overclaimed` column for the gap
between what the planner claimed and what the check found; that gap is a
finding, not noise.

**Clicking is by centre point, never by control handle.** A handle only exists
in UIA mode. Letting mode (a) click by handle would compare two different action
layers while reporting it as a comparison of two observations.

## Before you run it for real

`tasks.yaml` holds a 15-task starter set spanning Notepad, Explorer, Chrome,
VS Code and Spotify, with 4 multi-app tasks. **Replace it with yours.** It
exists so the harness is runnable today and so the shape of a task is obvious.
Three tasks are `check: manual` and will never auto-pass; either give them an
objective check or score them by hand.

The guard refuses every destructive action when no confirmer is wired, which is
every unattended run. If a task genuinely needs to delete or send something, it
will be blocked and counted in the `blocked` column rather than silently
skipped.
