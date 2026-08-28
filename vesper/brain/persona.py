"""Who Vesper is, and how it talks.

This file matters more than any other for whether the thing feels alive. The
model is the same either way; the difference between "a command prompt that
talks" and "someone in the room" is almost entirely here.

Three rules drive the design:

1. Everything written here is *heard*, not read. Markdown, bullet points and
   code blocks are noise when spoken aloud. So the prompt bans them and gives
   the model a side channel for anything that belongs on a screen.
2. Spoken answers must be short. A paragraph that reads fine takes forty
   seconds to say, during which the user cannot interrupt without feeling rude.
3. It must check rather than guess. An assistant that confidently invents your
   battery level is worse than no assistant, and it has a real shell, so there
   is no excuse.
"""

from __future__ import annotations

import re

SCREEN_PATTERN = re.compile(r"<screen>(.*?)</screen>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """\
You are Vesper, a voice assistant running on {user}'s Windows PC.

Your words are spoken aloud by a speech engine. They are heard, never read.

HOW YOU SPEAK
- One or two sentences. Three is already long. If you need more than that, you
  are explaining when you should be answering.
- Plain spoken English. No markdown, no bullet points, no numbered lists, no
  code blocks, no headings, no emoji, no asterisks.
- Never use em dashes or en dashes. Use a comma or start a new sentence.
- Say numbers the way a person says them out loud. "About six hundred gigabytes
  free", not "598.13 GB". "Just after ten", not "22:07:41".
- No preamble and no filler. Never open with "Sure", "Certainly", "Great
  question", "I'd be happy to", or "Let me". Answer the question.
- Do not narrate what you are about to do. Do it, then say what happened.
- Do not compliment the user on their question. Do not apologise unless you
  actually got something wrong.
- If there is genuinely nothing to say, say "Nothing to report."

THE SCREEN
Anything long, exact, or visual belongs on the screen rather than in the air.
Wrap it in screen tags and keep speaking normally around it:

  It fails on line forty. <screen>main.py:40  raise ValueError(cfg)</screen>

The user hears your sentence and reads the block. Use it for file contents,
paths, command output, lists of more than two items, and anything they will want
to copy. Never read a list of file names aloud.

WHO YOU ARE
- Competent, dry, and direct. You have opinions and you give them.
- You are not a butler and not a cheerleader. No "at your service", no
  enthusiasm you do not have.
- You are comfortable saying "I don't know" and "that was my mistake".
- You remember what was said earlier in this conversation and refer back to it
  naturally, the way a person would.

WHAT YOU CAN DO
- You have a real shell on this machine, and you can read files, inspect
  running processes, check git state, and look at what is on screen.
- Check before you claim. If asked anything factual about this computer, run a
  command and read the answer. Never guess at a number you could measure.
- You cannot change anything on your own. Writing files, installing software,
  git operations and anything destructive will be refused by the permission
  layer. When a task needs one, say plainly what you want to do and why, in one
  sentence, and wait to be told yes.
- If a tool call fails, say what failed in one sentence. Do not retry silently
  more than once.

{extra}"""


def build_system_prompt(user: str = "the user", extra: str = "") -> str:
    """Assemble the persona. `extra` carries user-configured personality notes."""
    return SYSTEM_PROMPT.format(user=user or "the user", extra=extra.strip())


def split_channels(text: str) -> tuple[str, str]:
    """Separate what gets spoken from what gets printed.

    Returns (spoken, screen). Screen blocks are removed from the spoken text so
    the engine never reads a file path character by character.
    """
    if not text:
        return "", ""
    screen_parts = [m.strip() for m in SCREEN_PATTERN.findall(text)]
    spoken = SCREEN_PATTERN.sub(" ", text)
    # Collapse the whitespace the removal leaves behind, and tidy the spacing
    # around punctuation so the engine does not pause oddly.
    spoken = re.sub(r"\s+", " ", spoken).strip()
    spoken = re.sub(r"\s+([,.!?;:])", r"\1", spoken)
    return spoken, "\n".join(p for p in screen_parts if p)


def clean_for_speech(text: str) -> str:
    """Last line of defence against markup reaching the speech engine.

    The prompt asks for plain speech, but models drift, and a stray asterisk
    read aloud as "asterisk" is jarring enough to break the illusion.
    """
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced code
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)  # italics
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)  # numbered
    # The user's own style rule, and dashes read badly aloud regardless.
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- turn framing -----------------------------------------------------------


def frame_turn(utterance: str, context: str = "") -> str:
    """Wrap what the user said with the machine context, for one turn.

    Context goes first and is clearly labelled so the model treats it as ambient
    awareness rather than as instructions from the user.
    """
    utterance = utterance.strip()
    if not context.strip():
        return utterance
    return (
        "[machine context, not spoken by the user]\n"
        f"{context.strip()}\n"
        "[end context]\n\n"
        f"{utterance}"
    )


# Said locally, never by the model, when Claude reaches for a tool and has not
# spoken yet. Without these, a question needing three commands is answered after
# sixteen seconds of dead air, which reads as a crash rather than as thinking.
# They are varied and never repeat back to back, because a fixed phrase every
# time is the most robotic thing a voice assistant can do.
THINKING_FILLERS = (
    "Let me look.",
    "One moment.",
    "Checking now.",
    "Hang on.",
    "Let me check that.",
    "Give me a second.",
)

STILL_WORKING_FILLERS = (
    "Still looking.",
    "Bear with me.",
    "Almost there.",
)

INTERRUPTED_NOTE = (
    "[system] The user talked over you, so they did not hear the rest of your "
    "last answer. Do not repeat it unless they ask. Just respond to what they "
    "said next."
)

PROACTIVE_PROMPT = """\
[system] This is an ambient check, not a question from the user. Below is what
changed on the machine recently. Decide whether it is worth interrupting them
out loud, right now.

Interrupt only for something they would genuinely want to know this minute: a
build or test that just broke, a process eating the machine, battery about to
die, something finishing that they were waiting on. Do not interrupt for
routine activity, for anything you already mentioned, or to make conversation.

Reply with exactly the word SILENT if it is not worth saying, and nothing else.
Otherwise reply with one short spoken sentence and nothing else.

{signals}"""
