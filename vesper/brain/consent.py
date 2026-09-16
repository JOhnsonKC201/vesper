"""Asking before doing something big, and hearing the answer.

Vesper can read everything on this machine without asking. Changing something is
different, and the gate for it lives here.

The enforcement is not in this file and not in the prompt. It is the CLI's own
permission layer: the brain runs with `--permission-mode manual` and an
allowlist of read-only commands, so anything that writes, deletes, installs or
pushes is refused before it happens and reported back as a `permission_denied`
frame. This module turns that frame into a sentence a person can answer out
loud, decides what a yes actually grants, and keeps the grant as narrow as the
action that earned it.

Why the narrowness matters. An early version allowed `Bash` outright for reads,
which was worthless: asked to write a file with the Write tool denied, Claude
wrote it with `printf > file` instead and reported success. A gate that lists
bare `Bash` is decoration. Every grant here is scoped to the exact tool, and for
shell commands to the exact verb, so approving "commit this" cannot also approve
"delete that".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- what a yes and a no sound like -----------------------------------------

# Spoken, not typed. These are what Whisper returns for someone answering out
# loud, including the many answers that are not the word yes.
_YES = {
    "yes", "yeah", "yep", "yup", "ya", "yes please", "yes go ahead", "go ahead",
    "do it", "please do", "sure", "okay", "ok", "alright", "all right", "fine",
    "affirmative", "confirmed", "granted", "permission granted", "go for it",
    "yes do it", "sounds good", "of course", "absolutely", "please go ahead",
}
_NO = {
    "no", "nope", "nah", "no thanks", "no thank you", "dont", "do not",
    "stop", "cancel", "leave it", "leave it alone", "not now", "never mind",
    "nevermind", "forget it", "negative", "denied", "skip it", "hold off",
    "not yet", "no dont", "no do not",
}

# Leading noise that carries no decision, so "um, yes" and "well, no" still land.
_FILLER = {"um", "uh", "er", "well", "so", "hmm", "hm", "right", "then",
           "please", "just", "actually", "really", "ok", "okay"}

_DECISIVE_NO = {"no", "nope", "nah", "negative", "dont", "stop", "cancel"}
_DECISIVE_YES = {"yes", "yeah", "yep", "yup", "ya", "sure", "affirmative"}

# The whole vocabulary a genuine yes is allowed to contain.
#
# A word count is not enough. "Yeah, sure, that's fine, I'll call you back",
# said into a phone with Vesper's question still open, was read as consent and
# would have run the action. Every false positive found had the same shape: a
# yes token at the front of a sentence that was about something else. So a yes
# has to be made *entirely* of agreement. "Sure whatever you think" fails on
# "whatever", which is exactly right, because it is not an answer.
_AGREEMENT_WORDS = {
    "yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "alright", "all",
    "right", "fine", "please", "do", "it", "go", "ahead", "for", "of", "course",
    "absolutely", "definitely", "affirmative", "confirmed", "granted",
    "permission", "sounds", "good", "thanks", "thank", "you", "that", "thats",
    "one", "now",
}
# Even made of agreement words, a long utterance is a sentence, not an answer.
_MAX_YES_WORDS = 6

_WORD = re.compile(r"[a-z']+")

YES = "yes"
NO = "no"
UNCLEAR = "unclear"
# A yes with a condition hanging off it: "sure, do it, but in front of me".
# Not consent, because half an agreement must never run the action, and not a
# refusal either. On 2026-09-05 it was treated as NO, the refusal note told
# Claude never to try again, and the rest of the evening was Vesper refusing
# the thing the user had just agreed to. The caller keeps the question open,
# keeps the condition, and asks for a plain answer.
QUALIFIED = "qualified"


def hear_answer(text: str) -> str:
    """Classify a spoken reply to a yes-or-no question.

    Returns YES, NO, QUALIFIED or UNCLEAR. Unclear is the load-bearing one: it
    means the person said something else entirely, and the caller must treat
    that as a new question rather than as consent. Silence and mumbling never
    approve, and neither does a yes that carries a "but".
    """
    if not text:
        return UNCLEAR
    stripped = " ".join(_WORD.findall(text.lower())).replace("'", "")
    if not stripped:
        return UNCLEAR
    if stripped in _YES:
        return YES
    if stripped in _NO:
        return NO

    words = stripped.split()

    # A refusal anywhere in the utterance refuses. Leniency here is the safe
    # direction, and it catches the ordinary forms an earlier version missed:
    # "please don't", "just no" and "actually no" all came back UNCLEAR, which
    # meant they neither declined nor told Claude to drop the idea.
    if any(word in _DECISIVE_NO for word in words):
        return NO
    if "but" in words:
        # "yes but leave the second one" is a conversation, not a green light,
        # and "sure, do it, but in front of me" is a yes the person wants
        # heard. Everything before the "but" has to be agreement for this to be
        # the second kind; otherwise it is a sentence about something else.
        head = words[: words.index("but")]
        agreed = bool(head) and all(w in _AGREEMENT_WORDS or w in _FILLER for w in head)
        # "okay" and "ok" are fillers when they lead into something else and a
        # yes when they stand alone, so the yes check reads the raw head.
        if agreed and any(w in _DECISIVE_YES or w in _YES for w in head):
            return QUALIFIED

    trimmed = list(words)
    while len(trimmed) > 1 and trimmed[0] in _FILLER:
        trimmed.pop(0)
    if not trimmed:
        return UNCLEAR

    for phrase in _NO:
        rest = " ".join(trimmed)
        if rest == phrase or rest.startswith(phrase + " "):
            return NO

    # A yes has to be agreement and nothing else. Both halves matter: it must
    # actually contain a yes, and it must contain nothing that is not one.
    # Checked against the filler-stripped words, so "um, yes" still lands.
    if not any(word in _DECISIVE_YES or word in _YES for word in trimmed):
        return UNCLEAR
    if len(trimmed) > _MAX_YES_WORDS:
        return UNCLEAR
    if all(word in _AGREEMENT_WORDS for word in trimmed):
        return YES
    return UNCLEAR


# --- what was asked for -----------------------------------------------------

# Tools whose whole purpose is to change one named file. Approving one grants
# the tool, because its blast radius is already a single path.
_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# Read a shell line's real verbs. `cd x && git push` is a push, not a cd, and
# the grant has to say so or a yes would hand over more than was agreed.
_CHAIN = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Command substitution hides whole commands inside what looks like an argument.
# `start "" "url$(node -e '...')"` was described out loud as "run start, then
# echo" and granted exactly those, never naming node, so every approval was
# refused again and the question had understated what would run.
_BACKTICKS = re.compile(r"`([^`]*)`")
# Things that run something else. Naming only the wrapper is how `sudo rm -rf x`
# came to be spoken as "run sudo".
_WRAPPERS = {"sudo", "env", "start", "nohup", "time", "xargs"}
# Naming one of these truthfully is the best that can be done, because what it
# runs is a program in another language that cannot be parsed from here.
# Approving one hands over the whole interpreter, so the question says so.
_INTERPRETERS = {
    "python", "python3", "py", "node", "deno", "bun", "perl", "ruby", "sh",
    "bash", "zsh", "powershell", "pwsh", "cmd", "wscript", "cscript",
}
# `vasper` is here so a grant reads `Bash(vasper click:*)` rather than bare
# `Bash(vasper:*)`: approving the hands must not also buy `vasper` verbs that
# do not exist yet.
_SUBCOMMAND_TOOLS = {"git", "npm", "pnpm", "yarn", "pip", "winget", "choco",
                     "docker", "gh", "vasper"}
# The mouse and keyboard, which are granted together. A task is "click the
# address bar, type the search, press enter", and asking three separate
# questions for it made the hands unusable, while the person is sitting there
# watching each one happen. So the question is "use the mouse and keyboard to
# <the first thing>, and keep them for the rest of the session", and a yes
# covers these five verbs until "hands off" or a restart. The user's own request
# ("click on the LinkedIn tab") still buys one turn, not a session. Nothing else
# rides on it: `vasper open` stays free and `rm` stays its own question.
_HAND_VERBS = ("click", "type", "key", "scroll", "move")
_HAND_SPECS = tuple(f"Bash(vasper {verb}:*)" for verb in _HAND_VERBS)


HAND_SPECS = _HAND_SPECS

# What asking for the hands sounds like. "Click on the LinkedIn tab" is the
# request; answering it with "do I click for you?" is friction, and on
# 2026-09-05 01:44 the reply to that question was "no, no, click on the
# LinkedIn tab", whose "no" refused the click. So an utterance that itself
# names a hand action, used as a verb with an object, is the yes for that turn.
# Questions about the hands ("can you click when I ask") and anything carrying
# a refusal word are not requests.
_HAND_REQUEST = re.compile(
    r"\b(?:double[ -]?click|right[ -]?click|click|tap)\s+(?:on|the|that|this|it|at|here|there)\b"
    r"|\b(?:click|tap)\s+[\"']"
    r"|\btype\s+(?:in|into|the|this|that|it|out)\b"
    r"|\btype\s+[\"']"
    r"|\btype\s+\S.*\b(?:in|into)\b"
    r"|\bpress\s+(?:enter|return|escape|esc|tab|space|backspace|delete|ctrl|control|alt|shift|the\s+\S+\s+key)\b"
    r"|\bscroll\s+(?:up|down)\b"
    r"|\b(?:switch|go|move|change)\s+to\s+(?:the\s+)?(?:\S+\s+){0,3}tab\b"
    r"|\bselect\s+(?:the\s+)?(?:\S+\s+){0,3}(?:tab|option|item|button)\b",
    re.IGNORECASE,
)
_NOT_A_HAND_REQUEST = re.compile(
    r"\b(?:don'?t|do not|never|stop|are you able|would you be able|be able to|when i ask|if i ask|"
    r"how do i|how do you|what (?:does|is|type))\b",
    re.IGNORECASE,
)


def asked_for_hands(text: str) -> bool:
    """Does this utterance itself ask for a click, a key or typing?"""
    text = text or ""
    return bool(_HAND_REQUEST.search(text)) and not _NOT_A_HAND_REQUEST.search(text)


def request_after_refusal(text: str) -> str:
    """The instruction riding on a no: "no, no, click on the LinkedIn tab".

    Returns the part after the refusal words when it is a sentence in its own
    right, and "" when the whole thing was a refusal ("no thanks", "no, leave
    it"). The caller declines the pending question first, then treats what is
    left as the next thing the user said, instead of dropping it.
    """
    words = _WORD.findall((text or "").lower())
    while words and (words[0] in _DECISIVE_NO or words[0] in _FILLER or words[0] in {"thanks", "thank", "you"}):
        words.pop(0)
    rest = " ".join(words)
    if len(words) < 3 or hear_answer(rest) == NO:
        return ""
    return rest


def _is_hand(verb: str) -> bool:
    parts = verb.split()
    return len(parts) == 2 and parts[0] == "vasper" and parts[1] in _HAND_VERBS


def _describe_hand(command: str) -> str:
    """The first hand action in a shell line, as a person would say it."""
    for part in _CHAIN.split(command.strip()):
        tokens = part.strip().split()
        if len(tokens) < 2 or tokens[0].split("/")[-1].lower() != "vasper":
            continue
        verb = tokens[1].lower()
        rest = " ".join(t for t in tokens[2:] if not t.startswith("--")).strip("\"'")
        words = " ".join(rest.split()[:6])
        if verb in ("click", "move"):
            what = "at " + words if all(w.lstrip("-").isdigit() for w in rest.split()) and rest else words
            return f"{verb} {what}".strip() if what else verb
        if verb == "type":
            return f"type {words}" if words else "type"
        if verb == "key":
            return "press " + words.replace("+", " ") if words else "press a key"
        if verb == "scroll":
            return f"scroll {words}".strip()
    return "click and type"
# A verb carrying shell punctuation is a parse failure, not a verb. One such
# produced the allowlist spec `Bash(y):*)` from a curl of a url ending in /y.
_CLEAN_VERB = re.compile(r"^[A-Za-z0-9_.+-]+(?: [A-Za-z0-9_.+-]+)?$")
# More separate verbs than this in one line is not something anybody can weigh
# up from a single spoken sentence, so it is refused rather than summarised.
MAX_VERBS = 4


def _verbs(command: str) -> list[str]:
    """Every command verb in a shell line, in order, de-duplicated.

    Returns an empty list when the line cannot be read with confidence. That is
    a refusal, not a default: callers must never treat "no verbs found" as
    permission to grant something broad.
    """
    command = command.strip()
    if not command:
        return []

    # Pull the substituted commands out first, then parse them as commands in
    # their own right, because that is what the shell will do with them.
    outer, inner = _lift_substitutions(command)
    outer = _BACKTICKS.sub(lambda m: (inner.append(m.group(1)) or " "), outer)

    found: list[str] = []
    for source in [outer, *inner]:
        for part in _CHAIN.split(source.strip()):
            tokens = [t for t in part.strip().split() if not _ASSIGNMENT.match(t)]
            found.extend(_verb_of(tokens))

    unique = list(dict.fromkeys(v for v in found if _CLEAN_VERB.match(v)))
    # An unreadable line yields nothing, and the caller refuses. Silently
    # returning a bare Bash grant for it, which is what used to happen, meant
    # `VERSION=$(./build.sh)` was spoken as "run a command" and approved
    # everything for the rest of the turn.
    return unique


def _lift_substitutions(command: str) -> tuple[str, list[str]]:
    """Split `$(...)` groups out of a command, balancing nested parentheses.

    A regex cannot do this. `$(node -e 'console.log(1)')` contains parentheses
    of its own, so a non-greedy or bracket-excluding pattern either stops early
    or refuses to match, and node stays invisible in exactly the case that
    prompted the fix.
    """
    outer: list[str] = []
    inner: list[str] = []
    index = 0
    while index < len(command):
        if command.startswith("$(", index):
            depth, cursor = 1, index + 2
            while cursor < len(command) and depth:
                if command[cursor] == "(":
                    depth += 1
                elif command[cursor] == ")":
                    depth -= 1
                cursor += 1
            if depth:
                # Unbalanced, so the line cannot be read with confidence.
                # Everything from here is dropped and the caller refuses.
                return "", inner
            inner.append(command[index + 2 : cursor - 1])
            outer.append(" ")
            index = cursor
            continue
        outer.append(command[index])
        index += 1
    return "".join(outer), inner


def _verb_of(tokens: list[str]) -> list[str]:
    """The verb, or verbs, one command segment will actually run."""
    while tokens:
        verb = tokens[0].strip("\"'").replace("\\", "/").split("/")[-1].lower()
        if verb in _WRAPPERS and len(tokens) > 1:
            # Name the wrapper and keep going, so the thing it runs is named too.
            return [verb, *_verb_of(tokens[1:])]
        if verb in _SUBCOMMAND_TOOLS and len(tokens) > 1 and not tokens[1].startswith("-"):
            return [f"{verb} {tokens[1]}"]
        return [verb]
    return []


def _speakable_path(path: str) -> str:
    """A path as a person says it: the file name, never the whole path."""
    cleaned = path.replace("\\", "/").rstrip("/")
    name = cleaned.split("/")[-1] or cleaned
    return name.replace("_", " ").replace(".", " dot ")


def _speakable_url(url: str) -> str:
    """A URL as a person says it: the host, never the query string.

    The host is the part that answers "where is this going", which is the only
    question worth asking out loud. Reading a full URL aloud is unlistenable
    and the written form in the audit log keeps every character anyway.
    """
    cleaned = url.split("://", 1)[-1]
    host = cleaned.split("/", 1)[0].split("@")[-1].split(":")[0]
    return (host or url).replace(".", " dot ")


@dataclass(frozen=True)
class ActionRequest:
    """One thing Vesper was stopped from doing, and what approving it costs."""

    tool: str
    tool_input: dict = field(default_factory=dict)
    tool_use_id: str = ""
    message: str = ""

    def spoken(self) -> str:
        """The middle of "I want to ___. Do I do this for you?"."""
        if self.tool in _WRITE_TOOLS:
            path = self.tool_input.get("file_path") or self.tool_input.get("notebook_path") or ""
            verb = "create" if self.tool == "Write" else "edit"
            if not path:
                # No path to name. The grant is the same width either way, so
                # the question has to carry the whole weight of it.
                return f"{verb} a file, which lets me write to any of them"
            return f"{verb} {_speakable_path(path)}, which lets me write to other files too"
        if self.tool == "Bash":
            verbs = _verbs(str(self.tool_input.get("command") or ""))
            if not verbs:
                return "run a command I can't read well enough to describe"
            if len(verbs) > MAX_VERBS:
                return f"run {len(verbs)} separate commands in one line"
            if all(_is_hand(v) for v in verbs):
                # The grant is all five hand verbs for this turn, so the
                # question names the mouse and keyboard, not one verb.
                command = str(self.tool_input.get("command") or "")
                return f"use the mouse and keyboard to {_describe_hand(command)}"
            spoken = " and ".join(verbs) if len(verbs) < 3 else (
                ", then ".join(verbs[:-1]) + ", and " + verbs[-1]
            )
            if any(v.split()[0] in _INTERPRETERS for v in verbs):
                # Approving an interpreter approves whatever it is handed, and
                # that is not visible from here. Say so rather than let "run
                # python" sound like a small thing.
                return f"run {spoken}, which can do anything"
            return f"run {spoken}"
        if self.tool == "WebFetch":
            url = str(self.tool_input.get("url") or "")
            # Naming it matters more here than anywhere: this is the one action
            # that sends something off the machine, and "a page from the web"
            # told you nothing about which page or what it carried.
            if url:
                return f"fetch {_speakable_url(url)} from the web"
            return "fetch a page from the web"
        if self.tool == "WebSearch":
            # Only asked when `brain.free_web` is off. "Use WebSearch" is a
            # tool name read aloud; the search itself is what is being sent.
            query = " ".join(str(self.tool_input.get("query") or "").split())
            if query:
                return f"search the web for {query}"
            return "search the web"
        return f"use {self.tool}"

    def written(self) -> str:
        """The exact version, for the terminal and the audit log."""
        for key in ("command", "file_path", "notebook_path", "url", "pattern"):
            value = self.tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return f"{self.tool}: {value.strip()[:200]}"
        return self.tool

    @property
    def whole_tool(self) -> bool:
        """Does saying yes hand over the tool itself, not one use of it?

        Measured against claude 2.1.259, not assumed. `Bash(git commit:*)` is
        honoured and scopes an approval to one verb. A path scoped
        `Write(C:/Users/you/notes.txt)` is not honoured at all: every form
        tried, through `--allowedTools` and through `--settings`, refused even
        the file it named, while a bare `Write` allowed writing to a file that
        was never mentioned to the user. So for these tools the narrowest grant
        the CLI will accept is the tool itself, and the honest thing is to say
        so in the question rather than let "edit notes dot txt" sound like the
        approval stops at notes dot txt.
        """
        return self.tool != "Bash"

    @property
    def is_hands(self) -> bool:
        """Is this the mouse and keyboard, granted together for the turn?

        The approval note differs: "do exactly this one action" is right for a
        commit and wrong for the hands, where the approved thing is the task
        and it takes several clicks and keystrokes to do.
        """
        return self.grants() == _HAND_SPECS

    def grants(self) -> tuple[str, ...]:
        """The narrowest allowlist specs that let exactly this through.

        Deliberately not "whatever it asks for next". A yes to `git commit`
        returns `Bash(git commit:*)`, so that approval cannot be spent on `rm`.

        For everything else the CLI offers no finer grain than the tool, which
        `whole_tool` exists to make sayable. What keeps that honest is not this
        function: it is the question naming the wider grant, `revoke_soon()`
        taking it straight back, and `Conversation` logging any other file the
        grant is spent on while it is open.
        """
        if self.tool == "Bash":
            verbs = _verbs(str(self.tool_input.get("command") or ""))
            # Nothing, deliberately. An empty grant means the action cannot be
            # approved, which is the only safe answer when the command could
            # not be described. It must never fall back to bare `Bash`: that
            # turns the vaguest possible question into the widest possible
            # permission.
            if not verbs or len(verbs) > MAX_VERBS:
                return ()
            if any(_is_hand(v) for v in verbs) and not all(_is_hand(v) for v in verbs):
                # A hand action chained with something else is two different
                # questions in one line. Refused rather than described, the
                # same as a line with too many verbs.
                return ()
            if all(_is_hand(v) for v in verbs):
                return _HAND_SPECS
            return tuple(f"Bash({verb}:*)" for verb in verbs)
        return (self.tool,)

    @property
    def approvable(self) -> bool:
        """Can this be put to the user as a yes or no question at all?

        The invariant this exists to hold: whatever `grants()` returns must be
        covered by what `spoken()` said. An earlier version truncated the spoken
        list to three verbs while granting all of them, so a yes to "git add,
        then git commit, then git push" also handed over `curl` and `bash`
        without ever naming them.
        """
        return bool(self.grants())

    @property
    def key(self) -> str:
        """Identity for de-duplication within a turn: the action, not the call.

        It used to be the tool_use id. On 2026-09-08 Claude ran `vasper key
        ctrl+t`, was refused, and ran it again in the same turn; two ids, one
        command, and the user was asked about it "and one more thing after it"
        while the twin was logged as declined. Same words, same question.
        """
        return self.written()
