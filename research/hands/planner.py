"""Deciding the next action, held constant across all three modes.

Experiment 1 varies the observation and nothing else. That only holds if the
planner is identical in every arm, which is why it lives behind a protocol and
why the prompt is built from `Observation.render()` rather than from anything
mode-specific. If mode (a) got a prompt that mentioned control types and mode
(b) got one that mentioned pixels, the experiment would be comparing two
prompts and reporting it as a comparison of two observation strategies.

`ScriptedPlanner` is not only for tests. Running a task with a fixed action
list under all three modes is how you check the harness itself: the actions are
identical, so any difference in the result is the harness leaking, not the
planner deciding.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field

# The action space, spelled for the planner. Kept in one string so the prompt
# and the executor cannot drift apart.
ACTION_SPEC = """\
click    {"verb": "click", "eid": <int>}          press the element with that id
type     {"verb": "type", "text": "<str>"}        type into whatever has focus
hotkey   {"verb": "hotkey", "keys": "ctrl+s"}     a key combination
launch   {"verb": "launch", "target": "notepad"}  start an app or open a path
wait     {"verb": "wait", "seconds": <float>}     let the screen settle
use_skill {"verb": "use_skill", "skill": "<name>", "args": {}}  replay a known recipe
done     {"verb": "done", "success": <bool>}      the goal is met, or is not\
"""


@dataclass
class Action:
    verb: str
    eid: int | None = None
    text: str = ""
    keys: str = ""
    target: str = ""
    seconds: float = 0.0
    success: bool = False
    why: str = ""
    skill: str = ""
    skill_args: dict = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.verb == "done"

    def args(self) -> dict:
        if self.verb == "click":
            return {"eid": self.eid}
        if self.verb == "type":
            return {"text": self.text}
        if self.verb == "hotkey":
            return {"keys": self.keys}
        if self.verb == "launch":
            return {"target": self.target}
        if self.verb == "wait":
            return {"seconds": self.seconds}
        if self.verb == "use_skill":
            return {"skill": self.skill, "args": self.skill_args}
        return {}


def parse(blob: str) -> Action:
    """Pull one action out of whatever the model said.

    Tolerant of prose around the json, because every model produces some, and
    strict about the verb, because an unrecognised one must not be executed as
    something adjacent.
    """
    match = re.search(r"\{.*\}", blob or "", re.DOTALL)
    if not match:
        return Action("done", success=False, why=f"no json in reply: {blob[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Action("done", success=False, why=f"unparseable json: {exc}")

    verb = str(data.get("verb", "")).lower().strip()
    if verb not in {"click", "type", "hotkey", "launch", "wait", "use_skill", "done"}:
        return Action("done", success=False, why=f"unknown verb {verb!r}")

    eid = data.get("eid")
    skill_args = data.get("args")
    return Action(
        verb=verb,
        eid=int(eid) if isinstance(eid, (int, float, str)) and str(eid).isdigit() else None,
        text=str(data.get("text", "")),
        keys=str(data.get("keys", "")),
        target=str(data.get("target", "")),
        seconds=float(data.get("seconds", 0) or 0),
        success=bool(data.get("success", False)),
        why=str(data.get("why", "")),
        skill=str(data.get("skill", "")),
        skill_args=skill_args if isinstance(skill_args, dict) else {},
    )


def build_prompt(goal: str, observation, history: list[str],
                 skills: str = "") -> str:
    recent = history[-6:]
    # The skills block is omitted entirely rather than sent as "SKILLS: none"
    # when the library is empty or has no match. The without-library arm of
    # Experiment 2 has to see a prompt identical to the one it would have seen
    # before the library existed, or the comparison measures a prompt change.
    skills_block = f"""
RECIPES YOU ALREADY KNOW (prefer one of these when it fits):
{skills}
""" if skills else ""
    return f"""You are driving a Windows desktop, one action at a time.

GOAL: {goal}
{skills_block}

WINDOW: {observation.window_title or "unknown"}

WHAT IS ON SCREEN ({len(observation.elements)} elements, mode {observation.mode}):
{observation.render() or "(nothing was detected)"}

WHAT YOU HAVE DONE SO FAR:
{chr(10).join(recent) if recent else "(nothing yet)"}

Reply with exactly one JSON object and no other text. Options:
{ACTION_SPEC}

Only use an eid that appears in the list above. If the goal is already met,
reply with done and success true. If you are stuck or the screen shows nothing
you can use, reply with done and success false rather than guessing."""


@dataclass
class ScriptedPlanner:
    """Replays a fixed list of actions. The control for the whole experiment."""

    actions: list = field(default_factory=list)
    calls: int = 0

    def __post_init__(self) -> None:
        # Scripts arrive from yaml as plain dicts. Coerced once here rather
        # than at every use, and through the same `parse` the real planner's
        # output goes through, so a script cannot express an action the model
        # would not have been allowed to ask for.
        self.actions = [
            item if isinstance(item, Action) else parse(json.dumps(item))
            for item in self.actions
        ]

    def next_action(self, goal: str, observation, history: list[str],
                    skills: str = "") -> Action:
        if self.calls >= len(self.actions):
            return Action("done", success=False, why="script exhausted")
        action = self.actions[self.calls]
        self.calls += 1
        return action


@dataclass
class ClaudeCliPlanner:
    """One shot per step, through the Claude Code CLI already on this machine.

    Deliberately stateless. The loop hands over the full observation and a short
    history every step, so the planner has no memory the trace does not also
    have; anything it "knew" that the trace cannot show would make a run
    impossible to reconstruct afterwards.
    """

    executable: str = "claude"
    model: str = "sonnet"
    timeout_s: float = 90.0
    last_error: str = ""

    def next_action(self, goal: str, observation, history: list[str],
                    skills: str = "") -> Action:
        prompt = build_prompt(goal, observation, history, skills)
        argv = [self.executable, "-p", "--model", self.model,
                "--output-format", "text"]
        try:
            done = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.last_error = f"planner timed out after {self.timeout_s}s"
            return Action("done", success=False, why=self.last_error)
        except FileNotFoundError:
            self.last_error = f"{self.executable} is not on PATH"
            return Action("done", success=False, why=self.last_error)

        if done.returncode != 0:
            self.last_error = (done.stderr or "").strip()[:200]
            return Action("done", success=False,
                          why=f"planner exited {done.returncode}: {self.last_error}")
        return parse(done.stdout)
