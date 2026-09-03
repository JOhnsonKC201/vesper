"""Coming back when you open the lid, which login alone never covers.

The counterpart to `autostart.py`: that one covers the machine starting, this
one covers it waking. Nothing here relates to `wake.py`, which is about hearing
your own name.

The Startup folder shortcut runs when the shell session starts, meaning a
reboot or a sign out and back in. It emphatically does not run on resume from
sleep, and a laptop that is closed and opened for days can go a week without a
single login. This one went from Aug 31 to Sep 3 without one, so the shortcut
had no opportunity to fire and Vesper stayed dead from the moment it was killed.

Task Scheduler is the only thing Windows offers that hears a wake. Two triggers
rather than one, because they cover different machines: a session unlock is the
normal path when the lid opens onto a lock screen, and the Power-Troubleshooter
event covers a resume that never locked, where no unlock ever happens.

The action is `run_silent.vbs --if-idle`, and the flag is load bearing. These
triggers fire on every unlock, and Vesper is usually already running, so the
second copy takes the "already running" path in `main.run_voice`. With no
console attached that path puts a message box on screen, which would mean a
dialog every single time you opened your laptop.

Registered over COM rather than by spawning `schtasks`, which matters for two
reasons that are both checked by tests in `test_privacy.py`: this package
spawns nothing but the brain, and it contains no urls. Shelling out would break
the first, and the task xml that `schtasks /xml` wants carries a schema url
that would break the second. `shortcut.py` already reaches for `win32com`, so
this is the established way to ask Windows for something.
"""

from __future__ import annotations

import os
from pathlib import Path

TASK_NAME = "Vesper resume"
LAUNCHER = "run_silent.vbs"

# Task Scheduler's enums, which arrive over COM as bare integers. Named here
# because `Triggers.Create(11)` at the call site is unreadable, and because
# getting one wrong produces a task that registers cleanly and never fires.
_TRIGGER_EVENT = 0
_TRIGGER_SESSION_STATE_CHANGE = 11
_SESSION_UNLOCK = 8
_ACTION_EXEC = 0
_LOGON_INTERACTIVE_TOKEN = 3
_RUNLEVEL_LEAST_PRIVILEGE = 0
_CREATE_OR_UPDATE = 6
_INSTANCES_IGNORE_NEW = 2

# The other half of "come back on wake": a resume that never showed a lock
# screen. Power-Troubleshooter logs exactly one of these per wake.
_RESUME_EVENT = (
    "<QueryList><Query Id='0' Path='System'><Select Path='System'>"
    "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter']"
    " and EventID=1]]"
    "</Select></Query></QueryList>"
)


def _user() -> str:
    """The SAM name, which is what both the principal and the trigger want."""
    try:
        import win32api

        # NameSamCompatible. Asking Windows beats assembling it out of two
        # environment variables that a stripped-down session may not set.
        return win32api.GetUserNameEx(2)
    except Exception:
        domain = os.environ.get("USERDOMAIN") or os.environ.get("COMPUTERNAME") or ""
        name = os.environ.get("USERNAME") or ""
        # os.path.join rather than a literal separator, which keeps this file
        # free of the one character that makes it awkward to write about.
        return os.path.join(domain, name) if domain else name


def _wscript() -> str:
    return str(Path(os.environ.get("WINDIR", "C:/Windows")) / "System32" / "wscript.exe")


def plan(root: Path) -> dict:
    """Everything the task will say, as plain data.

    Split out so the tests can read the shape of it without a real Task
    Scheduler, and so that what gets registered is reviewable in one place
    rather than spread over thirty COM property assignments.
    """
    root = Path(root)
    return {
        "name": TASK_NAME,
        "user": _user(),
        "description": (
            "Starts Vesper when you unlock or wake this machine. The Startup "
            "folder only runs at login, which never happens on a laptop that "
            "sleeps instead of shutting down. Safe to delete: Vesper still "
            "starts at login without it."
        ),
        "command": _wscript(),
        # --if-idle so an unlock while Vesper is already running is a no-op
        # rather than a dialog box.
        "arguments": '"' + str(root / LAUNCHER) + '" --if-idle',
        "working_directory": str(root),
        "triggers": ["session-unlock", "resume-from-sleep"],
        "resume_event": _RESUME_EVENT,
        "settings": {
            # Both false on purpose. The default refuses to start on battery,
            # which is exactly the moment a laptop lid opens.
            "DisallowStartIfOnBatteries": False,
            "StopIfGoingOnBatteries": False,
            # Vesper is meant to run all day, so no time limit.
            "ExecutionTimeLimit": "PT0S",
            "MultipleInstances": _INSTANCES_IGNORE_NEW,
            "RunOnlyIfIdle": False,
            "StartWhenAvailable": False,
            "Hidden": False,
            "Enabled": True,
        },
    }


def _folder():
    """The root task folder, over COM. Raises if Task Scheduler is unreachable."""
    import win32com.client

    scheduler = win32com.client.Dispatch("Schedule.Service")
    scheduler.Connect()
    # A backslash, and only a backslash. A forward slash here is rejected with
    # 0x8007007B, "the filename, directory name, or volume label syntax is
    # incorrect", which does not obviously mean "wrong kind of slash".
    return scheduler, scheduler.GetFolder("\\")


def is_installed() -> bool:
    try:
        _, folder = _folder()
        folder.GetTask(TASK_NAME)
        return True
    except Exception:
        # GetTask raises both for "no such task" and for "no Task Scheduler",
        # and the honest answer to either is the same: it will not come back.
        return False


def install(root: Path) -> tuple[bool, str]:
    """Register the resume task. Returns (ok, something to tell the user).

    Never raises, for the same reason `autostart.install` does not: every
    caller is a CLI flag or a self check, and neither wants a traceback.
    """
    root = Path(root)
    if not (root / LAUNCHER).exists():
        # Same trap as the startup shortcut, one step further from anyone
        # watching: a task pointing at nothing fails silently, at unlock.
        return False, LAUNCHER + " is missing from " + str(root)

    spec = plan(root)
    try:
        scheduler, folder = _folder()
        task = scheduler.NewTask(0)

        task.RegistrationInfo.Author = spec["user"]
        task.RegistrationInfo.Description = spec["description"]

        settings = task.Settings
        for key, value in spec["settings"].items():
            setattr(settings, key, value)
        settings.AllowDemandStart = True

        unlock = task.Triggers.Create(_TRIGGER_SESSION_STATE_CHANGE)
        unlock.StateChange = _SESSION_UNLOCK
        unlock.UserId = spec["user"]
        unlock.Enabled = True

        woke = task.Triggers.Create(_TRIGGER_EVENT)
        woke.Subscription = spec["resume_event"]
        woke.Enabled = True

        action = task.Actions.Create(_ACTION_EXEC)
        action.Path = spec["command"]
        action.Arguments = spec["arguments"]
        action.WorkingDirectory = spec["working_directory"]

        principal = task.Principal
        principal.UserId = spec["user"]
        principal.LogonType = _LOGON_INTERACTIVE_TOKEN
        principal.RunLevel = _RUNLEVEL_LEAST_PRIVILEGE

        folder.RegisterTaskDefinition(
            TASK_NAME, task, _CREATE_OR_UPDATE, None, None,
            _LOGON_INTERACTIVE_TOKEN,
        )
    except ImportError:
        return False, "pywin32 is not available, so the task cannot be created"
    except Exception as exc:
        return False, "Task Scheduler refused it: " + str(exc)
    return True, 'runs on unlock and on wake  (task "' + TASK_NAME + '")'


def uninstall() -> tuple[bool, str]:
    """Remove the resume task. Succeeds if it was already gone."""
    if not is_installed():
        return True, "there was no resume task"
    try:
        _, folder = _folder()
        folder.DeleteTask(TASK_NAME, 0)
    except Exception as exc:
        return False, "could not remove it: " + str(exc)
    return True, 'removed "' + TASK_NAME + '"'


def describe() -> str:
    """One line for the self check."""
    if is_installed():
        return 'comes back on unlock and wake  (task "' + TASK_NAME + '")'
    return "not installed, so it only comes back at login"
