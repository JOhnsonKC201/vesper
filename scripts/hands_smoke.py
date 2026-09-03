"""Prove the hands work, and how fast, without spending a Claude turn.

    python scripts/hands_smoke.py

Everything here is what Vasper itself runs. Nothing is mocked. The last section
is the one that matters: it checks the shim resolves the way the brain will
find it, which is where this broke twice while it was being built.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesper import tool
from vesper.brain.claude import child_env
from vesper.config import BrainSettings

ROOT = Path(__file__).resolve().parent.parent


def timed(label: str, fn):
    started = time.monotonic()
    result = fn()
    print(f"  {label:32s} {(time.monotonic() - started) * 1000:7.1f}ms")
    return result


def main() -> int:
    print("\nlooking\n")
    windows = timed("vasper windows", tool.visible_windows)
    apps = timed("vasper apps", tool.installed_apps)
    print(f"\n  {len(windows)} windows, {len(apps)} installed apps\n")
    for w in windows[:5]:
        print(f"    {w.process:22s} {tool._safe_title(w.title)[:52]}")

    print("\ncapturing\n")
    code = timed("vasper screenshot", lambda: tool.main(["screenshot"]))
    if code != 0:
        print("  screenshot failed")

    print("\nwhat may run without asking\n")
    for entry in sorted(t for t in BrainSettings().allowed_tools if "vasper" in t):
        print(f"    free    {entry}")
    for gated in ("click", "type"):
        print(f"    asks    Bash(vasper {gated}:*)")

    print("\nthe shim, the way the brain finds it\n")
    # Two shims exist because the brain's shell is git-bash, which will not
    # resolve a bare `vasper` to `vasper.cmd`. This section is here because the
    # POSIX one was missing at first, and then ran from the wrong directory.
    for name in ("vasper", "vasper.cmd"):
        path = ROOT / name
        print(f"    {name:12s} {'present' if path.is_file() else 'MISSING'}")

    env = child_env()
    first = env.get("PATH", "").split(os.pathsep)[0]
    print(f"    child PATH[0] {first}")
    print(f"    {'ok' if first == str(ROOT) else 'WRONG: the brain cannot reach the shim'}")

    print("\n  running it from another directory, as the brain does:")
    result = subprocess.run(
        [str(ROOT / "vasper.cmd"), "apps", "chrome"],
        cwd=str(Path.home()), capture_output=True, text=True, timeout=60,
    )
    out = (result.stdout or result.stderr).strip().splitlines()
    print(f"    exit {result.returncode}: {out[0] if out else '(no output)'}")
    print()
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
