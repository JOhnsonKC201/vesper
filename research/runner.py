"""Experiment 1's harness: every task, in every mode, into one report.

Order matters and is not the obvious one. This runs mode by mode across all
tasks, rather than task by task across all modes, so that a machine that drifts
during a long run (a browser that accumulates tabs, a disk that fills) degrades
one arm rather than smearing across all three unevenly. It also means an
interrupted run leaves complete data for the modes that finished instead of a
third of every arm.

Usage:
    python runner.py --dry-run                  wiring check, touches nothing
    python runner.py --modes uia                one arm
    python runner.py --tasks tasks.yaml         the whole comparison
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from hands import loop as loop_module
from hands import observe as observe_module
from hands import trace as trace_module
from hands.guard import Guard
from hands.observe_vision import OmniParser
from hands.planner import ClaudeCliPlanner, ScriptedPlanner
from hands.loop import Outcome
from hands.skills import SkillLibrary

HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_tasks(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", [])
    seen = set()
    for task in tasks:
        task_id = task.get("id")
        if not task_id:
            raise ValueError(f"a task has no id: {task}")
        if task_id in seen:
            raise ValueError(f"duplicate task id {task_id!r}")
        seen.add(task_id)
        if not task.get("check"):
            # Refused rather than defaulted. A task with no check can only ever
            # be scored on the planner's own say-so, and one of those in the
            # set is enough to make the whole success column untrustworthy.
            raise ValueError(f"task {task_id!r} has no check, so it cannot be scored")
    return tasks


def summarise(outcomes: list) -> dict:
    """Per-mode aggregates. Median as well as mean, because one task that
    times out drags a mean of fifteen a long way."""
    by_mode: dict[str, list] = {}
    for outcome in outcomes:
        by_mode.setdefault(outcome.mode, []).append(outcome)

    rows = {}
    for mode, runs in by_mode.items():
        wins = [r for r in runs if r.ok]
        rows[mode] = {
            "tasks": len(runs),
            "solved": len(wins),
            "success_rate": len(wins) / len(runs) if runs else 0.0,
            "mean_steps": statistics.mean([r.steps for r in runs]) if runs else 0,
            "mean_steps_on_wins": statistics.mean([r.steps for r in wins]) if wins else 0,
            "median_wall_s": statistics.median([r.wall_s for r in runs]) if runs else 0,
            "mean_observation_s": statistics.mean([r.observation_s for r in runs]) if runs else 0,
            "mean_planner_s": statistics.mean([r.planner_s for r in runs]) if runs else 0,
            # The gap between what the planner claimed and what the check found.
            "overclaimed": sum(1 for r in runs if r.claimed_ok and not r.ok),
            "blocked": sum(r.blocked for r in runs),
        }
    return rows


def report(outcomes: list, rows: dict, config: dict) -> str:
    lines = [
        "# Experiment 1: observation mode versus task success",
        "",
        f"Run {time.strftime('%Y-%m-%d %H:%M')}. "
        f"Planner {config.get('planner', {}).get('model', 'unknown')}, "
        f"max {config.get('loop', {}).get('max_steps', '?')} steps per task.",
        "",
        "Success is verified against the world by each task's `check`, never by "
        "the planner's own verdict. `overclaimed` counts tasks the planner said "
        "it had finished and the check disagreed.",
        "",
        "## By mode",
        "",
        "| mode | solved | of | success | mean steps | steps (wins) | median wall s | observe s | plan s | overclaimed | blocked |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in observe_module.MODES:
        row = rows.get(mode)
        if not row:
            continue
        lines.append(
            f"| {mode} | {row['solved']} | {row['tasks']} | "
            f"{row['success_rate']:.0%} | {row['mean_steps']:.1f} | "
            f"{row['mean_steps_on_wins']:.1f} | {row['median_wall_s']:.1f} | "
            f"{row['mean_observation_s']:.2f} | {row['mean_planner_s']:.1f} | "
            f"{row['overclaimed']} | {row['blocked']} |"
        )

    lines += ["", "## By task", "",
              "| task | " + " | ".join(observe_module.MODES) + " |",
              "|---|" + "---|" * len(observe_module.MODES)]
    by_task: dict[str, dict] = {}
    for outcome in outcomes:
        by_task.setdefault(outcome.task_id, {})[outcome.mode] = outcome
    for task_id, per_mode in by_task.items():
        cells = []
        for mode in observe_module.MODES:
            got = per_mode.get(mode)
            cells.append("-" if got is None
                         else (f"ok {got.steps}" if got.ok else f"fail {got.steps}"))
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")

    lines += ["", "## Failures", ""]
    for outcome in outcomes:
        if not outcome.ok:
            lines.append(f"- **{outcome.task_id}** ({outcome.mode}): {outcome.detail}")
    if all(o.ok for o in outcomes):
        lines.append("(none)")
    return "\n".join(lines) + "\n"


def report_skills(passes: dict, library, config: dict) -> str:
    """Experiment 2's table: the same tasks, before and after the library.

    Both counts are shown because only one of them is a fair claim. A skill
    turns six actions into one planner decision, so `decisions` falls by
    construction the moment the library is used at all. If `actions` does not
    fall too, the library has not made the agent more efficient, it has only
    moved the bookkeeping.
    """
    without = passes["without"]
    with_lib = passes["with"]
    by_id = {o.task_id: o for o in without}

    lines = [
        "# Experiment 2: does a skill library help on repeat tasks?",
        "",
        f"Run {time.strftime('%Y-%m-%d %H:%M')}. "
        f"Planner {config.get('planner', {}).get('model', 'unknown')}. "
        f"Library holds {len(library)} skill(s) after the first pass.",
        "",
        "Pass 1 starts with an empty library and learns from verified wins only. "
        "Pass 2 repeats the same tasks with the library available. Every task is "
        "reset to its starting state before both passes, or pass 2 would pass its "
        "check on pass 1's leftovers.",
        "",
        "`decisions` is planner calls; `actions` is primitive actions performed. "
        "A skill collapses many actions into one decision, so decisions falling "
        "alone proves nothing.",
        "",
        "## Totals",
        "",
        "| pass | solved | of | success | decisions | actions | median wall s |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, runs in (("without", without), ("with", with_lib)):
        wins = [r for r in runs if r.ok]
        lines.append(
            f"| {label} | {len(wins)} | {len(runs)} | "
            f"{len(wins) / len(runs) if runs else 0:.0%} | "
            f"{sum(r.steps for r in runs)} | {sum(r.actions for r in runs)} | "
            f"{statistics.median([r.wall_s for r in runs]) if runs else 0:.1f} |"
        )

    lines += ["", "## By task", "",
              "| task | without | with | skill used | decisions | actions |",
              "|---|---|---|---|---|---|"]
    for after in with_lib:
        before = by_id.get(after.task_id)
        if before is None:
            continue
        used = ", ".join(after.skills_used) or "-"
        lines.append(
            f"| {after.task_id} | {'ok' if before.ok else 'fail'} | "
            f"{'ok' if after.ok else 'fail'} | {used} | "
            f"{before.steps} to {after.steps} | {before.actions} to {after.actions} |"
        )

    lines += ["", "## What was learned", ""]
    if len(library):
        for skill in library.skills.values():
            reliability = skill.reliability
            score = "untried" if reliability is None else f"{reliability:.0%} of {skill.uses}"
            lines.append(f"- `{skill.name}` ({len(skill.steps)} steps, "
                         f"params {skill.params or 'none'}, {score}) "
                         f"from {skill.source_task}")
    else:
        lines.append("(nothing: no task produced a verified multi-action win)")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--tasks", default=str(HERE / "tasks.yaml"))
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"any of {observe_module.MODES}")
    parser.add_argument("--only", nargs="*", default=None, help="task ids")
    parser.add_argument("--dry-run", action="store_true",
                        help="observe and plan, but perform no action")
    parser.add_argument("--scripted", action="store_true",
                        help="replay each task's `script` instead of asking a model")
    parser.add_argument("--experiment", type=int, choices=(1, 2), default=1,
                        help="1 compares observation modes, 2 compares the skill library")
    parser.add_argument("--skills-file", default=None,
                        help="where the library lives (default var/skills.json)")
    parser.add_argument("--pass", dest="which_pass", type=int,
                        choices=(1, 2), default=None,
                        help="run only one of experiment 2's two passes, so the "
                             "operator can put the desktop back between them. "
                             "Pass 1 learns and saves; pass 2 compares against it.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override loop.max_steps for one run")
    parser.add_argument("--keep-skills", action="store_true",
                        help="start experiment 2 from the existing library "
                             "instead of an empty one")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    tasks = load_tasks(Path(args.tasks))
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only)]
        if not tasks:
            print("no tasks matched --only", file=sys.stderr)
            return 1

    modes = args.modes or config.get("experiment", {}).get("modes", list(observe_module.MODES))
    for mode in modes:
        if mode not in observe_module.MODES:
            print(f"unknown mode {mode!r}", file=sys.stderr)
            return 1

    loop_cfg = dict(config.get("loop", {}))
    if args.max_steps is not None:
        loop_cfg["max_steps"] = args.max_steps
    out_dir = HERE / config.get("output", {}).get("dir", "var")
    shots_dir = out_dir / "shots"
    out_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = config.get("vision", {})
    detector = OmniParser(weights=vision_cfg.get("weights"),
                          box_threshold=float(vision_cfg.get("box_threshold", 0.05)))
    if any(m in ("vision", "both") for m in modes) and not detector.available():
        # Said once, loudly, before anything runs. Finding out after a
        # forty-minute sweep that two of three arms saw nothing is the worst
        # possible time to learn it.
        print(f"WARNING: vision detector unavailable ({detector.why_not}).",
              file=sys.stderr)
        print("         modes 'vision' and 'both' will observe nothing.",
              file=sys.stderr)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    scratch_root = out_dir / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    def make_planner(task):
        return (ScriptedPlanner(task.get("script", []))
                if args.scripted
                else ClaudeCliPlanner(**config.get("planner", {})))

    def one_pass(label, mode, trace, library, learn):
        runs = []
        for task in tasks:
            print(f"[{label}/{mode}] {task['id']}: {task.get('goal', '')[:52]}",
                  file=sys.stderr)
            outcome = loop_module.run_task(
                task, mode, make_planner(task), guard=Guard(), trace=trace,
                detector=detector, shots_dir=shots_dir,
                max_steps=int(loop_cfg.get("max_steps", 20)),
                max_elements=int(loop_cfg.get("max_elements", 200)),
                max_depth=int(loop_cfg.get("max_depth", 12)),
                dry_run=args.dry_run,
                settle_s=float(loop_cfg.get("settle_s", 0.4)),
                library=library, learn=learn, scratch_root=scratch_root,
            )
            runs.append(outcome)
            print(f"    {'ok' if outcome.ok else 'fail'} in {outcome.steps} "
                  f"decisions / {outcome.actions} actions"
                  + (f", learned {outcome.learned}" if outcome.learned else "")
                  + (f", used {outcome.skills_used}" if outcome.skills_used else ""),
                  file=sys.stderr)
        return runs

    if args.experiment == 2:
        # One observation mode throughout. Experiment 2 asks whether the
        # library helps; varying the observation at the same time would leave
        # any difference unattributable to either.
        mode = modes[0]
        skills_path = Path(args.skills_file or (out_dir / "skills.json"))
        library = SkillLibrary(skills_path)
        if args.keep_skills:
            library.load()
        elif skills_path.exists():
            # Started empty by default, because a library carried over from an
            # earlier run makes pass 1 something other than the without arm.
            skills_path.unlink()

        # Pass 1's outcomes go to disk so pass 2 can be a separate invocation.
        # That exists so the operator can put the desktop back between the two
        # (close the app a screen-reading check would otherwise pass on), which
        # is hygiene the harness should not be doing by killing processes.
        pass1_path = out_dir / "pass1.json"

        with trace_module.Trace(out_dir / f"trace-{run_id}.jsonl", run_id=run_id) as trace:
            if args.which_pass == 2:
                if not pass1_path.exists():
                    print(f"no {pass1_path}; run --pass 1 first", file=sys.stderr)
                    return 1
                library.load()
                without = [Outcome(**row) for row in
                           json.loads(pass1_path.read_text(encoding="utf-8"))]
                with_lib = one_pass("with", mode, trace, library, learn=False)
                library.save()
            else:
                without = one_pass("without", mode, trace, library, learn=True)
                library.save()
                pass1_path.write_text(
                    json.dumps([vars(o) for o in without], indent=2, default=str),
                    encoding="utf-8")
                print(f"--- learned {len(library)} skill(s) ---", file=sys.stderr)
                if args.which_pass == 1:
                    for skill in library.skills.values():
                        print(f"    {skill.name}: {len(skill.steps)} steps, "
                              f"params {skill.params}", file=sys.stderr)
                    print(f"pass 1 saved to {pass1_path}. Put the desktop back, "
                          f"then rerun with --pass 2.", file=sys.stderr)
                    return 0
                with_lib = one_pass("with", mode, trace, library, learn=False)
                library.save()

        text = report_skills({"without": without, "with": with_lib}, library, config)
        destination = out_dir / f"report-skills-{run_id}.md"
        destination.write_text(text, encoding="utf-8")
        print(text)
        print(f"report:  {destination}", file=sys.stderr)
        print(f"skills:  {skills_path}", file=sys.stderr)
        return 0

    outcomes = []
    with trace_module.Trace(out_dir / f"trace-{run_id}.jsonl", run_id=run_id) as trace:
        for mode in modes:
            for task in tasks:
                planner = (ScriptedPlanner(task.get("script", []))
                           if args.scripted
                           else ClaudeCliPlanner(**config.get("planner", {})))
                # A fresh guard per task, so the blocked count is per task and
                # a refusal in one does not colour the next one's numbers.
                guard = Guard()
                print(f"[{mode}] {task['id']}: {task.get('goal', '')[:60]}",
                      file=sys.stderr)
                outcome = loop_module.run_task(
                    task, mode, planner, guard=guard, trace=trace,
                    detector=detector, shots_dir=shots_dir,
                    max_steps=int(loop_cfg.get("max_steps", 20)),
                    max_elements=int(loop_cfg.get("max_elements", 200)),
                    max_depth=int(loop_cfg.get("max_depth", 12)),
                    dry_run=args.dry_run,
                    settle_s=float(loop_cfg.get("settle_s", 0.4)),
                    # Reset here too. Each task runs once per mode, so
                    # without it modes 2 and 3 pass their checks on the
                    # file mode 1 left behind.
                    scratch_root=scratch_root,
                )
                outcomes.append(outcome)
                print(f"    {'ok' if outcome.ok else 'fail'} "
                      f"in {outcome.steps} steps, {outcome.wall_s:.1f}s",
                      file=sys.stderr)

    rows = summarise(outcomes)
    text = report(outcomes, rows, config)
    destination = out_dir / f"report-{run_id}.md"
    destination.write_text(text, encoding="utf-8")
    print(text)
    print(f"report:  {destination}", file=sys.stderr)
    print(f"trace:   {out_dir / f'trace-{run_id}.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
