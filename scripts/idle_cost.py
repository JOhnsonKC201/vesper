"""What Vesper costs while nobody is talking to it.

An assistant that runs from login is only welcome if you cannot feel it. Nothing
in this repo measured that until now: `scripts/latency.py` measures how fast it
answers, which is a different question and the easier one.

So this launches the real thing, leaves it alone, and samples what it actually
uses. Real process, real microphone, real models, no stubs, because the whole
point is the steady state you will be living with.

    python scripts/idle_cost.py            60 seconds
    python scripts/idle_cost.py 180        longer, catches a proactive tick

Say nothing while it runs. If you talk to it, you are measuring a conversation
rather than an idle loop, and it will tell you it saw one.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Long enough to average out model loading and the first few seconds of
# scheduler noise, short enough that you will actually run it twice.
DEFAULT_SECONDS = 60.0
SETTLE_SECONDS = 12.0
SAMPLE_EVERY = 0.5


def main() -> int:
    import psutil

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS

    print(f"\nstarting vesper, settling for {SETTLE_SECONDS:.0f}s, then sampling "
          f"for {seconds:.0f}s")
    print("say nothing while this runs\n")

    child = subprocess.Popen(
        [sys.executable, "-m", "vesper.main", "--no-voice"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        process = psutil.Process(child.pid)
        # Everything Vesper spawns counts. The `claude` child sits blocked on
        # stdin, but it is resident memory you are paying for either way.
        time.sleep(SETTLE_SECONDS)
        if child.poll() is not None:
            print("vesper exited during startup:")
            print((child.stderr.read() or "")[:2000])
            return 1

        # Summed cpu_times deltas rather than cpu_percent. Two reasons, both
        # learned by getting a confident 0.0% out of the first version of this
        # script: psutil.children() hands back new objects every call, and
        # cpu_percent on a new object always reports 0.0 because it has nothing
        # to compare against. And the process actually doing the work is a
        # descendant, not the one launched here, so the tree has to be summed.
        def tree_cpu_seconds() -> tuple[float, float]:
            total, rss = 0.0, 0.0
            for target in [process, *process.children(recursive=True)]:
                try:
                    times = target.cpu_times()
                    total += times.user + times.system
                    rss += target.memory_info().rss
                except psutil.Error:
                    continue
            return total, rss / (1024 * 1024)

        cpu_samples: list[float] = []
        rss_samples: list[float] = []
        last_cpu, _ = tree_cpu_seconds()
        last_at = time.monotonic()

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(SAMPLE_EVERY)
            used, rss = tree_cpu_seconds()
            now = time.monotonic()
            elapsed = now - last_at
            if elapsed > 0:
                cpu_samples.append(100.0 * (used - last_cpu) / elapsed)
                rss_samples.append(rss)
            last_cpu, last_at = used, now

        if not cpu_samples:
            print("no samples taken")
            return 1

        cores = psutil.cpu_count(logical=True) or 1
        ordered = sorted(cpu_samples)
        median = ordered[len(ordered) // 2]
        peak = ordered[-1]
        mean = sum(cpu_samples) / len(cpu_samples)

        print(f"  samples          {len(cpu_samples)} over {seconds:.0f}s")
        print(f"  cpu, median      {median:5.1f}% of one core "
              f"({median / cores:5.2f}% of the machine)")
        print(f"  cpu, mean        {mean:5.1f}% of one core")
        print(f"  cpu, peak        {peak:5.1f}% of one core")
        print(f"  memory, median   {sorted(rss_samples)[len(rss_samples) // 2]:6.0f} MB")
        print(f"  memory, peak     {max(rss_samples):6.0f} MB")

        if median == 0.0 and mean == 0.0:
            print("\n  exactly zero is not a plausible idle cost. the microphone")
            print("  probably never opened, so this measured nothing.")
        elif peak > max(median * 8, 20):
            print("\n  that peak looks like a conversation, not an idle loop.")
            print("  run it again without talking.")
        return 0
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


if __name__ == "__main__":
    raise SystemExit(main())
