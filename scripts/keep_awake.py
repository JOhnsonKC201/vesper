"""Hold the machine awake without changing any power settings.

SetThreadExecutionState is a per-process request. Windows reverts it the
instant this process exits, so there is nothing to clean up and nothing
permanent is changed. Contrast with `powercfg /change`, which edits the
user's actual power plan and survives a reboot.
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def hold(poll_seconds: float = 30.0) -> None:
    kernel32 = ctypes.windll.kernel32
    if kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        print("SetThreadExecutionState failed", file=sys.stderr)
        raise SystemExit(1)
    print("awake hold acquired; system sleep suppressed", flush=True)
    try:
        while True:
            # Re-assert periodically. Some drivers and group policies reset the
            # flag out from under a long-lived process.
            kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(poll_seconds)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("awake hold released", flush=True)

if __name__ == "__main__":
    hold()
