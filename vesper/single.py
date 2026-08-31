"""One Vesper at a time.

Autostart makes this a real problem rather than a tidiness one. The shortcut
runs at every login, and starting it by hand is the obvious thing to do when it
seems unresponsive, so two copies is not an unusual accident, it is the expected
one. It happened here within a day.

Two copies is worse than it sounds. Both hold the microphone, so every utterance
is transcribed twice, answered twice and spoken twice, over each other. Both
spend your subscription window. Both write the same log, which is how it was
noticed: every line appearing in pairs, with two different voice scores for one
sentence.

A named mutex rather than a pid file. Windows releases it when the process ends,
however it ends, so there is no stale lock to clean up after a crash and no
window where a recycled pid looks like a running Vesper.
"""

from __future__ import annotations

# Global scope so it is one lock per machine per user session, which matches
# what is actually being protected: the microphone and the speakers.
MUTEX_NAME = "Global\\VesperSingleInstance"

_HANDLE = None


def claim(name: str = MUTEX_NAME) -> bool:
    """Take the lock. False if another Vesper already holds it.

    The handle is kept in a module global on purpose. Letting it be collected
    would release the mutex while the process is still running, which is a
    subtle way of making this function do nothing at all.
    """
    global _HANDLE
    if _HANDLE is not None:
        return True
    try:
        import win32api
        import win32event
        import winerror

        handle = win32event.CreateMutex(None, True, name)
        if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            win32api.CloseHandle(handle)
            return False
        _HANDLE = handle
        return True
    except ImportError:
        # No pywin32 means no lock. Better to run than to refuse to start over
        # a guard that is not available.
        return True
    except Exception:
        return True


def release() -> None:
    global _HANDLE
    if _HANDLE is None:
        return
    try:
        import win32api
        import win32event

        win32event.ReleaseMutex(_HANDLE)
        win32api.CloseHandle(_HANDLE)
    except Exception:
        pass
    finally:
        _HANDLE = None
