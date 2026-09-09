"""Making stdout and stderr accept the characters we actually print.

Windows still defaults its consoles to a legacy code page, and both entry
points here print text that is not ASCII. Each had its own copy of this, with
its own name and its own idea of what to catch, so this is one copy and both
reasons.

`vesper.main` prints Vesper's own replies, which carry curly quotes and accents
from Claude, and on a legacy code page those come out as question marks.

`vesper.tool` prints window titles, which are worse. Real titles on this
machine right now carry U+2733 and U+25D1, and printing them through a
redirected pipe raised UnicodeEncodeError and returned a failure for a command
that had worked. `brain/claude.py` sets PYTHONIOENCODING for its own child, but
`vasper` runs as a grandchild through a .cmd shim, so it settles the question
itself rather than trusting what it inherited.
"""

from __future__ import annotations

import sys


def force_utf8() -> None:
    """Reconfigure stdout and stderr to utf-8. Never raises.

    A stream that cannot be reconfigured is one that was replaced, usually by a
    test capturing output or by a pipe already opened in text mode. Neither is
    a reason to fail to start, and both behave the way they did before.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
