"""A written record of everything Vesper was allowed to change.

Consent is spoken, which means it is easy to give and impossible to review
later. Nobody remembers on Thursday what they said yes to on Monday, and "what
did it do to my files" is exactly the question you want answered without having
to trust the thing you are asking.

So every request, and what was decided about it, lands here as one line of plain
text. Append only, never read back by Vesper itself, and safe to delete.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

APPROVED = "approved"
DECLINED = "declined"
IGNORED = "ignored"
UNDONE = "undone"
# A tool call that rode in on somebody else's yes. The CLI will not scope a
# Write or an Edit to one path, so an approval for one file is an approval for
# the tool until it is taken back. This is how the ones you were never asked
# about stop being invisible.
UNASKED = "unasked"
# The hands, granted because the user's own words asked for the click or the
# typing ("click on the LinkedIn tab"). Not an approval given to a question,
# because no question was asked; recorded so the log still says why the mouse
# moved.
ASKED_FOR = "asked-for"
# The hands, granted for the whole session after a question that said so. Later
# clicks in the same session are covered by this line, not by new ones, which
# is why the log can be quiet while the mouse is busy.
STANDING = "standing"
# The user took them back out loud. From here every click asks again.
HANDS_OFF = "hands-off"


def record(path: Path | str, decision: str, action: str, *, at: datetime | None = None) -> None:
    """Append one decision to the log. Never raises: a failed write must not
    take down a conversation, and an unwritable log is not a reason to refuse
    an action the user already approved out loud."""
    line = "{when}\t{decision}\t{action}\n".format(
        when=(at or datetime.now()).isoformat(timespec="seconds"),
        decision=decision,
        action=" ".join(str(action).split()),
    )
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except (OSError, ValueError):
        # ValueError covers a path the OS will not even parse, such as one
        # carrying a null byte. Both mean the same thing here: no log, and the
        # conversation carries on regardless.
        pass
