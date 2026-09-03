"""Taking back the last thing you said yes to.

Two failure modes are worse than having no undo at all, and most of these tests
are about them. Claiming to have restored something without restoring it is the
first. Silently restoring the wrong thing, or an older thing, is the second.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from vesper import audit
from vesper.audio.speaker import Speaker
from vesper.brain.protocol import PermissionNeeded, TurnComplete
from vesper.conversation import Conversation, ConversationConfig
from vesper.undo import MAX_SNAPSHOT_BYTES, UndoStore, targets
from vesper.wake import WakeConfig, WakeGate

from conftest import FakeBrain, FakeMic, FakeSTT, FakeVoice, RecordingUI


# --- working out what an action touches -------------------------------------


def test_a_file_tool_names_its_target():
    assert targets("Write", {"file_path": "C:/notes.txt"})[0] == ["C:/notes.txt"]
    assert targets("Edit", {"file_path": "C:/notes.txt"})[0] == ["C:/notes.txt"]
    assert targets("NotebookEdit", {"notebook_path": "C:/a.ipynb"})[0] == ["C:/a.ipynb"]


def test_a_plain_removal_names_its_targets():
    found, _ = targets("Bash", {"command": 'rm -rf "C:/tmp/a.txt" C:/tmp/b.txt'})
    assert found == ["C:/tmp/a.txt", "C:/tmp/b.txt"]


@pytest.mark.parametrize(
    "command",
    [
        "rm $TARGET",
        "rm *.txt",
        "cat a.txt && rm b.txt",
        "echo hi > out.txt",
        "git push",
        "npm install",
    ],
)
def test_anything_it_cannot_read_with_certainty_is_not_guessed_at(command):
    """A wrong guess means undo quietly misses a file, so the bar is certainty.
    Globs, variables and chains all fail that bar."""
    found, reason = targets("Bash", {"command": command})
    assert found == []
    assert reason, "a refusal to guess still owes the user an explanation"


def test_every_unsupported_case_explains_itself():
    """"I can't undo that" is a useful answer. Silence is not."""
    for tool, payload in (("Bash", {"command": "git push"}), ("WebFetch", {})):
        found, reason = targets(tool, payload)
        assert found == [] and reason


# --- keeping and restoring --------------------------------------------------


def test_an_edited_file_goes_back_to_what_it_was(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("the original", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("Write: notes.txt", "Write", {"file_path": str(target)})
    target.write_text("overwritten by the assistant", encoding="utf-8")

    done, said = store.undo_latest()
    assert done
    assert target.read_text(encoding="utf-8") == "the original"
    assert "notes.txt" in said


def test_a_created_file_is_removed_again(tmp_path):
    """Undoing a file that did not exist before means deleting the one that
    does now, not restoring an empty copy of it."""
    target = tmp_path / "new.txt"
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("Write: new.txt", "Write", {"file_path": str(target)})
    target.write_text("created", encoding="utf-8")

    done, _ = store.undo_latest()
    assert done
    assert not target.exists()


def test_a_deleted_file_comes_back(tmp_path):
    """The case undo matters most for."""
    target = tmp_path / "precious.txt"
    target.write_text("a whole afternoon of work", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("Bash: rm", "Bash", {"command": f'rm "{target}"'})
    target.unlink()

    done, _ = store.undo_latest()
    assert done
    assert target.read_text(encoding="utf-8") == "a whole afternoon of work"


def test_undo_only_reaches_the_most_recent_change(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("a", "Write", {"file_path": str(first)})
    first.write_text("one changed", encoding="utf-8")
    store.keep_copy("b", "Write", {"file_path": str(second)})
    second.write_text("two changed", encoding="utf-8")

    store.undo_latest()
    assert second.read_text(encoding="utf-8") == "two"
    assert first.read_text(encoding="utf-8") == "one changed", "reached too far back"

    store.undo_latest()
    assert first.read_text(encoding="utf-8") == "one"


def test_undoing_twice_over_does_not_claim_success(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("original", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")
    store.keep_copy("a", "Write", {"file_path": str(target)})
    target.write_text("changed", encoding="utf-8")

    assert store.undo_latest()[0]
    done, said = store.undo_latest()
    assert not done
    assert "nothing to undo" in said.lower()


def test_an_unreversible_action_says_why(tmp_path):
    store = UndoStore(tmp_path / "undo")
    store.keep_copy("Bash: git push", "Bash", {"command": "git push origin main"})
    done, said = store.undo_latest()
    assert not done
    assert "shell command" in said


def test_a_file_too_large_to_copy_is_reported_not_silently_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("vesper.undo.MAX_SNAPSHOT_BYTES", 8)
    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * 64)
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("Write: big", "Write", {"file_path": str(target)})
    done, said = store.undo_latest()
    assert not done
    assert "too big" in said


def test_old_snapshots_are_pruned_with_their_copies(tmp_path):
    store = UndoStore(tmp_path / "undo", keep=2)
    for index in range(4):
        target = tmp_path / f"f{index}.txt"
        target.write_text(str(index), encoding="utf-8")
        store.keep_copy(f"a{index}", "Write", {"file_path": str(target)})

    kept = list((tmp_path / "undo").glob("*.txt"))
    assert len(kept) == 2, "snapshots accumulate forever"


def test_a_snapshot_failure_never_blocks_an_approved_action(tmp_path):
    """The user has already said yes out loud. An unwritable snapshot directory
    is a reason for undo to be unavailable, not for the action to be refused.

    The earlier version of this test could not fail. It asserted the return was
    not None when no path in keep_copy returns None, and it named a file that
    did not exist, so the copy it was meant to be testing never even ran."""
    source = tmp_path / "x.txt"
    source.write_text("real content", encoding="utf-8")
    store = UndoStore(tmp_path / "undo" / "\0bad")

    snapshot = store.keep_copy("a", "Write", {"file_path": str(source)})

    assert not snapshot.reversible, "claimed a copy it could not have made"
    assert snapshot.reason, "gave no reason undo will be unavailable"
    assert store.undo_latest()[0] is False


# --- through the conversation -----------------------------------------------


def _audio(seconds=1.0, level=0.2):
    return np.full(int(16000 * seconds), level, dtype=np.float32)


def _build(brain, transcripts, tmp_path, *, undo=True):
    voice = FakeVoice(duration=0.0)
    speaker = Speaker(voice)
    ui = RecordingUI()
    conversation = Conversation(
        brain=brain,
        stt=FakeSTT(transcripts),
        speaker=speaker,
        mic=FakeMic(),
        wake=WakeGate(WakeConfig()),
        config=ConversationConfig(
            greet_on_start=False,
            undo_dir=(tmp_path / "undo") if undo else None,
            audit_log=tmp_path / "actions.log",
        ),
        ui=ui,
    )
    return conversation, speaker, ui, voice


class WritingBrain(FakeBrain):
    """Refuses a write, then performs it once approved."""

    def __init__(self, target):
        super().__init__(["I could not write that.", "Done."])
        self.target = Path(target)
        self.turns_asked = 0

    def ask(self, text):
        self.turns_asked += 1
        if self.turns_asked == 1:
            self.asked.append(text)
            yield PermissionNeeded(
                "Write", "d",
                tool_input={"file_path": str(self.target)},
                tool_use_id="toolu_1",
            )
            yield TurnComplete(text=self.replies.pop(0), turns=1)
            return
        self.target.write_text("written by the assistant", encoding="utf-8")
        yield from super().ask(text)


def test_a_spoken_undo_puts_the_file_back(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("what was there before", encoding="utf-8")
    brain = WritingBrain(target)
    conv, speaker, ui, voice = _build(brain, ["Vesper save that", "Vesper yes"], tmp_path)

    conv._on_utterance(_audio())
    conv._on_utterance(_audio())
    assert target.read_text(encoding="utf-8") == "written by the assistant"

    conv.undo_last()
    assert speaker.wait_until_idle(5.0)
    speaker.close()

    assert target.read_text(encoding="utf-8") == "what was there before"
    assert conv.undos == 1
    assert (audit.UNDONE, "Put notes.txt back.") in ui.decisions


def test_undo_is_answered_locally_and_never_reaches_claude(tmp_path):
    """"Put it back" is what you say when something has gone wrong, so it must
    not depend on the network, or on Claude working out what it changed."""
    brain = FakeBrain(["should never be asked"])
    conv, speaker, ui, voice = _build(brain, [], tmp_path)
    conv.hear("undo that")
    assert speaker.wait_until_idle(5.0)
    speaker.close()

    assert brain.asked == [], "undo went to Claude"
    assert voice.lines == ["There's nothing to undo."]


def test_undo_with_copies_switched_off_says_so(tmp_path):
    brain = FakeBrain([])
    conv, speaker, ui, voice = _build(brain, [], tmp_path, undo=False)
    conv.hear("put it back")
    assert speaker.wait_until_idle(5.0)
    speaker.close()

    assert conv.undo_store is None
    assert "not keeping copies" in voice.lines[0]


# --- what the reviews found -------------------------------------------------


def test_a_native_windows_path_survives_parsing(tmp_path):
    r"""The worst bug this project has had. shlex in posix mode treats a
    backslash as an escape, so `del C:\Users\johns\notes.txt` came apart as
    `C:Usersjohnsnotes.txt`. That path does not exist, so the file was recorded
    as one that had never existed, the real file was deleted, and undo then
    reported success while the original was gone for good."""
    target = tmp_path / "notes.txt"
    target.write_text("IRREPLACEABLE", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")

    store.keep_copy("del", "Bash", {"command": f"del {target}"})
    target.unlink()
    done, said = store.undo_latest()

    assert done and target.exists(), "undo claimed success on a file it lost"
    assert target.read_text(encoding="utf-8") == "IRREPLACEABLE"


def test_windows_style_flags_are_not_mistaken_for_files(tmp_path):
    """`del /f /q file` is the idiomatic form. `/f` and `/q` were being treated
    as targets, inflating the count undo reported and, if anything by that name
    existed, deleting it on undo."""
    found, _ = targets("Bash", {"command": 'del /f /q "C:/tmp/notes.txt"'})
    assert found == ["C:/tmp/notes.txt"]


def test_a_filename_after_the_end_of_options_marker_is_kept():
    """`rm -f -- --oddname` uses the standard idiom for a file whose name looks
    like a flag. It was dropped as a flag, so the file was deleted with no
    backup at all."""
    found, _ = targets("Bash", {"command": "rm -f -- --oddname"})
    assert found == ["--oddname"]


def test_two_files_with_the_same_name_both_come_back(tmp_path):
    """The copy was named by timestamp plus basename, so two same-named files
    in one action could share a filename. The first came back holding the
    second one's contents, the second was unrecoverable, and undo announced
    success for both."""
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    (left / "notes.txt").write_text("LEFT", encoding="utf-8")
    (right / "notes.txt").write_text("RIGHT", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")

    store.keep_copy(
        "rm", "Bash",
        {"command": f'rm "{left / "notes.txt"}" "{right / "notes.txt"}"'},
    )
    (left / "notes.txt").unlink()
    (right / "notes.txt").unlink()
    assert store.undo_latest()[0]

    assert (left / "notes.txt").read_text(encoding="utf-8") == "LEFT"
    assert (right / "notes.txt").read_text(encoding="utf-8") == "RIGHT"


def test_a_failed_restore_keeps_the_snapshot_for_another_try(tmp_path):
    """The ledger entry was popped whether or not anything was restored, so one
    transient failure, a locked file or a passing antivirus scan, cost the
    recovery permanently."""
    target = tmp_path / "notes.txt"
    target.write_text("original", encoding="utf-8")
    store = UndoStore(tmp_path / "undo")
    store.keep_copy("a", "Write", {"file_path": str(target)})
    target.write_text("changed", encoding="utf-8")

    # The copy disappears underneath us, exactly as a cleaner or a scanner does.
    for stray in (tmp_path / "undo").glob("*notes.txt"):
        stray.unlink()

    done, said = store.undo_latest()
    assert not done
    assert "missing" in said
    assert store.latest() is not None, "the last chance to undo was thrown away"


def test_a_malformed_ledger_does_not_take_the_assistant_down(tmp_path):
    """Valid json of the wrong shape raised an uncaught AttributeError from a
    comprehension that sat outside the try block. Nothing up the call chain
    catches it, so the whole process died on the next "undo that"."""
    store = UndoStore(tmp_path / "undo")
    store.directory.mkdir(parents=True, exist_ok=True)
    store.ledger.write_text('[{"files": "not a list"}]', encoding="utf-8")

    assert store.undo_latest() == (False, "There's nothing to undo.")
    assert store.keep_copy("a", "Write", {"file_path": str(tmp_path / "x")}) is not None


# --- partial restores, and the ledger surviving a crash ---------------------


def test_a_partial_restore_keeps_the_entry_and_says_what_it_could_not_reach(tmp_path):
    """The failure this replaces reported success and threw away the evidence.

    Two files snapshotted, one copy missing. The old code popped the whole
    entry, which orphaned the surviving copy so no later undo could ever reach
    it, and named `files[0]` in the sentence whether or not that was the file
    that actually came back.
    """
    store = UndoStore(tmp_path / "undo")
    first, second = tmp_path / "notes.txt", tmp_path / "draft.txt"
    first.write_text("original notes", encoding="utf-8")
    second.write_text("original draft", encoding="utf-8")
    # A plain two file removal is the multi file snapshot this code path is for.
    store.keep_copy(
        "delete two files", "Bash", {"command": f"rm {first} {second}"}
    )

    snapshot = store.latest()
    assert len(snapshot.files) == 2
    # The copy of the *first* file goes missing, so the one that survives is
    # the second. This is exactly the case that used to name the wrong file.
    Path(snapshot.files[0].copy).unlink()
    first.write_text("changed", encoding="utf-8")
    second.write_text("changed", encoding="utf-8")

    done, sentence = store.undo_latest()
    assert done is True
    assert "1 of 2" in sentence
    assert second.read_text(encoding="utf-8") == "original draft"
    # The entry stays, so the unrecovered half is still on the list rather than
    # orphaned on disk with nothing pointing at it.
    assert store.latest() is not None


def test_the_ledger_survives_a_crash_midway_through_writing_it(tmp_path):
    """Every other store here writes to a temp file and replaces. This one did
    not, and it is the one whose whole job is being there when something failed."""
    store = UndoStore(tmp_path / "undo")
    target = tmp_path / "notes.txt"
    target.write_text("first", encoding="utf-8")
    store.keep_copy("edit notes", "Write", {"file_path": str(target)})
    assert store.latest() is not None

    # A truncated ledger is what an interrupted write leaves behind.
    store.ledger.write_text('[{"at": 1.0, "action"', encoding="utf-8")
    assert store.latest() is None, "an unreadable ledger must read as empty"

    # And no stray temp file is left lying around by a normal write.
    target.write_text("second", encoding="utf-8")
    store.keep_copy("edit notes again", "Write", {"file_path": str(target)})
    assert list((tmp_path / "undo").glob("*.tmp")) == []
