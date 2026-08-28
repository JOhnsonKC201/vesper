"""Cross-restart memory.

Claude keeps session state server side, so remembering yesterday is just a
matter of persisting one id and passing --resume. The care here is all in not
losing it, and in not resuming something so old that the context is a liability.
"""

import json
import time

import pytest

from vesper.brain.session_store import SessionStore, StoredSession


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session.json")


def test_nothing_stored_yet_returns_none(store):
    assert store.load() is None


def test_a_saved_session_comes_back(store):
    store.save("abc-123", turns=4, cost_usd=0.05)
    loaded = store.load()
    assert loaded.session_id == "abc-123"
    assert loaded.turns == 4
    assert loaded.cost_usd == pytest.approx(0.05)


def test_saving_replaces_the_previous_session(store):
    store.save("first")
    store.save("second")
    assert store.load().session_id == "second"


def test_an_empty_session_id_is_not_saved(store):
    store.save("")
    assert store.load() is None


def test_a_stale_session_is_not_resumed(tmp_path):
    """A resumed session carries its whole history, so old ones are dropped
    rather than dragged along forever."""
    store = SessionStore(tmp_path / "session.json", max_age_hours=12)
    store.save("old-one")
    assert store.load(now=time.time() + 13 * 3600) is None


def test_a_recent_session_is_resumed(tmp_path):
    store = SessionStore(tmp_path / "session.json", max_age_hours=12)
    store.save("recent")
    assert store.load(now=time.time() + 3600).session_id == "recent"


def test_age_is_reported_in_hours(store):
    store.save("x")
    assert store.load().age_hours(now=time.time() + 7200) == pytest.approx(2.0, abs=0.1)


def test_corrupt_file_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert SessionStore(path).load() is None


def test_a_file_missing_the_session_id_is_ignored(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"turns": 3}), encoding="utf-8")
    assert SessionStore(path).load() is None


def test_a_file_with_the_wrong_types_is_ignored(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps({"session_id": "x", "updated_at": "yesterday"}), encoding="utf-8"
    )
    assert SessionStore(path).load() is None


def test_a_json_list_is_ignored(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert SessionStore(path).load() is None


def test_saving_creates_the_directory(tmp_path):
    store = SessionStore(tmp_path / "nested" / "deep" / "session.json")
    store.save("abc")
    assert store.load().session_id == "abc"


def test_no_partial_file_is_left_behind(tmp_path):
    """Written to a temp file and replaced, so a crash cannot truncate it."""
    store = SessionStore(tmp_path / "session.json")
    store.save("abc")
    assert list(p.name for p in tmp_path.iterdir()) == ["session.json"]


def test_clear_forgets_the_session(store):
    store.save("abc")
    store.clear()
    assert store.load() is None


def test_clearing_when_empty_is_harmless(store):
    store.clear()


def test_an_unwritable_path_does_not_raise(tmp_path):
    """Remembering is a convenience. It must never be a reason to fail."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    SessionStore(blocker / "session.json").save("abc")


def test_stored_session_age_is_never_negative():
    future = StoredSession(session_id="x", updated_at=time.time() + 10_000)
    assert future.age_hours() == 0.0
