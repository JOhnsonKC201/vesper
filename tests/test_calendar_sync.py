"""The calendar download, tested without the network.

The private iCal address is a credential: anyone holding it reads the whole
calendar. So beyond "does it download", these pin that the address only ever
lives in var/calendar_url.txt and in this script's memory, never in a message,
and that the vesper package cannot read that file at all.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECRET = "https://calendar.google.com/calendar/ical/me%40gmail.com/private-abc123SECRET/basic.ics"
ICS = b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n"


def _load():
    spec = importlib.util.spec_from_file_location("calendar_sync", ROOT / "scripts" / "calendar_sync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load()


def _url_file(tmp_path, text=SECRET):
    path = tmp_path / "calendar_url.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_good_download_lands_as_the_file(tmp_path):
    out = tmp_path / "calendar.ics"
    size = sync.sync(_url_file(tmp_path), out, fetcher=lambda url: ICS)
    assert out.read_bytes() == ICS and size == len(ICS)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = _url_file(tmp_path, "# my calendar\n\n" + SECRET + "\n")
    assert sync.read_url(path) == SECRET


def test_webcal_is_the_same_address(tmp_path):
    path = _url_file(tmp_path, SECRET.replace("https://", "webcal://"))
    assert sync.read_url(path) == SECRET


def test_plain_http_is_refused(tmp_path):
    with pytest.raises(sync.SyncError):
        sync.read_url(_url_file(tmp_path, SECRET.replace("https://", "http://")))


def test_a_page_that_is_not_a_calendar_does_not_replace_the_last_good_one(tmp_path):
    out = tmp_path / "calendar.ics"
    out.write_bytes(ICS)
    with pytest.raises(sync.SyncError):
        sync.sync(_url_file(tmp_path), out, fetcher=lambda url: b"<html>sign in</html>")
    assert out.read_bytes() == ICS


def test_an_oversized_download_is_refused(tmp_path):
    with pytest.raises(sync.SyncError):
        sync.validate(b"BEGIN:VCALENDAR" + b"x" * (sync.MAX_BYTES + 1))


def test_no_address_file_says_where_to_put_one(tmp_path):
    with pytest.raises(sync.SyncError, match="calendar_url.txt"):
        sync.read_url(tmp_path / "calendar_url.txt")


def test_messages_name_the_host_never_the_secret_path():
    assert sync.host_of(SECRET) == "calendar.google.com"
    assert "SECRET" not in sync.host_of(SECRET)


def test_a_failed_fetch_does_not_echo_the_address(monkeypatch):
    import urllib.error

    def boom(*args, **kwargs):
        raise urllib.error.HTTPError(SECRET, 404, "Not Found", None, None)

    monkeypatch.setattr(sync.urllib.request, "urlopen", boom)
    with pytest.raises(sync.SyncError) as caught:
        sync.fetch(SECRET)
    assert "SECRET" not in str(caught.value)
    assert "404" in str(caught.value)


def test_the_task_runs_this_script_every_fifteen_minutes():
    argv = sync.install_argv()
    assert argv[:5] == ["schtasks", "/Create", "/F", "/TN", sync.TASK_NAME]
    assert argv[argv.index("/MO") + 1] == "15"
    assert "calendar_sync.py" in argv[argv.index("/TR") + 1]


def test_the_package_never_reads_the_address_file():
    """Only the script may hold the secret. If the package could read it, the
    address could reach Claude through a tool printout."""
    offenders = [
        path.name for path in (ROOT / "vesper").rglob("*.py")
        if "calendar_url" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
