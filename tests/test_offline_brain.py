"""The brain without the internet.

The whole design already made this cheap: the brain is a subprocess, not an API
client, so the same `claude` binary speaking the same stream-json protocol can be
pointed at a model server on this machine instead of at Anthropic. Verified by
hand on 2026-09-11 against Ollama 0.23.2, including with every non-localhost
connection black-holed, so what these tests pin is the wiring rather than the
idea.

Two rules from test_privacy.py shape it. No module but the voice and the
transcriber may contain a URL literal, so the host is configured as a host and
the scheme is composed. And nothing may open a socket, so there is no
reachability probe anywhere: the only honest way to know whether Anthropic is
reachable is to have tried, which is what `auth_status` already does.
"""

from vesper.brain import failures
from vesper.brain.claude import BrainConfig, child_env


def _config(**kwargs):
    defaults = dict(
        model="opus",
        local_model="vesper-local:3b",
        local_host="localhost:11434",
        tools=("Bash", "Read", "WebSearch"),
    )
    defaults.update(kwargs)
    return BrainConfig(**defaults)


# --- which provider, and why ------------------------------------------------


def test_cloud_is_the_default_so_nothing_changes_for_anyone_else():
    assert BrainConfig().provider == "auto"
    assert _config().local is False


def test_forcing_local_uses_the_local_model():
    config = _config(local=True)
    assert "vesper-local:3b" in config.argv()
    assert "opus" not in config.argv()


def test_cloud_uses_the_subscription_model():
    config = _config(local=False)
    assert "opus" in config.argv()
    assert "vesper-local:3b" not in config.argv()


def test_a_local_brain_cannot_search_the_web():
    """WebSearch is a round trip to a search engine. Offline it can only fail."""
    argv = _config(local=True).argv()
    tools = argv[argv.index("--tools") + 1]
    assert "WebSearch" not in tools
    assert "Read" in tools


def test_a_cloud_brain_keeps_web_search():
    argv = _config(local=False).argv()
    tools = argv[argv.index("--tools") + 1]
    assert "WebSearch" in tools


def test_the_gate_is_untouched_by_going_local():
    """Offline is not a reason to act without asking."""
    for local in (True, False):
        argv = _config(local=local, permission_mode="manual").argv()
        assert argv[argv.index("--permission-mode") + 1] == "manual"
        assert "--safe-mode" in argv


# --- the environment the child gets -----------------------------------------


def test_local_points_the_cli_at_this_machine():
    env = child_env(_config(local=True))
    assert env["ANTHROPIC_BASE_URL"].endswith("localhost:11434")
    assert env["ANTHROPIC_BASE_URL"].startswith("http")
    # Required by the CLI and ignored by the server, but it must not be empty or
    # the CLI falls back to looking for a real key.
    assert env["ANTHROPIC_AUTH_TOKEN"]


# Assembled rather than written out, the same way test_privacy.py builds its
# fixture key. A literal would be a credential-shaped string in the tree, and CI
# greps the tree for exactly that shape, correctly refusing to care that this one
# is fake.
_FAKE_KEY = "sk" + "-ant-" + "notarealkeyjustatestvalue"


def test_local_clears_any_real_api_key(monkeypatch):
    """A key left in the environment would send the turn to Anthropic after all.

    The single most important test in this file. Offline mode exists so that
    nothing leaves the machine, and a key that outranks the base URL would undo
    the whole thing while every other assertion here still passed.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    env = child_env(_config(local=True))
    assert not env.get("ANTHROPIC_API_KEY")


def test_cloud_adds_nothing_to_the_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    env = child_env(_config(local=False))
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_cloud_does_not_strip_a_key_the_user_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    env = child_env(_config(local=False))
    assert env["ANTHROPIC_API_KEY"] == _FAKE_KEY


def test_vespers_own_secrets_are_still_withheld(monkeypatch):
    """The existing rule, which going local must not quietly undo."""
    monkeypatch.setenv("VESPER_ELEVEN_API_KEY", "secret")
    for local in (True, False):
        assert "VESPER_ELEVEN_API_KEY" not in child_env(_config(local=local))


def test_the_shim_is_still_on_the_path_offline():
    """Vesper's hands have nothing to do with which model is answering."""
    for local in (True, False):
        assert "PATH" in child_env(_config(local=local))


# --- failing over -----------------------------------------------------------


def test_a_local_server_that_is_not_running_is_named_as_such():
    """Otherwise it reads as "Claude is down", and the fix is the opposite."""
    kind = failures.classify_local("connect ECONNREFUSED 127.0.0.1:11434")
    assert kind == failures.NO_LOCAL_SERVER


def test_the_spoken_line_for_a_missing_local_server_says_what_to_do():
    line = failures.spoken_line(failures.NO_LOCAL_SERVER)
    assert line
    assert "—" not in line  # the repo bans em dashes, including in speech


def test_an_unknown_local_model_is_its_own_failure():
    kind = failures.classify_local("model 'vesper-local:3b' not found, try pulling it")
    assert kind == failures.NO_LOCAL_MODEL


def test_an_ordinary_error_is_not_mistaken_for_a_local_one():
    assert failures.classify_local("permission denied writing to the file") is None
    assert failures.classify_local("") is None


# --- falling back in the live loop ------------------------------------------
#
# Driven through the real Conversation, because the decision is not "is the
# login dead" but "given a dead login, does anybody still get an answer".

from tests.test_conversation import build  # noqa: E402


def test_a_dead_login_moves_the_brain_onto_this_machine():
    conv, brain, _, _, speaker, ui = build()
    brain.logged_in = False
    brain.may_fall_back = True
    conv.start_brain()
    speaker.close()

    assert brain.local is True
    assert brain.started is True
    assert conv.locked_out is False
    assert any("on this machine" in w for w in ui.warnings), ui.warnings


def test_a_dead_login_still_locks_out_when_the_user_asked_for_cloud():
    """`provider: cloud` is a choice, and downgrading it silently is not ours."""
    conv, brain, _, _, speaker, ui = build()
    brain.logged_in = False
    brain.may_fall_back = False
    conv.start_brain()
    speaker.close()

    assert brain.local is False
    assert conv.locked_out is True


def test_a_healthy_login_stays_on_the_subscription():
    conv, brain, _, _, speaker, _ = build()
    brain.logged_in = True
    brain.may_fall_back = True
    conv.start_brain()
    speaker.close()

    assert brain.local is False
    assert brain.started is True


def test_falling_back_drops_the_session_thread():
    """A session id belongs to the server that issued it, and only to that one."""
    conv, brain, _, _, speaker, _ = build()
    brain.logged_in = False
    brain.may_fall_back = True
    brain.session_id = "a-cloud-session"
    conv.start_brain()
    speaker.close()

    assert brain.session_id == ""


def test_a_local_server_that_will_not_start_returns_to_the_honest_error():
    """If neither brain is available, say the true thing about the login."""
    conv, brain, _, _, speaker, _ = build()
    brain.logged_in = False
    brain.may_fall_back = True

    def refuse_to_start(**kwargs):
        brain.alive = False

    brain.start = refuse_to_start
    conv.start_brain()
    speaker.close()

    assert conv.locked_out is True
    assert brain.local is False


def test_it_does_not_switch_twice():
    conv, brain, _, _, speaker, _ = build()
    brain.logged_in = False
    brain.may_fall_back = True
    conv.start_brain()
    conv.start_brain()
    speaker.close()

    assert brain.local_switches == [True]
