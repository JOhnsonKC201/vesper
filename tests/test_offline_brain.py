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

from conftest import FakeBrain  # noqa: E402
from tests.test_conversation import build, settle  # noqa: E402


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


# --- saying the true thing when the local brain is the broken part ----------
#
# classify_local shipped as dead code: it was written, tested in isolation, and
# never called. So a local server that was not running produced "That didn't go
# through on Claude's side", which is close to the opposite of the truth, and
# sends you to check a subscription that is fine.


class BreakingBrain(FakeBrain):
    """Fails every turn the way the CLI does when it cannot reach its server."""

    def __init__(self, message, *, as_error_frame=True):
        super().__init__([])
        self.message = message
        self.as_error_frame = as_error_frame

    def ask(self, payload):
        from vesper.brain.protocol import BrainError, TurnComplete

        self.asked.append(payload)
        if self.as_error_frame:
            yield TurnComplete(
                text=self.message, is_error=True, api_error="", cost_usd=0.0,
                duration_ms=1, session_id="fake-session",
            )
        else:
            yield BrainError(message=self.message)


def _breaking(message, *, local, as_error_frame=True):
    conv, _, _, voice, speaker, ui = build()
    brain = BreakingBrain(message, as_error_frame=as_error_frame)
    brain.local = local
    conv.brain = brain
    return conv, voice, speaker, ui


REFUSED = "connect ECONNREFUSED 127.0.0.1:11434"


def test_a_dead_local_server_is_not_blamed_on_claude():
    conv, voice, speaker, _ = _breaking(REFUSED, local=True)
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    spoken = " ".join(voice.lines).lower()
    assert "claude" not in spoken, voice.lines
    assert "offline" in spoken or "this machine" in spoken, voice.lines


def test_the_same_error_online_still_blames_the_right_side():
    """Only a local brain reinterprets a connection error this way."""
    conv, voice, speaker, _ = _breaking(REFUSED, local=False)
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    assert any("Claude" in line for line in voice.lines), voice.lines


def test_a_missing_local_model_says_so():
    conv, voice, speaker, _ = _breaking(
        "model 'vesper-local:3b' not found, try pulling it", local=True
    )
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    spoken = " ".join(voice.lines).lower()
    assert "model" in spoken, voice.lines


def test_an_ordinary_local_failure_keeps_the_ordinary_words():
    """Not every offline failure is the server. Do not over-claim."""
    conv, voice, speaker, _ = _breaking("the tool returned nothing useful", local=True)
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    spoken = " ".join(voice.lines).lower()
    assert "ollama" not in spoken, voice.lines


def test_a_dead_local_server_reported_as_a_brain_error_also_says_so():
    """The CLI can fail before any turn exists, and that arrives differently."""
    conv, voice, speaker, _ = _breaking(REFUSED, local=True, as_error_frame=False)
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    spoken = " ".join(voice.lines).lower()
    assert "claude" not in spoken, voice.lines


def test_a_brain_error_online_is_unchanged():
    conv, voice, speaker, _ = _breaking(
        "[WinError 232] The pipe is being closed", local=False, as_error_frame=False
    )
    conv.respond("what time is it")
    settle(speaker)
    speaker.close()

    assert any("Claude" in line for line in voice.lines), voice.lines
