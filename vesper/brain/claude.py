"""The brain: a long-lived `claude` process Vesper talks to over stream-json.

Why a subprocess rather than the Anthropic API: this runs on the user's Claude
subscription with no API key anywhere, which was the whole requirement.

The flags below are not arbitrary. Measured on this machine, asking Claude to
say one word:

    default environment                             284,681 tokens  $1.14  7.1s
    --safe-mode + explicit --system-prompt/--tools   600-5,000       $0.008 2.0s

The default cost comes from the user's global CLAUDE.md plus the tool schemas of
every configured MCP server. A voice assistant re-pays that on every session, so
--safe-mode is load-bearing, not an optimisation. It is also specifically the
right flag because it keeps normal OAuth subscription auth. (--bare looks
similar but forces ANTHROPIC_API_KEY, which would defeat the point.)
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .protocol import BrainError, Event, StreamParser, TurnComplete, encode_user_message

# Vesper's own secrets, which the child has no use for and every ability to
# read. The ElevenLabs key is the live example: `config.eleven_key()` documents
# the environment as the safer place to put it, "so the key need not sit on
# disk at all", and copying the whole environment into this child undid that.
#
# It matters because of what the child is. Once a Bash or interpreter grant is
# approved, Claude can run arbitrary code, and arbitrary code can read
# os.environ and send it somewhere. That is a channel entirely outside the
# "only eleven_api.py touches the network" rule, because that rule is about
# Vesper's package, not about what Vesper's subprocess is allowed to run.
_PRIVATE_ENV_PREFIX = "VESPER_"


def child_env() -> dict:
    """The environment the brain gets: ours, minus anything that is a secret."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(_PRIVATE_ENV_PREFIX)
    }
    env["PYTHONIOENCODING"] = "utf-8"
    # `vasper` is Vasper's own hands (vesper/tool.py, via vasper.cmd). The brain
    # runs with cwd set to the home directory, so without this the shim is not
    # on any path it can reach and every call comes back "not recognized".
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PATH"] = os.pathsep.join([root, env.get("PATH", "")])
    return env

# Windows: keep the child's console window from flashing on screen.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_SENTINEL = object()


@dataclass
class BrainConfig:
    """Everything that shapes the child process invocation."""

    executable: str = "claude"
    model: str = "sonnet"
    cwd: str = str(Path.home())
    system_prompt: str = "You are Vesper, a helpful voice assistant."
    # Read freely. Anything that changes the machine is absent from this list
    # and therefore prompts, which the protocol surfaces as PermissionNeeded.
    tools: tuple[str, ...] = ("Bash", "Read", "Grep", "Glob", "WebSearch")
    allowed_tools: tuple[str, ...] = ()
    add_dirs: tuple[str, ...] = ()
    turn_timeout_s: float = 180.0
    # `manual` is what makes the allowlist mean anything. Measured against the
    # real CLI: without it the session runs in `auto`, where the allowlist is
    # largely advisory and a Write to a path under cwd goes through unannounced.
    # With it, every call outside the allowlist is refused before it happens and
    # reported as a permission_denied frame. Set to "auto" only if you want a
    # Vesper that acts without asking.
    permission_mode: str = "manual"

    def argv(self, resume_session: str = "", grants: tuple[str, ...] = ()) -> list[str]:
        args = [
            self.executable,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--safe-mode",
            "--exclude-dynamic-system-prompt-sections",
            "--model", self.model,
            "--system-prompt", self.system_prompt,
        ]
        # Never `if self.permission_mode:`. A blank or commented out value in
        # config.yaml dropped the flag entirely, and the session then ran in
        # `auto`, where a Write under cwd goes through unannounced. A typo
        # silently turning off the one thing holding the gate shut is the worst
        # available failure, so an unrecognised value falls back to manual
        # rather than to nothing.
        mode = (self.permission_mode or "").strip() or "manual"
        args += ["--permission-mode", mode]
        if self.tools:
            args += ["--tools", ",".join(self.tools)]
        # Grants are the one-shot widening earned by a spoken yes. They ride on
        # the same flag as the standing read-only allowlist and last exactly as
        # long as the process they were passed to.
        allowed = tuple(dict.fromkeys(self.allowed_tools + tuple(grants)))
        if allowed:
            args += ["--allowedTools", ",".join(allowed)]
        for directory in self.add_dirs:
            args += ["--add-dir", directory]
        if resume_session:
            args += ["--resume", resume_session]
        return args


class ClaudeBrain:
    """A conversation with Claude that survives across many spoken turns.

    One process, one session id, many turns. The process is respawned with
    --resume if it dies, so a crash costs a pause rather than the conversation.
    """

    def __init__(self, config: BrainConfig | None = None, *, log=None) -> None:
        self.config = config or BrainConfig()
        self._log = log or (lambda message: None)
        self._process: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._parser = StreamParser()
        self._interrupted = threading.Event()
        self._busy = threading.Lock()
        self.session_id: str = ""
        self.last_turn: TurnComplete | None = None
        self.total_cost_usd: float = 0.0
        self.turn_count: int = 0
        # Permissions the user granted out loud, live only for the process they
        # were spawned with. Empty is the resting state and the safe one.
        self._grants: tuple[str, ...] = ()

    # --- lifecycle ----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def busy(self) -> bool:
        """True while a turn is in flight.

        The ambient loop checks this and skips rather than queueing, because a
        proactive remark that arrives four minutes late, behind the answer to a
        real question, is worse than one that never arrives.
        """
        return self._busy.locked()

    @property
    def grants(self) -> tuple[str, ...]:
        """What the running process is currently permitted beyond reading."""
        return self._grants

    @staticmethod
    def _child_env() -> dict:
        return child_env()

    def start(self, *, resume: bool = False) -> None:
        if self.alive:
            return
        argv = self.config.argv(self.session_id if resume else "", self._grants)
        self._log("spawning brain: " + " ".join(argv[:8]) + " ...")

        env = child_env()

        self._process = subprocess.Popen(
            argv,
            cwd=self.config.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=_NO_WINDOW,
        )
        self._parser = StreamParser()
        self._events = queue.Queue()
        for target, name in (
            (self._pump_stdout, "brain-stdout"),
            (self._pump_stderr, "brain-stderr"),
        ):
            threading.Thread(target=target, daemon=True, name=name).start()

    def stop(self, timeout: float = 5.0) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait(timeout=timeout)
        except Exception:
            process.kill()
        finally:
            self._log("brain stopped")

    def restart(self) -> None:
        """Respawn, resuming the same conversation if we have a session id."""
        self.stop()
        self.start(resume=bool(self.session_id))

    # --- permission grants --------------------------------------------------

    def grant(self, specs) -> None:
        """Widen the allowlist for the next turn, after a spoken yes.

        The CLI fixes its allowlist at spawn, so a grant means a respawn. That
        is affordable precisely because it only happens when the user has just
        agreed to something: the conversation itself survives via --resume, and
        a second of process start is invisible next to the sentence Vesper is
        about to speak.
        """
        specs = tuple(dict.fromkeys(s for s in specs if s))
        if not specs:
            return
        # Under the same lock ask() uses. Without it, a yes arriving while the
        # ambient loop was mid-turn tore that turn's process down underneath
        # it, and the old process's reader threads went on feeding a queue the
        # new process now owned.
        with self._busy:
            self._grants = specs
            self._log("granted for one turn: " + ", ".join(specs))
            self.stop()
            self.start(resume=bool(self.session_id))

    def revoke(self) -> None:
        """Drop back to read-only. Called as soon as the approved turn ends.

        Not deferred to the next question: the ambient loop shares this brain,
        and a grant left standing would let an unattended proactive turn use
        permission the user gave for something else entirely.
        """
        if not self._grants:
            return
        self._grants = ()
        self._log("grants revoked, back to read only")
        self.stop()
        self.start(resume=bool(self.session_id))

    def revoke_soon(self) -> threading.Thread:
        """Revoke in the background, behind the answer being spoken.

        Returns the thread so tests can join it. The respawn takes the busy
        lock, so the next question waits for read-only to be restored rather
        than racing it.
        """
        def worker() -> None:
            with self._busy:
                self.revoke()

        thread = threading.Thread(target=worker, daemon=True, name="brain-revoke")
        thread.start()
        return thread

    # --- pumps --------------------------------------------------------------

    def _pump_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                for event in self._parser.feed(line):
                    self._events.put(event)
        except Exception as exc:  # pragma: no cover - only on pipe teardown
            self._events.put(BrainError("stdout reader failed: " + str(exc)))
        finally:
            self._events.put(_SENTINEL)

    def _pump_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                line = line.strip()
                if line:
                    self._log("[brain stderr] " + line)
        except Exception:  # pragma: no cover
            pass

    # --- conversation -------------------------------------------------------

    def interrupt(self) -> None:
        """Abandon the current turn's output. Used for barge-in.

        The turn is not cancelled remotely: it keeps running and is drained in
        the background so the stream stays in sync for the next question. We
        simply stop relaying it to the voice.
        """
        self._interrupted.set()

    def ask(self, text: str) -> Iterator[Event]:
        """Send one user turn, yield events until the turn completes.

        Yields nothing further once interrupt() is called, but keeps consuming
        the child's output so the next ask() starts from a clean boundary.
        """
        if not text.strip():
            return

        with self._busy:
            # Inside the lock, so a background revoke cannot respawn the
            # process between the check and the write to its stdin.
            if not self.alive:
                self.start(resume=bool(self.session_id))
            self._interrupted.clear()
            process = self._process
            if process is None or process.stdin is None:
                yield BrainError("brain is not running")
                return

            try:
                process.stdin.write(encode_user_message(text))
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                yield BrainError("could not reach the brain: " + str(exc))
                self._process = None
                return

            deadline = time.monotonic() + self.config.turn_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    yield BrainError("that took too long, so I stopped waiting")
                    return
                try:
                    item = self._events.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if not self.alive:
                        yield BrainError("the brain process exited unexpectedly")
                        return
                    continue

                if item is _SENTINEL:
                    yield BrainError("the brain closed its output stream")
                    self._process = None
                    return

                if isinstance(item, TurnComplete):
                    self.session_id = item.session_id or self.session_id
                    self.last_turn = item
                    self.total_cost_usd += item.cost_usd
                    self.turn_count += 1
                    if not self._interrupted.is_set():
                        yield item
                    return

                if not self._interrupted.is_set():
                    yield item

    def note(self, text: str) -> None:
        """Send a turn we do not want spoken, and discard its reply.

        Used to tell Claude something happened, for example that the user cut it
        off mid-sentence, without triggering a spoken response.
        """
        for _ in self.ask(text):
            pass
