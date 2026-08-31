"""Entry point. Wires the parts together and starts listening.

Modes:
    python -m vesper.main             talk to it
    python -m vesper.main --text      type to it, no microphone needed
    python -m vesper.main --say "..." one question, then exit
    python -m vesper.main --devices   list microphones
    python -m vesper.main --check     verify every dependency, then exit
    python -m vesper.main --enroll    teach it your voice
    python -m vesper.main --install-autostart    start with Windows
    python -m vesper.main --check --offline   skip the paid gate check
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import autostart as autostart_module
from . import config as config_module
from . import single
from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig
from .brain.claude import BrainConfig, ClaudeBrain
from .brain.persona import build_system_prompt
from .brain.session_store import SessionStore
from .conversation import Conversation, ConversationConfig
from .logfile import LogFile, attach
from .proactive import ProactiveConfig, ProactiveLoop
from .stt.voiceprint import VoicePrint
from .stt.whisper import Listener, WhisperConfig, wake_word_prompt
from .tts import shaping
from .tts.base import NullVoice
from .ui.tui import TerminalUI
from .wake import WakeConfig, WakeGate


# --- assembly ---------------------------------------------------------------


def build_voice(cfg: config_module.Config, ui: TerminalUI):
    """Pick a speech backend, degrading rather than failing."""
    engine = (cfg.voice.engine or "piper").lower()

    if engine == "none":
        return NullVoice(), "silent"

    if engine == "piper":
        from .tts.piper_voice import PiperTTS

        model = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
        if model is None:
            ui.warn(
                f"no piper voice in {cfg.voices_path()}. "
                f"Run: python -m piper.download_voices {cfg.voice.model} "
                f"--data-dir {cfg.voice.voices_dir}"
            )
        else:
            try:
                voice = PiperTTS(
                    model,
                    speed=cfg.voice.speed,
                    volume=cfg.voice.volume,
                    character=shaping.preset(cfg.voice.character),
                )
                return voice, f"{voice.name} (piper, {voice.character.name})"
            except Exception as exc:
                ui.warn(f"piper failed to load ({exc}); falling back to Windows speech")

    from .tts.sapi import SapiTTS

    if SapiTTS.available():
        return SapiTTS(voice_hint=cfg.voice.sapi_voice_hint), "windows sapi"

    ui.warn("no speech backend available; Vesper will be silent")
    return NullVoice(), "silent"


def build(cfg: config_module.Config, *, with_mic: bool = True):
    ui = TerminalUI(show_cost=cfg.ui.show_cost)
    # Tee diagnostics to a file before anything else can fail, because
    # started from your login there is no console to print them to.
    log = LogFile(cfg.log_path())
    attach(ui, log)

    voice, voice_label = build_voice(cfg, ui)
    speaker = Speaker(voice, on_start=ui.spoke, on_error=lambda e: ui.error(str(e)))

    brain = ClaudeBrain(
        BrainConfig(
            executable=cfg.brain.executable,
            model=cfg.brain.model,
            cwd=cfg.brain_cwd(),
            system_prompt=build_system_prompt(
                cfg.identity.user, cfg.identity.personality
            ),
            tools=tuple(cfg.brain.tools),
            allowed_tools=tuple(cfg.brain.allowed_tools),
            add_dirs=tuple(cfg.brain.add_dirs),
            turn_timeout_s=cfg.brain.turn_timeout_s,
            permission_mode=cfg.brain.permission_mode,
        ),
        log=ui.info if "-v" in sys.argv else (lambda message: None),
    )

    stt = Listener(
        WhisperConfig(
            model=cfg.listening.whisper_model,
            initial_prompt=wake_word_prompt(
                cfg.identity.wake_words, cfg.identity.name
            ),
            cpu_threads=cfg.listening.cpu_threads,
        ),
        log=ui.info,
    )

    mic = Microphone(device=cfg.listening.device, on_error=lambda e: ui.warn(str(e)))

    store = SessionStore(
        cfg.session_path(), max_age_hours=cfg.brain.remember_max_age_hours
    )
    if cfg.brain.remember_across_restarts:
        previous = store.load()
        if previous is not None:
            brain.session_id = previous.session_id
            ui.info(
                f"picking up where we left off, "
                f"{previous.age_hours():.1f}h ago, {previous.turns} turns"
            )

    conversation = Conversation(
        brain=brain,
        stt=stt,
        speaker=speaker,
        mic=mic,
        wake=WakeGate(
            WakeConfig(
                words=tuple(cfg.identity.wake_words),
                follow_up_window_s=cfg.listening.follow_up_window_s,
                require_wake_word=cfg.listening.wake_required,
            )
        ),
        config=ConversationConfig(
            half_duplex=cfg.listening.half_duplex,
            tail_mute_ms=cfg.listening.tail_mute_ms,
            barge_in_blocks=cfg.listening.barge_in_blocks,
            self_mute_ms=cfg.listening.self_mute_ms,
            greet_on_start=cfg.ui.greet_on_start,
            consent_enabled=cfg.consent.enabled,
            consent_window_s=cfg.consent.window_s,
            audit_log=cfg.audit_path(),
            undo_dir=cfg.undo_path(),
        ),
        endpoint_config=EndpointConfig(
            end_silence_ms=cfg.listening.end_silence_ms,
            max_utterance_s=cfg.listening.max_utterance_s,
        ),
        ui=ui,
    )
    conversation.session_store = store if cfg.brain.remember_across_restarts else None
    conversation.voiceprint = VoicePrint(cfg.voiceprint_path())
    conversation.log = log

    # The ambient loop shares the brain and the speaker, and skips its check
    # rather than queueing whenever a real conversation is in progress.
    conversation.proactive = ProactiveLoop(
        brain=brain,
        speak=conversation._say,
        config=ProactiveConfig(
            enabled=cfg.proactive.enabled,
            check_interval_s=cfg.proactive.check_interval_s,
            min_interval_s=cfg.proactive.min_interval_s,
            quiet_hours=cfg.proactive.quiet_hours,
        ),
        ui=ui,
    )
    return conversation, ui, voice_label


# --- running unattended -----------------------------------------------------


def _start_tray(cfg: config_module.Config, conversation, ui):
    """The icon, if it is wanted and this machine can show one.

    Failure here is reported and then ignored. A missing tray icon is a worse
    experience; refusing to start over it would be a worse assistant.
    """
    if not cfg.runtime.tray:
        return None

    from .ui.dashboard import Dashboard
    from .ui.icon import ensure as ensure_icons
    from .ui.tray import TrayIcon

    def open_log() -> None:
        path = cfg.log_path()
        if path is not None and path.exists():
            os.startfile(str(path))  # noqa: S606 - opening our own log

    dashboard = Dashboard(
        conversation.status,
        name=cfg.identity.name,
        on_toggle=conversation.pause,
        on_open_log=open_log,
        on_quit=conversation.shutdown,
    )

    tray = TrayIcon(
        name=cfg.identity.name,
        on_toggle=lambda listening: conversation.pause(not listening),
        on_quit=conversation.shutdown,
        on_open_log=open_log,
        on_dashboard=dashboard.show,
        icons=ensure_icons(config_module.ROOT / "var" / "icons"),
        log=getattr(conversation, "log", None),
    )
    conversation.dashboard = dashboard
    if not tray.start():
        ui.warn("no tray icon; stop it with ctrl-c or by saying 'shut down'")
        return None
    tray.update(detail="say the wake word to talk")
    return tray


def _handle_signals(conversation, ui) -> None:
    """Make taskkill and a closing console as clean as saying "shut down".

    Without this, anything other than ctrl-c in a console skipped the exit
    path: the session id was never written, so the conversation was lost, and
    the claude child was left running.
    """
    import signal

    def bow_out(signum, frame):
        ui.info(f"signal {signum}, shutting down")
        conversation.shutdown()

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        handler = getattr(signal, name, None)
        if handler is None:
            continue
        try:
            signal.signal(handler, bow_out)
        except (ValueError, OSError):
            # Not the main thread, or a platform without it. Not fatal.
            pass


# --- modes ------------------------------------------------------------------


def run_voice(cfg: config_module.Config) -> int:
    if not single.claim():
        # Two copies both hold the microphone, so every utterance is answered
        # and spoken twice, over each other, and both spend your subscription
        # window. With autostart on, a second copy is the expected accident
        # rather than an unusual one.
        print("Vesper is already running. Use its tray icon to quit it first.")
        return 1

    conversation, ui, voice_label = build(cfg)
    ui.banner(
        brain=f"{cfg.brain.model} via claude cli, safe mode",
        voice=voice_label,
        ears=f"whisper {cfg.listening.whisper_model}",
        wake=cfg.identity.wake_words[0] if cfg.identity.wake_words else "vesper",
        cwd=cfg.brain_cwd(),
    )
    ui.info("loading models...")

    tray = _start_tray(cfg, conversation, ui)
    _handle_signals(conversation, ui)

    try:
        if conversation.proactive is not None and not conversation.proactive.muted:
            conversation.proactive.start()
        conversation.run()
    finally:
        # The dashboard first, and waited for. Its Tk objects have to be gone
        # before the process exits or they are finalised on the wrong thread,
        # which aborts rather than raises.
        dashboard = getattr(conversation, "dashboard", None)
        if dashboard is not None:
            dashboard.close()
        if tray is not None:
            tray.stop()
        if conversation.proactive is not None:
            conversation.proactive.stop()
        conversation.remember_session()
        single.release()
        ui.farewell()
    return 0


def run_text(cfg: config_module.Config, one_shot: str = "") -> int:
    """Type instead of talk. The fastest way to test without a microphone."""
    conversation, ui, voice_label = build(cfg)
    ui.banner(
        brain=f"{cfg.brain.model} via claude cli, safe mode",
        voice=voice_label,
        ears="text input",
        wake="type and press enter",
        cwd=cfg.brain_cwd(),
    )
    conversation.brain.start(resume=bool(conversation.brain.session_id))
    try:
        if one_shot:
            ui.heard(one_shot, addressed=True)
            conversation.hear(one_shot)
            conversation.speaker.wait_until_idle(timeout=60)
            return 0

        while True:
            try:
                line = input("\nyou     ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "bye"):
                break
            conversation.hear(line)
            conversation.speaker.wait_until_idle(timeout=120)
    finally:
        conversation.speaker.close()
        conversation.remember_session()
        conversation.brain.stop()
        ui.farewell()
    return 0


def list_devices() -> int:
    devices = Microphone.list_devices()
    if not devices:
        print("no input devices found")
        return 1
    print("input devices:")
    for device in devices:
        print(f"  [{device['index']:>2}] {device['name']}  ({device['channels']}ch)")
    print("\nset listening.device in config.yaml to one of these indexes")
    return 0


def check(cfg: config_module.Config, *, gate: bool = True) -> int:
    """Verify every moving part, and say which one is broken."""
    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = "ok  " if good else "FAIL"
        print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")

    print("\nvesper self check\n")

    import shutil

    claude = shutil.which(cfg.brain.executable)
    report("claude cli", claude is not None, claude or "not on PATH")

    try:
        import sounddevice  # noqa: F401

        count = len(Microphone.list_devices())
        report("microphone", count > 0, f"{count} input device(s)")
    except Exception as exc:
        report("microphone", False, str(exc))

    try:
        import faster_whisper  # noqa: F401

        report("whisper", True, cfg.listening.whisper_model)
    except Exception as exc:
        report("whisper", False, f"{type(exc).__name__}: {exc}")

    try:
        from .audio.vad import VoiceActivity

        report("silero vad", VoiceActivity().using_neural, "neural path")
    except Exception as exc:
        report("silero vad", False, str(exc))

    from .tts.piper_voice import PiperTTS

    model = PiperTTS.find_voice(cfg.voices_path(), cfg.voice.model)
    report("piper voice", model is not None, str(model) if model else "no .onnx found")

    from .tts.sapi import SapiTTS

    report("windows speech", SapiTTS.available(), "fallback")

    # Not a pass or fail, just the answer to "will it come back after a
    # reboot", which is the whole question once it runs from login.
    print(f"  --    autostart  {autostart_module.describe()}")
    from .stt.voiceprint import available as voice_model_ready

    print(f"  --    voice model  "
          f"{'ready' if voice_model_ready() else 'not downloaded, run --enroll'}")
    voiceprint = cfg.voiceprint_path()
    enrolled = voiceprint is not None and voiceprint.exists()
    print(f"  --    your voice  "
          f"{'enrolled' if enrolled else 'not enrolled, every voice is accepted'}")

    # Last, because it is the only check that talks to Anthropic, and because
    # everything above has to work for it to mean anything. It is also the only
    # check whose failure is dangerous rather than merely inconvenient: a
    # permission gate that has quietly stopped working looks exactly like one
    # that works, right up until something changes a file nobody approved.
    if gate:
        if not cfg.consent.enabled:
            report("permission gate", True, "consent disabled, nothing to verify")
        else:
            from . import gatecheck

            print("  ..    permission gate  asking the cli to do something forbidden")
            try:
                result = gatecheck.verify(cfg)
                report("permission gate", result.ok, result.detail)
                if not result.ok:
                    print(
                        "\n  Vesper can change files on a spoken yes, and that is only\n"
                        "  safe while the cli refuses what it is not allowed to do.\n"
                        "  Set consent.enabled to false until this passes."
                    )
            except Exception as exc:  # a broken check must not look like a pass
                report("permission gate", False, f"{type(exc).__name__}: {exc}")

    print()
    return 0 if ok else 1


# --- cli --------------------------------------------------------------------


def _force_utf8_console() -> None:
    """Windows consoles default to a legacy code page, which mangles anything
    outside ASCII into question marks."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(prog="vesper", description="A copilot you talk to.")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--text", action="store_true", help="type instead of speaking")
    parser.add_argument("--say", metavar="TEXT", help="ask one thing, then exit")
    parser.add_argument("--devices", action="store_true", help="list microphones")
    parser.add_argument("--enroll", action="store_true",
                        help="teach it your voice, so it ignores everyone else")
    parser.add_argument("--install-autostart", action="store_true",
                        help="start with Windows, hidden")
    parser.add_argument("--uninstall-autostart", action="store_true",
                        help="stop starting with Windows")
    parser.add_argument("--check", action="store_true", help="verify dependencies")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the permission gate check, which costs one short turn",
    )
    parser.add_argument("--no-voice", action="store_true", help="stay silent")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.devices:
        return list_devices()

    cfg = config_module.load(args.config)
    if args.no_voice:
        cfg.voice.engine = "none"

    if args.install_autostart:
        ok, detail = autostart_module.install(config_module.ROOT)
        print(f"autostart installed: {detail}" if ok else f"could not install: {detail}")
        return 0 if ok else 1
    if args.uninstall_autostart:
        ok, detail = autostart_module.uninstall()
        print(f"autostart removed: {detail}" if ok else f"could not remove: {detail}")
        return 0 if ok else 1
    if args.enroll:
        from .enroll import run as run_enroll

        return run_enroll(cfg)
    if args.check:
        return check(cfg, gate=not args.offline)
    if args.say:
        return run_text(cfg, one_shot=args.say)
    if args.text:
        return run_text(cfg)
    return run_voice(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
