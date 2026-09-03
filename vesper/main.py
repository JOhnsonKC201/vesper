"""Entry point. Wires the parts together and starts listening.

Modes:
    python -m vesper.main             talk to it
    python -m vesper.main --text      type to it, no microphone needed
    python -m vesper.main --say "..." one question, then exit
    python -m vesper.main --devices   list microphones
    python -m vesper.main --check     verify every dependency, then exit
    python -m vesper.main --enroll    teach it your voice
    python -m vesper.main --install-autostart    start with Windows
    python -m vesper.main --desktop-icon         put a launcher on the desktop
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
from . import desktop as desktop_module
from . import resume as resume_module
from . import single
from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig
from .brain.claude import BrainConfig, ClaudeBrain
from .brain.persona import build_system_prompt
from .brain.session_store import SessionStore
from .conversation import Conversation, ConversationConfig
from .learning import Lessons
from .logfile import LogFile, attach
from .proactive import ProactiveConfig, ProactiveLoop
from .stt.voiceprint import VoicePrint
from .stt.whisper import Listener, WhisperConfig, wake_word_prompt
from .tts import shaping
from .tts.base import NullVoice
from .ui.tui import TerminalUI
from .wake import WakeConfig, WakeGate


# --- assembly ---------------------------------------------------------------


def build_local_voice(cfg: config_module.Config, ui: TerminalUI, *,
                      engine: str, chosen=None):
    """The voice that works with the network unplugged.

    Split out from `build_voice` because the cloud backend does not replace
    this, it sits on top of it. `conversation.py` handles "Quiet from now on."
    and "Shutting down." locally so they survive an outage, and that promise
    only holds if there is always a local voice underneath.

    `chosen` is what was picked in the dashboard, and it wins over config.yaml
    for the same reason it does for the cloud voice: a click is more recent and
    more deliberate than a file edited last month. A local id carries both the
    model and the delivery, so restoring one restores how it sounded, not just
    which voice it was.
    """
    picked_engine, picked_id = "", ""
    if chosen is not None and chosen.local:
        picked_engine, picked_id = chosen.engine, chosen.voice_id

    want = picked_engine or ("piper" if engine in ("piper", "elevenlabs") else engine)

    if want == "piper":
        from .tts import catalog
        from .tts.piper_voice import PiperTTS

        stem, character = catalog.split_piper_id(picked_id)
        model = PiperTTS.find_voice(cfg.voices_path(), stem or cfg.voice.model)
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
                    character=shaping.preset(
                        character if picked_id else cfg.voice.character
                    ),
                )
                return voice, f"{voice.name} (piper, {voice.character.name})"
            except Exception as exc:
                ui.warn(f"piper failed to load ({exc}); falling back to Windows speech")

    from .tts.sapi import SapiTTS

    hint = picked_id if picked_engine == "sapi" else cfg.voice.sapi_voice_hint
    if SapiTTS.available():
        return SapiTTS(voice_hint=hint), "windows sapi"

    # A remembered SAPI voice on a machine with no SAPI. Falling back to Piper
    # beats going silent over a click made months ago on another machine.
    if picked_engine and want != "piper":
        return build_local_voice(cfg, ui, engine="piper")

    ui.warn("no speech backend available; Vesper will be silent")
    return NullVoice(), "silent"


def build_voice(cfg: config_module.Config, ui: TerminalUI):
    """Pick a speech backend, degrading rather than failing."""
    engine = (cfg.voice.engine or "piper").lower()

    if engine == "none":
        return NullVoice(), "silent"

    from . import voicechoice

    # Read once and given to both halves: the same file records a local pick
    # and a cloud one, and which it is decides who honours it.
    chosen = voicechoice.load(cfg.voice_choice_path())

    local, label = build_local_voice(cfg, ui, engine=engine, chosen=chosen)
    if engine != "elevenlabs":
        return local, label

    key = cfg.eleven_key()
    if not key:
        # Not a warning worth alarming about: it is exactly what happens
        # between switching the engine on and pasting a key in.
        ui.warn(f"voice.engine is elevenlabs but no API key is set; using {label}")
        return local, label

    from .tts import eleven

    # A voice picked in the dashboard beats the one written in config.yaml,
    # because clicking it is both more recent and more deliberate. Only when
    # it is an ElevenLabs voice: a remembered Piper model is an id this API
    # would reject, and it has already been honoured by the local half above.
    cloud_id = "" if chosen.local else chosen.voice_id
    settings = cfg.voice.eleven
    try:
        cloud = eleven.build(
            local,
            api_key=key,
            voice_id=cloud_id or settings.voice_id,
            model_id=settings.model_id,
            cache_dir=cfg.voice_cache_path(),
            budget_path=cfg.voice_budget_path(),
            monthly_characters=settings.monthly_characters,
            timeout_s=settings.timeout_s,
            log=ui.info,
        )
    except Exception as exc:
        ui.warn(f"elevenlabs failed to start ({exc}); using {label}")
        return local, label

    if settings.prewarm:
        # Off the main thread: it is a handful of network calls and the first
        # reply must not wait for them.
        cloud.prewarm_async()
    return cloud, f"{cloud.name} over {label}"


def build(cfg: config_module.Config, *, with_mic: bool = True):
    ui = TerminalUI(show_cost=cfg.ui.show_cost)
    # Tee diagnostics to a file before anything else can fail, because
    # started from your login there is no console to print them to.
    log = LogFile(cfg.log_path())
    attach(ui, log)

    # Built before the brain, because what he has learned goes into the
    # system prompt and the prompt is fixed for the life of the process.
    lessons = Lessons(cfg.lessons_path())
    if lessons.active():
        ui.info(lessons.summary())

    voice, voice_label = build_voice(cfg, ui)
    speaker = Speaker(voice, on_start=ui.spoke, on_error=lambda e: ui.error(str(e)))

    brain = ClaudeBrain(
        BrainConfig(
            executable=cfg.brain.executable,
            model=cfg.brain.model,
            cwd=cfg.brain_cwd(),
            system_prompt=build_system_prompt(
                cfg.identity.user, cfg.identity.personality, lessons.prompt_block()
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
            device=cfg.listening.whisper_device,
            compute_type=cfg.listening.whisper_compute,
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
            name=cfg.identity.name,
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
    # The same store the prompt was built from, so "forget that" removes the
    # line Claude was actually given rather than a second copy of it.
    conversation.lessons = lessons if cfg.runtime.lessons else None
    conversation.voiceprint = VoicePrint(cfg.voiceprint_path())
    conversation.log = log

    # The ambient loop shares the brain and the speaker, and skips its check
    # rather than queueing whenever a real conversation is in progress.
    conversation.proactive = ProactiveLoop(
        brain=brain,
        # `volunteer`, not `_say`: a remark he started opens the follow-up
        # window, so "your battery is at nine percent" can be answered with
        # "plug it in" rather than with "Vesper, plug it in".
        speak=conversation.volunteer,
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

    # `conversation.speaker`, not `speaker`. This function receives only cfg,
    # conversation and ui: the bare name was a local of `build()`, a different
    # function, so this raised NameError on every launch with the tray on,
    # which is the default. No test called _start_tray, so it went unnoticed.
    voice = conversation.speaker.voice

    # Whichever backend is running gets a picker. This used to be the cloud
    # one or nothing, on the grounds that a local picker would have one entry
    # and nothing to say, and that left the default install with no way to
    # change voice short of editing config.yaml and restarting. It is also
    # untrue: three deliveries per Piper model, plus every voice Windows has.
    panel = None
    if hasattr(voice, "say_as"):
        from .ui.voicepanel import VoicePanel

        panel = VoicePanel(voice, choice_path=cfg.voice_choice_path(), log=ui.info)
    elif isinstance(voice, NullVoice):
        # `engine: none` is a deliberate vow of silence, and a picker here
        # would be an offer to break it. Not a setting anyone went looking for.
        panel = None
    else:
        from .ui.voicepanel import LocalVoicePanel

        panel = LocalVoicePanel(
            conversation.speaker,
            voices_dir=cfg.voices_path(),
            speed=cfg.voice.speed,
            volume=cfg.voice.volume,
            choice_path=cfg.voice_choice_path(),
            log=ui.info,
        )
        # Neither a Piper model nor SAPI: nothing to pick between, and a
        # picker with no rows is the dead control the old comment feared.
        if not panel.list():
            panel = None

    dashboard = Dashboard(
        conversation.status,
        name=cfg.identity.name,
        on_toggle=conversation.pause,
        on_open_log=open_log,
        on_quit=conversation.shutdown,
        voices=panel,
        # A second accessor rather than a field on `status`, which two tests
        # hold to scalars. Both are read from the dashboard thread and both
        # hand back a fresh immutable copy.
        transcript=conversation.transcript,
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
    # So the audio loop can push awake and asleep to the icon the moment either
    # happens. Set before `start()`, because the first block can arrive while
    # the tray thread is still coming up.
    conversation.tray = tray
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


def console_is_visible() -> bool:
    """Can a printed line actually be read by anyone?

    Started from the desktop icon it cannot. The launcher is wscript running
    `cmd /c run.bat` with the window hidden, so there is a console attached and
    it has never been on screen. `isatty()` says yes and is wrong, which is why
    this asks the window manager instead.

    When it cannot tell, it says yes and the caller stays quiet: an unexpected
    dialog box is worse than a message nobody needed.
    """
    try:
        import ctypes

        window = ctypes.windll.kernel32.GetConsoleWindow()
        return bool(window) and bool(ctypes.windll.user32.IsWindowVisible(window))
    except Exception:
        return True


def say_on_screen(*lines: str, title: str = "Vesper") -> bool:
    """Put a message where someone with no console will see it.

    Returns whether it was shown, which is only interesting to the tests.
    """
    if console_is_visible():
        return False
    try:
        import ctypes

        # OK, information icon, topmost: without topmost it can open behind
        # whatever is full screen, which makes the double click look ignored
        # all over again.
        ctypes.windll.user32.MessageBoxW(
            0, os.linesep.join(lines), title, 0x40 | 0x40000
        )
        return True
    except Exception:
        return False


def run_voice(cfg: config_module.Config, *, if_idle: bool = False) -> int:
    if not single.claim():
        if if_idle:
            # The resume task fires on every unlock, and Vesper is usually
            # already running by then. Saying so in a dialog box every time
            # the lid opens would be worse than the thing it reports.
            return 0
        # Two copies both hold the microphone, so every utterance is answered
        # and spoken twice, over each other, and both spend your subscription
        # window. With autostart on, a second copy is the expected accident
        # rather than an unusual one.
        #
        # Said twice on purpose. Started from a terminal the print is enough;
        # started from the desktop icon there is no window to print into, and
        # the double click would otherwise do nothing at all, visibly.
        headline = "Vesper is already running."
        detail = ("Its icon is in the tray, by the clock. Right click it "
                  "for the dashboard, or to quit.")
        print(headline, detail)
        say_on_screen(headline, "", detail)
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

        # Loading it, not merely importing it. The import proves a package is
        # installed; it does not answer "will this transcribe, and on what".
        # A configured `cuda` that cannot really work builds a model happily
        # and then raises on every utterance, so the only honest answer here
        # comes from a model that has actually been built and probed.
        probe = Listener(
            WhisperConfig(
                model=cfg.listening.whisper_model,
                device=cfg.listening.whisper_device,
                compute_type=cfg.listening.whisper_compute,
                cpu_threads=cfg.listening.cpu_threads,
            )
        )
        probe.load()
        report(
            "whisper",
            True,
            f"{probe.resolved_model} on {probe.resolved_device}/{probe.resolved_compute}",
        )
        if probe.resolved_device != "cuda":
            from .stt import accel

            hint = accel.missing_runtime_hint()
            if hint:
                print(f"  --    gpu  {hint}")
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
    print(f"  --    desktop icon  {desktop_module.describe()}")
    print(f"  --    on wake  {resume_module.describe()}")
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
    parser.add_argument("--desktop-icon", action="store_true",
                        help="put a launcher icon on the desktop")
    parser.add_argument("--remove-desktop-icon", action="store_true",
                        help="take that icon off the desktop again")
    parser.add_argument("--uninstall-autostart", action="store_true",
                        help="stop starting with Windows")
    parser.add_argument("--install-resume", action="store_true",
                        help="also come back on unlock and wake, not just login")
    parser.add_argument("--uninstall-resume", action="store_true",
                        help="stop coming back on unlock and wake")
    parser.add_argument("--if-idle", action="store_true",
                        help="do nothing if Vesper is already running")
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
    if args.desktop_icon:
        ok, detail = desktop_module.install(config_module.ROOT)
        print(f"desktop icon created: {detail}" if ok
              else f"could not create it: {detail}")
        return 0 if ok else 1
    if args.remove_desktop_icon:
        ok, detail = desktop_module.uninstall()
        print(f"desktop icon removed: {detail}" if ok
              else f"could not remove it: {detail}")
        return 0 if ok else 1
    if args.uninstall_autostart:
        ok, detail = autostart_module.uninstall()
        print(f"autostart removed: {detail}" if ok else f"could not remove: {detail}")
        return 0 if ok else 1
    if args.install_resume:
        ok, detail = resume_module.install(config_module.ROOT)
        print(f"resume installed: {detail}" if ok else f"could not install: {detail}")
        return 0 if ok else 1
    if args.uninstall_resume:
        ok, detail = resume_module.uninstall()
        print(f"resume removed: {detail}" if ok else f"could not remove: {detail}")
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
    return run_voice(cfg, if_idle=args.if_idle)


if __name__ == "__main__":
    raise SystemExit(main())
