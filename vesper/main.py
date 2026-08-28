"""Entry point. Wires the parts together and starts listening.

Modes:
    python -m vesper.main             talk to it
    python -m vesper.main --text      type to it, no microphone needed
    python -m vesper.main --say "..." one question, then exit
    python -m vesper.main --devices   list microphones
    python -m vesper.main --check     verify every dependency, then exit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config as config_module
from .audio.mic import Microphone
from .audio.speaker import Speaker
from .audio.vad import EndpointConfig
from .brain.claude import BrainConfig, ClaudeBrain
from .brain.persona import build_system_prompt
from .brain.session_store import SessionStore
from .conversation import Conversation, ConversationConfig
from .proactive import ProactiveConfig, ProactiveLoop
from .stt.whisper import Listener, WhisperConfig, wake_word_prompt
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
                    model, speed=cfg.voice.speed, volume=cfg.voice.volume
                )
                return voice, f"{voice.name} (piper, offline)"
            except Exception as exc:
                ui.warn(f"piper failed to load ({exc}); falling back to Windows speech")

    from .tts.sapi import SapiTTS

    if SapiTTS.available():
        return SapiTTS(voice_hint=cfg.voice.sapi_voice_hint), "windows sapi"

    ui.warn("no speech backend available; Vesper will be silent")
    return NullVoice(), "silent"


def build(cfg: config_module.Config, *, with_mic: bool = True):
    ui = TerminalUI(show_cost=cfg.ui.show_cost)

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
        ),
        log=ui.info if "-v" in sys.argv else (lambda message: None),
    )

    stt = Listener(
        WhisperConfig(
            model=cfg.listening.whisper_model,
            initial_prompt=wake_word_prompt(
                cfg.identity.wake_words, cfg.identity.name
            ),
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
            barge_in_blocks=cfg.listening.barge_in_blocks,
            self_mute_ms=cfg.listening.self_mute_ms,
            greet_on_start=cfg.ui.greet_on_start,
        ),
        endpoint_config=EndpointConfig(
            end_silence_ms=cfg.listening.end_silence_ms,
            max_utterance_s=cfg.listening.max_utterance_s,
        ),
        ui=ui,
    )
    conversation.session_store = store if cfg.brain.remember_across_restarts else None

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


# --- modes ------------------------------------------------------------------


def run_voice(cfg: config_module.Config) -> int:
    conversation, ui, voice_label = build(cfg)
    ui.banner(
        brain=f"{cfg.brain.model} via claude cli, safe mode",
        voice=voice_label,
        ears=f"whisper {cfg.listening.whisper_model}",
        wake=cfg.identity.wake_words[0] if cfg.identity.wake_words else "vesper",
        cwd=cfg.brain_cwd(),
    )
    ui.info("loading models...")
    try:
        if conversation.proactive is not None and not conversation.proactive.muted:
            conversation.proactive.start()
        conversation.run()
    finally:
        if conversation.proactive is not None:
            conversation.proactive.stop()
        conversation.remember_session()
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
            conversation.respond(one_shot)
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
            conversation.respond(line)
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


def check(cfg: config_module.Config) -> int:
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
    parser.add_argument("--check", action="store_true", help="verify dependencies")
    parser.add_argument("--no-voice", action="store_true", help="stay silent")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.devices:
        return list_devices()

    cfg = config_module.load(args.config)
    if args.no_voice:
        cfg.voice.engine = "none"

    if args.check:
        return check(cfg)
    if args.say:
        return run_text(cfg, one_shot=args.say)
    if args.text:
        return run_text(cfg)
    return run_voice(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
