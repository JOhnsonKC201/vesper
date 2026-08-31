"""A small window that answers "what has it been doing".

Running from login, Vesper has no terminal, so everything it used to print goes
nowhere you can see. The tray tooltip holds about a hundred characters. This is
the rest: what it is doing now, what it has done this session, and what it
changed on your disk.

Tkinter, because it ships with Python. A prettier toolkit would be a new
dependency to draw one panel, on a machine that has had wheels blocked, and this
is a status panel rather than a product surface.

It runs on its own thread with its own event loop. Tk is not thread safe, so
every call into it happens on that thread and the only thing crossing the
boundary is a plain dictionary, read once per second.
"""

from __future__ import annotations

import threading
from datetime import timedelta

# The icon's palette, so the window and the tray look like one thing.
NIGHT = "#1c2038"
PANEL = "#232842"
LINE = "#2f3557"
STAR = "#fcdb95"
COOL = "#7edce2"
TEXT = "#e8eaf2"
MUTED = "#8b91ad"
WARN = "#f0a868"


class Dashboard:
    """A live status window, opened and closed from the tray."""

    def __init__(self, snapshot, *, on_toggle=None, on_open_log=None,
                 on_quit=None, voices=None, name: str = "Vesper") -> None:
        # A callable returning a plain dict. Deliberately not the Conversation
        # itself: this thread must never touch the audio loop's state directly.
        self.snapshot = snapshot
        self._on_toggle = on_toggle or (lambda paused: None)
        self._on_open_log = on_open_log or (lambda: None)
        self._on_quit = on_quit or (lambda: None)
        # Optional. None means there is nothing to pick between at all, and
        # then the picker is left out rather than shown as a dead control.
        # Kept as an opaque object so this file knows nothing about ElevenLabs,
        # Piper or SAPI: it renders whatever the panel reports, the same way it
        # renders `snapshot`.
        self.voices = voices
        self.name = name

        self._thread: threading.Thread | None = None
        self._root = None
        self._fields: dict[str, object] = {}
        self._ticks = 0
        self._closing = threading.Event()
        # Set from other threads, acted on by the Tk thread. Never a Tk
        # call from outside, which is the rule this whole design exists
        # to keep.
        self._wants_raise = threading.Event()
        self._wants_close = threading.Event()

        # Previewing a voice is a network call, or a model load off disk, and
        # either takes seconds. Doing it on the Tk thread would freeze the
        # window, so a worker does the work and leaves a plain string here for
        # the next tick to render. Neither side touches the other's objects.
        # Switching goes the same way for the same reason: picking a local
        # voice loads an ONNX model before it can answer.
        self._voice_note = ""
        self._voice_busy = threading.Event()

    # --- lifecycle ----------------------------------------------------------

    @property
    def open(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def show(self) -> None:
        """Open it, or bring it forward if it is already open.

        Called from the tray thread, so it touches no Tk object at all. Not
        even `after()`, which is itself a Tk call: doing it cross-thread is
        what made this abort the whole process rather than raise. Other threads
        set a flag; the Tk thread notices it on its next tick.
        """
        if self.open:
            self._wants_raise.set()
            return
        self._closing.clear()
        self._wants_close.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dashboard")
        self._thread.start()

    def close(self, timeout: float = 4.0) -> None:
        """Ask the window to close, and wait for its thread to finish.

        The wait is not politeness. A daemon thread still holding Tk objects
        when the process exits finalises them on the wrong thread, which calls
        Tcl_Panic and aborts, so this has to actually be done before shutdown
        carries on.
        """
        self._wants_close.set()
        # A preview is a network call that can outlive the window it was
        # started from. Left running, its `finally` clears `_voice_busy` long
        # after the fact, and until then the button on the *reopened* window
        # silently does nothing. Cancel it and drop the flag here instead.
        self._cancel_preview()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _raise(self) -> None:
        try:
            self._root.deiconify()
            self._root.lift()
            self._root.focus_force()
        except Exception:
            pass

    def _destroy(self) -> None:
        """End the event loop. The window is destroyed by the thread that owns it.

        Only `quit()` here, deliberately. Destroying the interpreter from a
        thread other than the one that created it produces
        "Tcl_AsyncDelete: async handler deleted by the wrong thread", which is
        Tcl noticing exactly that mistake. `_run` does the destroy once
        mainloop has returned, which is always on the right thread.
        """
        self._closing.set()
        try:
            self._root.quit()
        except Exception:
            pass

    # --- the window ---------------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            return

        try:
            root = tk.Tk()
            self._root = root
            root.title(f"{self.name}")
            root.configure(bg=NIGHT)
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self._destroy)
            try:
                root.iconbitmap(default=self._icon_path())
            except Exception:
                pass

            self._build(tk, root)
            self._refresh()
            root.mainloop()
        except Exception:
            pass
        finally:
            # On this thread, after mainloop has returned. Tk objects have to
            # be torn down by the thread that made them, and every reference to
            # one has to go with them. A single widget left in `_fields` is
            # enough: the main thread's garbage collector finalises it later,
            # Tcl notices the thread is wrong, and calls Tcl_Panic, which does
            # not raise an exception, it aborts the process.
            self._closing.set()
            try:
                if self._root is not None:
                    self._root.destroy()
            except Exception:
                pass
            self._fields.clear()
            self._root = None

    def _icon_path(self) -> str:
        from ..config import ROOT

        return str(ROOT / "var" / "icons" / "vesper-listening.ico")

    def _build(self, tk, root) -> None:
        pad = {"padx": 18}

        # Header: the one thing worth seeing from across the room.
        header = tk.Frame(root, bg=NIGHT)
        header.pack(fill="x", pady=(16, 4), **pad)
        self._fields["state"] = tk.Label(
            header, text="listening", font=("Segoe UI", 17, "bold"),
            bg=NIGHT, fg=STAR, anchor="w")
        self._fields["state"].pack(side="left")
        self._fields["uptime"] = tk.Label(
            header, text="", font=("Segoe UI", 10), bg=NIGHT, fg=MUTED, anchor="e")
        self._fields["uptime"].pack(side="right", pady=(8, 0))

        self._fields["detail"] = tk.Label(
            root, text="", font=("Segoe UI", 9), bg=NIGHT, fg=MUTED,
            anchor="w", justify="left", wraplength=340)
        self._fields["detail"].pack(fill="x", pady=(0, 12), **pad)

        # Numbers, in a monospace so they stop jittering as they change.
        self._counters(tk, root, "this session", [
            ("turns", "turns"), ("cost", "cost"), ("interruptions", "cut off"),
        ])
        self._counters(tk, root, "changes to your machine", [
            ("approvals", "approved"), ("refusals", "refused"),
            ("undos", "undone"),
        ])
        self._counters(tk, root, "who it heard", [
            ("voice_rejections", "not you"), ("echo_rejections", "itself"),
        ])

        if self.voices is not None:
            self._voice_section(tk, root)

        tk.Frame(root, bg=LINE, height=1).pack(fill="x", pady=(14, 0), **pad)

        buttons = tk.Frame(root, bg=NIGHT)
        buttons.pack(fill="x", pady=14, **pad)
        self._fields["pause"] = self._button(
            tk, buttons, "Pause", lambda: self._toggle())
        self._fields["pause"].pack(side="left")
        self._button(tk, buttons, "Open log", self._on_open_log).pack(side="left", padx=8)
        self._button(tk, buttons, "Quit", self._quit, danger=True).pack(side="right")

    def _voice_section(self, tk, root) -> None:
        """The voice picker, preview and the month's remaining allowance.

        A `tk.Listbox`, and the reason is not taste. The first version used
        `tk.OptionMenu`, which builds a `Menu` widget, and a menu belongs to
        the Tk interpreter rather than to the window. Open the dashboard, close
        it, and open it again: the second interpreter, created on a new thread,
        cannot build the menu, Tcl calls `Tcl_Panic` with "Failed to create the
        menu window", and the process dies with exit code 3.

        `scripts/dashboard_soak.py` is what found it, on the second cycle, and
        it is worth keeping because no unit test can catch this: a panic is not
        an exception, there is nothing to assert on, and the only symptom is
        that the process is gone.

        A Listbox also happens to be the better control here, since it shows
        all eighteen usable voices at once instead of hiding them behind a
        click.

        Every widget made here goes into `_fields`, which `_run` clears on the
        owning thread. One left on `self` would be finalised later by the main
        thread and abort the process in the same way.
        """
        tk.Label(root, text="VOICE", font=("Segoe UI", 8, "bold"),
                 bg=NIGHT, fg=MUTED, anchor="w").pack(fill="x", padx=18, pady=(12, 3))

        panel = tk.Frame(root, bg=PANEL)
        panel.pack(fill="x", padx=18)

        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", padx=10, pady=(9, 4))

        voices = self._available()
        listbox = tk.Listbox(
            row, height=5, activestyle="none", bg=PANEL, fg=TEXT,
            selectbackground=LINE, selectforeground=STAR, font=("Segoe UI", 9),
            relief="flat", bd=0, highlightthickness=0, exportselection=False,
            cursor="hand2",
        )
        for index, voice in enumerate(voices):
            listbox.insert("end", f"  {voice.label}")
            if not getattr(voice, "free", True):
                listbox.itemconfig(index, foreground=MUTED)
        if not voices:
            listbox.insert("end", "  no voices available")
        listbox.bind("<<ListboxSelect>>", lambda _event: self._pick())
        listbox.pack(side="left", fill="x", expand=True)
        self._fields["voice_list"] = listbox

        current = self._current_label([v.label for v in voices])
        for index, voice in enumerate(voices):
            if voice.label == current:
                listbox.selection_set(index)
                listbox.see(index)
                break

        self._fields["voice_preview"] = self._button(tk, row, "Preview", self._preview)
        self._fields["voice_preview"].pack(side="right", padx=(8, 0))

        # The allowance, drawn rather than written, because "how much is left"
        # is a glance question. Two frames: the track and the fill.
        #
        # Only when there is an allowance to run out of. A local voice buys
        # nothing, and a bar pinned at empty forever would be read as a warning
        # about something.
        if self._cap() > 0:
            track = tk.Frame(panel, bg=LINE, height=4)
            track.pack(fill="x", padx=10, pady=(2, 2))
            track.pack_propagate(False)
            fill = tk.Frame(track, bg=COOL, height=4)
            fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)
            self._fields["voice_bar"] = fill

        self._fields["voice_note"] = tk.Label(
            panel, text="", font=("Segoe UI", 8), bg=PANEL, fg=MUTED,
            anchor="w", justify="left", wraplength=320)
        self._fields["voice_note"].pack(fill="x", padx=10, pady=(2, 9))

    def _available(self):
        try:
            return list(self.voices.list())
        except Exception:
            return []

    def _cap(self) -> int:
        """How much this voice is allowed to spend a month. 0 means it is free."""
        try:
            return int(self.voices.budget()[1])
        except Exception:
            return 0

    def _current_label(self, names: list[str]) -> str:
        try:
            current = self.voices.current()
        except Exception:
            current = ""
        for name in names:
            if name.startswith(current):
                return name
        return names[0] if names else "no voices"

    def _counters(self, tk, root, title: str, items: list[tuple[str, str]]) -> None:
        tk.Label(root, text=title.upper(), font=("Segoe UI", 8, "bold"),
                 bg=NIGHT, fg=MUTED, anchor="w").pack(fill="x", padx=18, pady=(8, 3))
        row = tk.Frame(root, bg=PANEL)
        row.pack(fill="x", padx=18)
        for key, label in items:
            cell = tk.Frame(row, bg=PANEL)
            cell.pack(side="left", expand=True, fill="x", pady=9, padx=4)
            value = tk.Label(cell, text="0", font=("Cascadia Mono", 15),
                             bg=PANEL, fg=TEXT)
            value.pack()
            tk.Label(cell, text=label, font=("Segoe UI", 8),
                     bg=PANEL, fg=MUTED).pack()
            self._fields[key] = value

    def _button(self, tk, parent, text: str, command, danger: bool = False):
        return tk.Button(
            parent, text=text, command=command, font=("Segoe UI", 9),
            bg=PANEL, fg=WARN if danger else TEXT, activebackground=LINE,
            activeforeground=TEXT, relief="flat", bd=0, padx=14, pady=6,
            cursor="hand2", highlightthickness=0,
        )

    # --- live updates -------------------------------------------------------

    def _refresh(self) -> None:
        """The only place Tk is touched after startup, and it runs on Tk's own
        thread. Ticks four times a second so a close or a raise asked for by
        another thread feels immediate, but re-reads the numbers once a second,
        since nothing here changes faster than that."""
        root = self._root
        if root is None or self._closing.is_set():
            return

        if self._wants_close.is_set():
            self._destroy()
            return
        if self._wants_raise.is_set():
            self._wants_raise.clear()
            self._raise()

        self._ticks += 1
        if self._ticks % 4 == 1:
            try:
                self._apply(self.snapshot() or {})
            except Exception:
                # A dashboard that took down the assistant it reports on would
                # be a poor trade. Skip this tick and try again.
                pass
        try:
            root.after(250, self._refresh)
        except Exception:
            pass

    def _apply(self, data: dict) -> None:
        paused = bool(data.get("paused"))
        # Three states, not two. Asleep is the resting one and keeps the warm
        # star; awake means the follow-up window is open and plain speech is
        # being acted on, which is the one worth noticing from across the room.
        if paused:
            label, colour = "paused", MUTED
        elif data.get("awake"):
            label, colour = "awake", COOL
        else:
            label, colour = "asleep", STAR
        state = self._fields["state"]
        state.config(text=label, fg=colour)
        self._fields["pause"].config(text="Resume" if paused else "Pause")

        seconds = int(data.get("uptime_s") or 0)
        self._fields["uptime"].config(text=f"up {timedelta(seconds=seconds)}")

        detail = data.get("last_heard") or ""
        self._fields["detail"].config(
            text=f"last heard  {detail}" if detail
            else "nothing heard yet, say the wake word")

        for key in ("turns", "interruptions", "approvals", "refusals",
                    "undos", "voice_rejections", "echo_rejections"):
            widget = self._fields.get(key)
            if widget is not None:
                widget.config(text=str(data.get(key, 0)))

        cost = self._fields.get("cost")
        if cost is not None:
            cost.config(text=f"${float(data.get('cost_usd') or 0.0):.2f}")

        if self.voices is not None:
            self._apply_voice()

    # --- actions ------------------------------------------------------------

    def _toggle(self) -> None:
        try:
            paused = bool((self.snapshot() or {}).get("paused"))
            self._on_toggle(not paused)
        except Exception:
            pass

    def _quit(self) -> None:
        self._wants_close.set()
        try:
            self._on_quit()
        except Exception:
            pass

    # --- voice --------------------------------------------------------------

    def _selected_voice(self):
        """Which row is selected. Runs on the Tk thread.

        Reads the Listbox, which is the widget that has held the selection
        since the OptionMenu was removed. This went on reading a `voice_var`
        StringVar that the same change deleted, so it returned None every time:
        clicking a voice did nothing, Preview did nothing, and the picker was
        dead in the only build that ever showed it.

        By index, because that is what the widget reports and nothing promises
        two rows cannot read alike.
        """
        listbox = self._fields.get("voice_list")
        if listbox is None:
            return None
        try:
            selection = listbox.curselection()
        except Exception:
            return None
        if not selection:
            return None
        voices = self._available()
        index = int(selection[0])
        return voices[index] if 0 <= index < len(voices) else None

    def _pick(self) -> None:
        """Switch voice. Called from the list, so on the Tk thread.

        Dispatched for the same reason a preview is: switching to a local voice
        loads an ONNX model and hands it to the Speaker, which is roughly a
        second of work, and a window that stops redrawing for a second on every
        click reads as broken.
        """
        voice = self._selected_voice()
        if voice is None or self._voice_busy.is_set():
            return
        if self._is_current(voice):
            # Clicking the row that is already highlighted still lands here.
            # Tk's mouse binding fires <<ListboxSelect>> whether or not the
            # selection changed: tk8.6 listbox.tcl, ListboxBeginSelect ends in
            # FireListboxSelectEvent unconditionally. Without this, that click
            # would rebuild the voice already speaking and hand it to the
            # Speaker, which barges in, so Vesper would stop mid-sentence and
            # nothing on screen would say why.
            #
            # A selection set in code does not fire it, measured on tk 8.6.15,
            # so the row this window highlights when it opens is not the
            # problem. The click on it is.
            return
        if not getattr(voice, "free", True):
            self._voice_note = f"{voice.name} needs a paid ElevenLabs plan"
            return

        self._voice_busy.set()
        self._voice_note = f"switching to {voice.name}..."
        threading.Thread(
            target=self._pick_worker, args=(voice,), daemon=True,
            name="voice-pick",
        ).start()

    def _is_current(self, voice) -> bool:
        """Is this already the voice speaking? Asked of the panel, not cached,
        because it is the only side that can still be right after a switch."""
        try:
            current = self.voices.current()
        except Exception:
            return False
        return bool(current) and str(voice.label).startswith(current)

    def _pick_worker(self, voice) -> None:
        try:
            self.voices.choose(voice.voice_id)
            self._voice_note = f"{voice.name} it is"
        except Exception as exc:
            self._voice_note = f"could not switch: {type(exc).__name__}"
        finally:
            self._voice_busy.clear()

    def _preview(self) -> None:
        """Speak a sample line. Dispatched to a worker, never run here.

        A preview is an HTTP request that can take the full timeout. Running it
        on the Tk thread would freeze the window for twenty seconds, and the
        worker cannot touch Tk, so it leaves a string behind for the next tick.
        """
        if self._voice_busy.is_set():
            return
        voice = self._selected_voice()
        if voice is None:
            return
        if not getattr(voice, "free", True):
            self._voice_note = f"{voice.name} needs a paid ElevenLabs plan"
            return

        self._voice_busy.set()
        self._voice_note = f"speaking as {voice.name}..."
        threading.Thread(
            target=self._preview_worker, args=(voice,), daemon=True,
            name="voice-preview",
        ).start()

    def _cancel_preview(self) -> None:
        """Stop any preview in flight and let the button work again."""
        try:
            if self.voices is not None and hasattr(self.voices, "cancel"):
                self.voices.cancel()
        except Exception:
            pass
        self._voice_busy.clear()
        self._voice_note = ""

    def _preview_worker(self, voice) -> None:
        try:
            self._voice_note = self.voices.preview(voice.voice_id) or ""
        except Exception as exc:
            self._voice_note = f"preview failed: {type(exc).__name__}"
        finally:
            self._voice_busy.clear()

    def _apply_voice(self) -> None:
        """Render the voice panel. Runs on the Tk thread, once a second."""
        note = self._fields.get("voice_note")
        if note is None:
            return

        bar = self._fields.get("voice_bar")
        if bar is not None:
            try:
                used, cap = self.voices.budget()
            except Exception:
                used, cap = 0, 0
            share = 0.0 if cap <= 0 else min(1.0, max(0.0, used / cap))
            try:
                bar.place_configure(relwidth=share)
                bar.config(bg=WARN if share >= 1.0 else COOL)
            except Exception:
                pass

        # The steady line belongs to the panel: only it knows whether the thing
        # worth saying is an allowance, or that nothing is leaving the machine.
        if self._voice_note:
            text = self._voice_note
        else:
            try:
                text = self.voices.status()
            except Exception:
                text = ""
        try:
            note.config(text=text)
        except Exception:
            pass
