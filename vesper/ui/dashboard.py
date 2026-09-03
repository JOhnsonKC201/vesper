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
THEM = "#9fb4ff"   # what you said
MINE = "#e8eaf2"   # what it said
GOOD = "#8fd8a8"

# Spelled out so no layer of quoting can eat it.
NEWLINE = chr(10)


class Dashboard:
    """A live status window, opened and closed from the tray."""

    def __init__(self, snapshot, *, on_toggle=None, on_open_log=None,
                 on_quit=None, voices=None, name: str = "Vasper",
                 transcript=None) -> None:
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
        # A callable returning a tuple of ("you"|"vasper", text). Optional, so
        # a dashboard can still be built without a conversation behind it.
        self.transcript = transcript
        # How many lines have been drawn, so a refresh appends the new ones
        # rather than rebuilding the whole widget and throwing away the
        # scrollback the reader may be part way through.
        self._drawn = 0

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
        """The layout, top to bottom in order of how much it matters.

        What it is doing now, then anything waiting on you, then what was
        actually said, then the numbers. The old version led with three grids
        of counters, which answered questions nobody had while burying the
        conversation entirely.
        """
        pad = {"padx": 18}
        root.configure(bg=NIGHT)

        # --- state: the one thing worth reading from across the room --------
        header = tk.Frame(root, bg=NIGHT)
        header.pack(fill="x", pady=(16, 2), **pad)
        self._fields["dot"] = tk.Label(
            header, text="●", font=("Segoe UI", 15), bg=NIGHT, fg=STAR)
        self._fields["dot"].pack(side="left", padx=(0, 8))
        self._fields["state"] = tk.Label(
            header, text="starting", font=("Segoe UI", 17, "bold"),
            bg=NIGHT, fg=STAR, anchor="w")
        self._fields["state"].pack(side="left")
        self._fields["uptime"] = tk.Label(
            header, text="", font=("Segoe UI", 9), bg=NIGHT, fg=MUTED, anchor="e")
        self._fields["uptime"].pack(side="right", pady=(9, 0))

        self._fields["hint"] = tk.Label(
            root, text="", font=("Segoe UI", 9), bg=NIGHT, fg=MUTED, anchor="w")
        self._fields["hint"].pack(fill="x", pady=(0, 10), **pad)

        # --- anything waiting on you, which outranks everything below -------
        ask = tk.Frame(root, bg=PANEL)
        self._fields["ask_frame"] = ask
        self._fields["ask"] = tk.Label(
            ask, text="", font=("Segoe UI", 10, "bold"), bg=PANEL, fg=WARN,
            anchor="w", justify="left", wraplength=330)
        self._fields["ask"].pack(fill="x", padx=12, pady=(9, 1))
        tk.Label(ask, text='say "Vasper, yes" to allow it, or "no" to leave it',
                 font=("Segoe UI", 8), bg=PANEL, fg=MUTED, anchor="w").pack(
            fill="x", padx=12, pady=(0, 9))

        # --- what was actually said -----------------------------------------
        heading = tk.Label(root, text="CONVERSATION", font=("Segoe UI", 7, "bold"),
                           bg=NIGHT, fg=MUTED, anchor="w")
        heading.pack(fill="x", **pad)
        # Kept so the pending banner can be packed *before* it. Without the
        # anchor, re-packing appends to the end of the order and the question
        # you are being asked appears below the Quit button.
        self._fields["after_ask"] = heading
        wrap = tk.Frame(root, bg=PANEL)
        wrap.pack(fill="both", expand=True, pady=(4, 12), **pad)
        log = tk.Text(
            wrap, height=13, width=44, bg=PANEL, fg=MINE, bd=0,
            font=("Segoe UI", 9), wrap="word", relief="flat",
            padx=10, pady=8, highlightthickness=0, cursor="arrow")
        bar = tk.Scrollbar(wrap, command=log.yview, width=10,
                           bg=PANEL, troughcolor=PANEL, bd=0,
                           highlightthickness=0, activebackground=LINE)
        log.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        log.pack(side="left", fill="both", expand=True)
        log.tag_configure("you", foreground=THEM, spacing1=5)
        log.tag_configure("vasper", foreground=MINE, spacing3=3)
        log.tag_configure("empty", foreground=MUTED)
        # Read only, but still selectable so a line can be copied out. Disabled
        # would prevent that, so the state is flipped only around writes.
        log.configure(state="disabled")
        self._fields["log"] = log

        # --- the numbers, as one sentence rather than three grids ------------
        self._fields["summary"] = tk.Label(
            root, text="", font=("Consolas", 9), bg=NIGHT, fg=MUTED, anchor="w")
        self._fields["summary"].pack(fill="x", **pad)
        self._fields["care"] = tk.Label(
            root, text="", font=("Consolas", 9), bg=NIGHT, fg=MUTED, anchor="w")
        self._fields["care"].pack(fill="x", pady=(2, 0), **pad)

        if self.voices is not None:
            self._voice_section(tk, root)

        tk.Frame(root, bg=LINE, height=1).pack(fill="x", pady=(12, 0), **pad)

        self._fields["say"] = tk.Label(
            root, font=("Segoe UI", 8), bg=NIGHT, fg=MUTED, anchor="w",
            justify="left", wraplength=340,
            text=" ".join((
                'try  "Vasper, what am I looking at"  ·  "open chrome"',
                '·  "look at my screen".  anytime:  "go to sleep"',
                '·  "be quiet"  ·  "undo that"',
            )))
        self._fields["say"].pack(fill="x", pady=(10, 0), **pad)

        buttons = tk.Frame(root, bg=NIGHT)
        buttons.pack(fill="x", pady=14, **pad)
        self._fields["pause"] = self._button(tk, buttons, "Pause", lambda: self._toggle())
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
        self._apply_state(data)
        self._apply_ask(data)
        self._apply_transcript()
        self._apply_numbers(data)
        if self.voices is not None:
            self._apply_voice()

    def _apply_state(self, data: dict) -> None:
        """What it is doing, in words a person would use.

        Four states, not the old three. "waiting" is split out because it is
        the only one that needs something from you, and reading "awake" while
        a question sits unanswered told you nothing about that.
        """
        if data.get("paused"):
            label, colour, hint = "paused", MUTED, "not listening. press Resume."
        elif data.get("pending"):
            label, colour, hint = "waiting on you", WARN, "there is a question below."
        elif data.get("awake"):
            label, colour, hint = "listening", COOL, "just talk, no need to say the name."
        else:
            label, colour, hint = "asleep", STAR, 'say "Vasper" to wake him.'

        self._fields["state"].config(text=label, fg=colour)
        self._fields["dot"].config(fg=colour)
        self._fields["hint"].config(text=hint)
        self._fields["pause"].config(text="Resume" if data.get("paused") else "Pause")

        seconds = int(data.get("uptime_s") or 0)
        self._fields["uptime"].config(text=f"up {timedelta(seconds=seconds)}")

    def _apply_ask(self, data: dict) -> None:
        """The pending question, shown only while there is one.

        Packed and unpacked rather than blanked, so the space it takes is not
        held open by an empty box for the whole session.
        """
        pending = str(data.get("pending") or "")
        frame = self._fields["ask_frame"]
        if pending:
            self._fields["ask"].config(text=f"Shall I {pending}?")
            if not frame.winfo_ismapped():
                anchor = self._fields.get("after_ask")
                if anchor is not None:
                    frame.pack(fill="x", padx=18, pady=(0, 12), before=anchor)
                else:
                    frame.pack(fill="x", padx=18, pady=(0, 12))
        elif frame.winfo_ismapped():
            frame.pack_forget()

    def _apply_transcript(self) -> None:
        """Append what is new. Never redraw what is already there.

        Rebuilding the widget each tick would fight anyone scrolled back
        through it, and reset the selection they were about to copy.
        """
        if self.transcript is None:
            return
        try:
            lines = self.transcript()
        except Exception:
            return

        log = self._fields["log"]
        # A shorter history than last time means the session restarted, so the
        # widget is stale rather than merely behind.
        if len(lines) < self._drawn:
            self._drawn = 0
            log.configure(state="normal")
            log.delete("1.0", "end")
            log.configure(state="disabled")

        if not lines:
            if self._drawn == 0:
                self._write(log, "nothing said yet." + NEWLINE, "empty")
                self._drawn = -1
            return
        if self._drawn < 0:
            log.configure(state="normal")
            log.delete("1.0", "end")
            log.configure(state="disabled")
            self._drawn = 0

        for who, text in lines[self._drawn:]:
            speaker = "you" if who == "you" else self.name.lower()
            self._write(log, f"{speaker}   {text}" + NEWLINE, "you" if who == "you" else "vasper")
        self._drawn = len(lines)
        log.see("end")

    @staticmethod
    def _write(log, text: str, tag: str) -> None:
        log.configure(state="normal")
        log.insert("end", text, tag)
        log.configure(state="disabled")

    def _apply_numbers(self, data: dict) -> None:
        """Two lines instead of three grids of counters.

        The old panel showed seven numbers with no scale and no meaning. These
        are the two questions a person actually has: how much have I used it,
        and has it touched anything of mine.
        """
        turns = int(data.get("turns") or 0)
        cost = float(data.get("cost_usd") or 0.0)
        learned = int(data.get("learned") or 0)
        summary = f"{turns} asked  ·  ${cost:.2f}"
        if learned:
            summary += f"  ·  {learned} thing{'s' if learned != 1 else ''} remembered"
        self._fields["summary"].config(text=summary)

        approvals = int(data.get("approvals") or 0)
        refusals = int(data.get("refusals") or 0)
        undos = int(data.get("undos") or 0)
        unasked = int(data.get("unasked") or 0)
        if not any((approvals, refusals, undos, unasked)):
            care, colour = "nothing on your machine has been changed", MUTED
        else:
            bits = [f"{approvals} approved"]
            if refusals:
                bits.append(f"{refusals} refused")
            if undos:
                bits.append(f"{undos} undone")
            care, colour = "  ·  ".join(bits), GOOD
            if unasked:
                # The one number that should always be zero.
                care += f"   {unasked} NOT ASKED ABOUT"
                colour = WARN
        self._fields["care"].config(text=care, fg=colour)

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
