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
                 on_quit=None, name: str = "Vesper") -> None:
        # A callable returning a plain dict. Deliberately not the Conversation
        # itself: this thread must never touch the audio loop's state directly.
        self.snapshot = snapshot
        self._on_toggle = on_toggle or (lambda paused: None)
        self._on_open_log = on_open_log or (lambda: None)
        self._on_quit = on_quit or (lambda: None)
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

        tk.Frame(root, bg=LINE, height=1).pack(fill="x", pady=(14, 0), **pad)

        buttons = tk.Frame(root, bg=NIGHT)
        buttons.pack(fill="x", pady=14, **pad)
        self._fields["pause"] = self._button(
            tk, buttons, "Pause", lambda: self._toggle())
        self._fields["pause"].pack(side="left")
        self._button(tk, buttons, "Open log", self._on_open_log).pack(side="left", padx=8)
        self._button(tk, buttons, "Quit", self._quit, danger=True).pack(side="right")

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
        state = self._fields["state"]
        state.config(text="paused" if paused else "listening",
                     fg=MUTED if paused else STAR)
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
