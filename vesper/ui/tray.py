"""A notification-area icon, so a windowless Vesper can still be seen and stopped.

This is not decoration. Before it existed there was no way to stop Vesper
without a console: no signal handling, and voice mode exits only on
KeyboardInterrupt. Killing it from Task Manager skips both `finally` blocks, so
the session id is never written and the `claude` child is left running. Starting
from login without this would mean an invisible process you cannot stop and
cannot debug.

Written directly against `Shell_NotifyIcon` through pywin32, which is already a
dependency. `pystray` would be the obvious choice and pulls in Pillow; this
machine has had wheels blocked by Windows Application Control before, and two
new packages to draw one 16x16 icon is a bad trade.

The icon lives on its own thread with its own message pump, because Windows
delivers notification-area callbacks to the thread that created the window and
Vesper's main thread is busy reading audio thirty times a second.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# Message ids. WM_USER + n is the conventional private range.
_WM_TRAY = 0x0400 + 20
_ID_TOGGLE = 1001
_ID_LOG = 1002
_ID_QUIT = 1003
_ID_DASHBOARD = 1004


@dataclass
class TrayState:
    """What the icon should currently say."""

    listening: bool = True
    # Awake means the follow-up window is open and plain speech counts as
    # talking to him. Asleep, which is the resting state, means he is waiting
    # to hear his name. Both are "listening": the difference is what he will
    # act on, and that is worth being able to see from across the room.
    awake: bool = False
    detail: str = ""

    def tooltip(self, name: str = "Vesper") -> str:
        if not self.listening:
            head = f"{name}: paused"
        elif self.awake:
            head = f"{name}: awake, say anything"
        else:
            head = f"{name}: asleep, say his name"
        # Windows truncates tooltips at 128 characters and silently fails on
        # some builds if the string is longer.
        return (f"{head}\n{self.detail}" if self.detail else head)[:127]


class TrayIcon:
    """A notification icon driven from another thread.

    Every callback runs on the tray thread, not the caller's, so anything they
    touch has to be safe to call from anywhere. `on_quit` in particular is
    expected to ask the conversation to stop rather than stopping it in place.
    """

    def __init__(
        self,
        *,
        on_toggle=None,
        on_quit=None,
        on_open_log=None,
        on_dashboard=None,
        name: str = "Vesper",
        log=None,
        icons: dict | None = None,
    ) -> None:
        self.name = name
        self.state = TrayState()
        self._on_toggle = on_toggle or (lambda listening: None)
        self._on_quit = on_quit or (lambda: None)
        self._on_open_log = on_open_log or (lambda: None)
        self._on_dashboard = on_dashboard or (lambda: None)
        self._icons = icons or {}
        # One handle per state, loaded once. See `_icon_handle`: an icon loaded
        # from a file is ours to own and ours to free, and it was neither.
        self._icon_handles: dict[str, int] = {}
        self._log = log
        self._thread: threading.Thread | None = None
        self._hwnd = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self.available = False

    # --- lifecycle ----------------------------------------------------------

    def start(self, timeout: float = 5.0) -> bool:
        """Show the icon. False if this machine cannot, which is not fatal."""
        try:
            import win32gui  # noqa: F401
        except ImportError:
            self._note("pywin32 missing, no tray icon")
            return False

        self._thread = threading.Thread(target=self._run, daemon=True, name="tray")
        self._thread.start()
        self._ready.wait(timeout)
        return self.available

    def stop(self) -> None:
        """Remove the icon and end its message pump."""
        self._stopping.set()
        if self._hwnd is None:
            return
        try:
            import win32gui

            win32gui.PostMessage(self._hwnd, 0x0010, 0, 0)  # WM_CLOSE
        except Exception:
            pass

    def update(self, *, listening: bool | None = None, awake: bool | None = None,
               detail: str | None = None) -> None:
        """Change what the tooltip and icon show.

        Called from whichever thread noticed the change, which is the audio
        loop for `awake` and the tray's own pump for `listening`. That is safe:
        the work is a `Shell_NotifyIcon` call, which talks to the shell rather
        than to this window's message queue.
        """
        if listening is not None:
            self.state.listening = listening
        if awake is not None:
            self.state.awake = awake
        if detail is not None:
            self.state.detail = detail
        self._refresh()

    # --- the tray thread ----------------------------------------------------

    def _run(self) -> None:
        try:
            import win32api
            import win32con
            import win32gui
        except ImportError:
            self._ready.set()
            return

        try:
            wndclass = win32gui.WNDCLASS()
            wndclass.hInstance = win32api.GetModuleHandle(None)
            wndclass.lpszClassName = "VesperTray"
            wndclass.lpfnWndProc = {
                _WM_TRAY: self._on_tray_message,
                win32con.WM_COMMAND: self._on_command,
                win32con.WM_CLOSE: self._on_close,
                win32con.WM_DESTROY: self._on_destroy,
            }
            atom = win32gui.RegisterClass(wndclass)
            # A message-only window: never shown, exists to receive callbacks.
            self._hwnd = win32gui.CreateWindow(
                atom, "Vesper", 0, 0, 0, 0, 0, 0, 0, wndclass.hInstance, None
            )
            win32gui.UpdateWindow(self._hwnd)
            self._add_icon()
            self.available = True
        except Exception as exc:
            self._note(f"tray icon unavailable: {exc}")
            self._ready.set()
            return

        self._ready.set()
        try:
            win32gui.PumpMessages()
        except Exception:
            pass

    def icon_state(self) -> str:
        """Which drawn icon the current state calls for.

        Its own method so it can be tested. Inside `_icon_handle` it sat
        between two win32 imports and a `LoadImage`, where nothing could reach
        it without a message pump and a desktop.
        """
        if not self.state.listening:
            return "paused"
        # The cool star for awake. It has been drawn into var/icons since the
        # icon set was written, and loaded by nothing until now.
        return "working" if self.state.awake else "listening"

    def _icon_handle(self):
        """The drawn icon for the current state, or a stock one if it is missing.

        The stock icon used to be the only option, and in a tray of fifteen
        identical grey squares it defeated the point: the icon exists so a
        glance tells you whether Vesper is listening.
        """
        import win32con
        import win32gui

        state = self.icon_state()
        cached = self._icon_handles.get(state)
        if cached:
            return cached

        path = self._icons.get(state)
        if path is not None:
            try:
                handle = win32gui.LoadImage(
                    0, str(path), win32con.IMAGE_ICON, 0, 0,
                    win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE,
                )
                if handle:
                    # LR_SHARED is deliberately absent, which makes this handle
                    # ours to own and ours to free. It was neither: a new one was
                    # loaded on every tray update and none was ever destroyed, so
                    # a day of waking and sleeping leaked one handle per
                    # transition. Only three states exist, so caching them is the
                    # fix and also stops the .ico being re-read from disk.
                    self._icon_handles[state] = handle
                    return handle
            except Exception:
                pass
        # Loaded from a null module, so this one is a shared system resource that
        # Windows owns. It must never be passed to DestroyIcon, which is exactly
        # why it is not kept alongside the others.
        which = win32con.IDI_APPLICATION if self.state.listening else win32con.IDI_HAND
        return win32gui.LoadIcon(0, which)

    def destroy_icons(self) -> None:
        """Free every icon handle loaded from a file. Safe to call twice."""
        if not self._icon_handles:
            return
        try:
            import win32gui
        except ImportError:
            self._icon_handles.clear()
            return
        for handle in self._icon_handles.values():
            try:
                win32gui.DestroyIcon(handle)
            except Exception:
                pass
        self._icon_handles.clear()

    def _notify(self, action: int) -> None:
        import win32gui

        flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
        entry = (
            self._hwnd, 0, flags, _WM_TRAY, self._icon_handle(),
            self.state.tooltip(self.name),
        )
        win32gui.Shell_NotifyIcon(action, entry)

    def _add_icon(self) -> None:
        import win32gui

        self._notify(win32gui.NIM_ADD)

    def _refresh(self) -> None:
        if self._hwnd is None or not self.available:
            return
        try:
            import win32gui

            self._notify(win32gui.NIM_MODIFY)
        except Exception:
            pass

    # --- callbacks, all on the tray thread ----------------------------------

    def _on_tray_message(self, hwnd, msg, wparam, lparam):
        import win32con

        if lparam == win32con.WM_LBUTTONDBLCLK:
            self._safely(self._on_dashboard)
        elif lparam in (win32con.WM_RBUTTONUP, win32con.WM_LBUTTONUP):
            self._show_menu()
        return True

    def _show_menu(self) -> None:
        import win32con
        import win32gui

        menu = win32gui.CreatePopupMenu()
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_DASHBOARD, "Dashboard")
        toggle = "Pause listening" if self.state.listening else "Resume listening"
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_TOGGLE, toggle)
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_LOG, "Open log")
        win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
        win32gui.AppendMenu(menu, win32con.MF_STRING, _ID_QUIT, "Quit Vesper")

        position = win32gui.GetCursorPos()
        # Windows leaves the menu up until the owner window is foregrounded.
        win32gui.SetForegroundWindow(self._hwnd)
        win32gui.TrackPopupMenu(
            menu, win32con.TPM_LEFTALIGN | win32con.TPM_BOTTOMALIGN,
            position[0], position[1], 0, self._hwnd, None,
        )
        win32gui.PostMessage(self._hwnd, win32con.WM_NULL, 0, 0)

    def _on_command(self, hwnd, msg, wparam, lparam):
        choice = wparam & 0xFFFF
        if choice == _ID_DASHBOARD:
            self._safely(self._on_dashboard)
        elif choice == _ID_TOGGLE:
            self.state.listening = not self.state.listening
            self._refresh()
            self._safely(lambda: self._on_toggle(self.state.listening))
        elif choice == _ID_LOG:
            self._safely(self._on_open_log)
        elif choice == _ID_QUIT:
            self._safely(self._on_quit)
        return True

    def _on_close(self, hwnd, msg, wparam, lparam):
        import win32gui

        try:
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (self._hwnd, 0))
        except Exception:
            pass
        win32gui.DestroyWindow(self._hwnd)
        return 0

    def _on_destroy(self, hwnd, msg, wparam, lparam):
        import win32gui

        # The window is going away, so the icons it was drawing with can go too.
        self.destroy_icons()
        win32gui.PostQuitMessage(0)
        return 0

    def _safely(self, action) -> None:
        """A failing menu handler must not kill the pump and strand the icon."""
        try:
            action()
        except Exception as exc:
            self._note(f"tray action failed: {exc}")

    def _note(self, message: str) -> None:
        if self._log is not None:
            try:
                self._log.warn(message)
            except Exception:
                pass
