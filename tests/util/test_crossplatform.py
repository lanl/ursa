import pytest

import ursa.util.crossplatform as crossplatform


@pytest.fixture
def clipboard_discovery(monkeypatch):
    for name in ("URSA_CLIPBOARD", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)

    def configure(platform, executables):
        monkeypatch.setattr(crossplatform.sys, "platform", platform)
        monkeypatch.setattr(crossplatform.shutil, "which", executables.get)

    return configure


def test_kitty_keyboard_support_uses_terminal_identity(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("ZELLIJ", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")

    assert crossplatform.expects_kitty_keyboard()


def test_kitty_keyboard_support_fails_closed_for_multiplexers(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,1,0")

    assert not crossplatform.expects_kitty_keyboard()


def test_kitty_keyboard_support_falls_back_to_terminfo_name(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("ZELLIJ", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    for name in crossplatform.KITTY_KEYBOARD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-kitty")

    assert crossplatform.expects_kitty_keyboard()


def test_kitty_keyboard_support_uses_vendor_session_identity(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("ZELLIJ", raising=False)
    monkeypatch.setenv("ALACRITTY_WINDOW_ID", "42")

    assert crossplatform.expects_kitty_keyboard()


def test_kitty_keyboard_support_fails_closed_for_unknown_terminal(monkeypatch):
    for name in (
        *crossplatform.KITTY_KEYBOARD_ENV_VARS,
        "TMUX",
        "ZELLIJ",
        "TERM_PROGRAM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    assert not crossplatform.expects_kitty_keyboard()


def test_copy_to_clipboard_runs_platform_tool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        crossplatform, "platform_clipboard", lambda: ["fake-copy"]
    )
    monkeypatch.setattr(
        crossplatform.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert crossplatform.copy_to_clipboard("hello")
    assert calls[0][0][0] == ["fake-copy"]
    assert calls[0][1]["input"] == "hello"
    assert calls[0][1]["timeout"] == 2


def test_copy_to_clipboard_returns_false_without_tool(monkeypatch):
    monkeypatch.setattr(crossplatform, "platform_clipboard", lambda: None)

    assert not crossplatform.copy_to_clipboard("hello")


def test_copy_to_clipboard_returns_false_on_failure(monkeypatch):
    monkeypatch.setattr(
        crossplatform, "platform_clipboard", lambda: ["fake-copy"]
    )

    def fail(*args, **kwargs):
        raise crossplatform.subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(crossplatform.subprocess, "run", fail)

    assert not crossplatform.copy_to_clipboard("hello")


def test_platform_clipboard_uses_env_override(monkeypatch):
    monkeypatch.setenv("URSA_CLIPBOARD", "/custom/clip --flag value")

    assert crossplatform.platform_clipboard() == [
        "/custom/clip",
        "--flag",
        "value",
    ]


def test_platform_clipboard_prefers_macos_pbcopy(clipboard_discovery):
    clipboard_discovery("darwin", {"pbcopy": "/usr/bin/pbcopy"})

    assert crossplatform.platform_clipboard() == ["pbcopy"]


def test_platform_clipboard_prefers_windows_clip(clipboard_discovery):
    clipboard_discovery("win32", {"clip": "C:/Windows/System32/clip.exe"})

    assert crossplatform.platform_clipboard() == ["clip"]


def test_platform_clipboard_prefers_wayland(monkeypatch, clipboard_discovery):
    clipboard_discovery("linux", {"wl-copy": "/usr/bin/wl-copy"})
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)

    assert crossplatform.platform_clipboard() == ["wl-copy"]


def test_platform_clipboard_prefers_xclip_then_xsel(
    monkeypatch, clipboard_discovery
):
    clipboard_discovery("linux", {"xclip": "/usr/bin/xclip"})
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    assert crossplatform.platform_clipboard() == [
        "xclip",
        "-selection",
        "clipboard",
    ]

    clipboard_discovery("linux", {"xsel": "/usr/bin/xsel"})

    assert crossplatform.platform_clipboard() == [
        "xsel",
        "--clipboard",
        "--input",
    ]


def test_platform_clipboard_returns_none_when_unavailable(
    monkeypatch, clipboard_discovery
):
    clipboard_discovery("linux", {})
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    assert crossplatform.platform_clipboard() is None


def test_platform_clipboard_returns_none_in_ssh_session(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")

    assert crossplatform.platform_clipboard() is None
