import ursa.util.crossplatform as crossplatform


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


def test_platform_clipboard_prefers_macos_pbcopy(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(crossplatform.sys, "platform", "darwin")
    monkeypatch.setattr(
        crossplatform.shutil,
        "which",
        lambda name: "/usr/bin/pbcopy" if name == "pbcopy" else None,
    )

    assert crossplatform.platform_clipboard() == ["pbcopy"]


def test_platform_clipboard_prefers_windows_clip(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(crossplatform.sys, "platform", "win32")
    monkeypatch.setattr(
        crossplatform.shutil,
        "which",
        lambda name: "C:/Windows/System32/clip.exe" if name == "clip" else None,
    )

    assert crossplatform.platform_clipboard() == ["clip"]


def test_platform_clipboard_prefers_wayland(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(crossplatform.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        crossplatform.shutil,
        "which",
        lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None,
    )

    assert crossplatform.platform_clipboard() == ["wl-copy"]


def test_platform_clipboard_prefers_xclip_then_xsel(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(crossplatform.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        crossplatform.shutil,
        "which",
        lambda name: "/usr/bin/xclip" if name == "xclip" else None,
    )

    assert crossplatform.platform_clipboard() == [
        "xclip",
        "-selection",
        "clipboard",
    ]

    monkeypatch.setattr(
        crossplatform.shutil,
        "which",
        lambda name: "/usr/bin/xsel" if name == "xsel" else None,
    )

    assert crossplatform.platform_clipboard() == [
        "xsel",
        "--clipboard",
        "--input",
    ]


def test_platform_clipboard_returns_none_when_unavailable(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(crossplatform.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(crossplatform.shutil, "which", lambda name: None)

    assert crossplatform.platform_clipboard() is None


def test_platform_clipboard_returns_none_in_ssh_session(monkeypatch):
    monkeypatch.delenv("URSA_CLIPBOARD", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")

    assert crossplatform.platform_clipboard() is None
