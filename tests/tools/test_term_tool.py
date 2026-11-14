"""Public contracts for the managed-terminal tool family."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

import ursa.tools as tools_package
from ursa.tools import term_tool
from ursa.tools.terminal import TerminalRenderSnapshot, TerminalSpan

TERM_ID = "Ab12Cd34"


class Manager:
    """Recording manager double shared by the public wrapper tests."""

    def __init__(self, *, output="output\n", screen=True):
        self.terminal = SimpleNamespace(term_id=TERM_ID)
        self.output, self.screen, self.calls = output, screen, []

    async def create(self, command, **kwargs):
        self.calls.append(("create", command, kwargs))
        return self.terminal

    def __getattr__(self, name):
        """Implement manager wrappers while retaining every call for assertions."""

        async def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name in {"contents", "read"}:
                return self.output
            if name == "is_alive":
                return {"is_alive": True}
            if name == "resources":
                return {"term_id": args[0], "status": "running", "pid": 1}
            if name == "wait_for":
                return args[1]
            if name == "wait_screen":
                return kwargs["condition"]
            if name == "cursor":
                return (2, 3)
            if name == "size":
                return (24, 80)
            if name == "render_snapshot":
                return TerminalRenderSnapshot(
                    args[0], (TerminalSpan("screen"),), 1, 6, self.screen
                )
            return 0 if name == "wait" else None

        return method

    def supports_screen(self):
        return self.screen


def runtime(tmp_path: Path):
    return SimpleNamespace(context=SimpleNamespace(workspace=str(tmp_path)))


@pytest.fixture(autouse=True)
def allow_command(monkeypatch):
    def allow(*_):
        return SimpleNamespace(is_safe=True, reason="safe")

    monkeypatch.setattr(term_tool, "assess_command_safety", allow)


def install(monkeypatch, manager=None):
    manager = manager or Manager()
    monkeypatch.setattr(term_tool, "term_manager", manager)
    return manager


def test_public_catalog_and_capability_filtering(monkeypatch):
    base = {tool.name for tool in term_tool._BASE_TERM_TOOLS}
    screen = {tool.name for tool in term_tool._SCREEN_TERM_TOOLS}
    assert not hasattr(term_tool, "term_send_keycode")
    assert not hasattr(tools_package, "term_send_keycode")
    for supported, expected in ((False, base), (True, base | screen)):
        monkeypatch.setattr(
            term_tool.term_manager, "supports_screen", lambda: supported
        )
        assert {
            tool.name for tool in term_tool.get_supported_term_tools()
        } == expected
    assert {tool.name for tool in term_tool.TERM_TOOLS} == base | screen


@pytest.mark.parametrize(
    ("key", "modifiers", "encoded"),
    [
        ("c", ["ctrl"], b"\x03"),
        ("x", ["alt"], b"\x1bx"),
        ("up", ["ctrl"], b"\x1b[1;5A"),
        ("tab", ["shift"], b"\x1b[Z"),
        ("page-down", ["ctrl"], b"\x1b[6;5~"),
        ("f1", None, b"\x1bOP"),
    ],
)
def test_key_encoding_representative_protocols(key, modifiers, encoded):
    assert term_tool.encode_term_key(key, modifiers) == encoded


@pytest.mark.parametrize(
    ("key", "modifiers", "message"),
    [
        ("not-a-key", None, "unknown key"),
        ("a", ["hyper"], "unknown modifier"),
        ("a", ["ctrl", "control"], "duplicate modifier"),
        ("enter", ["ctrl"], "not supported"),
        ("é", ["ctrl"], "representable"),
        (None, None, "non-empty"),
    ],
)
def test_key_encoding_rejects_invalid_input(key, modifiers, message):
    with pytest.raises(ValueError, match=message):
        term_tool.encode_term_key(key, modifiers)


@pytest.mark.parametrize(
    ("output", "session", "expected", "removed"),
    [
        ("quick\n", False, "Terminal contents:\nquick\n", True),
        ("quick\n", True, f"Terminal ID: {TERM_ID}", False),
        ("x" * 21, False, f"Terminal ID: {TERM_ID}", False),
    ],
)
async def test_term_lifecycle_results(
    monkeypatch, tmp_path, output, session, expected, removed
):
    manager = install(monkeypatch, Manager(output=output))
    monkeypatch.setattr(term_tool, "TERM_MAX_BYTES", 20)
    assert (
        await term_tool.term.coroutine(
            "echo", runtime(tmp_path), session=session
        )
        == expected
    )
    assert any(call[0] == "remove" for call in manager.calls) is removed


async def test_term_timeout_cancellation_and_safety(monkeypatch, tmp_path):
    manager = install(monkeypatch)
    real_wait_for = asyncio.wait_for

    async def timeout(awaitable, *_args, **_kwargs):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(term_tool.asyncio, "wait_for", timeout)
    assert (
        await term_tool.term.coroutine("slow", runtime(tmp_path))
        == f"Terminal ID: {TERM_ID}"
    )
    monkeypatch.setattr(term_tool.asyncio, "wait_for", real_wait_for)
    started = asyncio.Event()

    async def waiting(term_id):
        started.set()
        await asyncio.Event().wait()

    manager.wait = waiting
    task = asyncio.create_task(
        term_tool.term.coroutine("slow", runtime(tmp_path))
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.calls[-1] == ("remove", (TERM_ID,), {"terminate": True})

    def deny(*_):
        return SimpleNamespace(is_safe=False, reason="no")

    monkeypatch.setattr(term_tool, "assess_command_safety", deny)
    assert "[UNSAFE]" in await term_tool.term.coroutine(
        "bad", runtime(tmp_path)
    )
    with pytest.raises(ValueError, match="command-execution flags"):
        await term_tool.term.coroutine(
            "echo", runtime(tmp_path), shell=["bash", "-c"]
        )


@pytest.mark.parametrize(
    ("tool", "arguments", "method"),
    [
        (term_tool.term_send_bytes, {"data": [65]}, "send_bytes"),
        (term_tool.term_send_text, {"text": "text"}, "send_text"),
        (term_tool.term_paste_text, {"text": "paste"}, "paste_text"),
        (term_tool.term_send_line, {"line": "line"}, "send_line"),
        (
            term_tool.term_send_key,
            {"key": "c", "modifiers": ["ctrl"]},
            "send_bytes",
        ),
        (term_tool.term_read, {"offset": 1, "lines": 2}, "read"),
        (term_tool.term_is_alive, {}, "is_alive"),
        (term_tool.term_resources, {}, "resources"),
        (
            term_tool.term_wait_for,
            {"pattern": "ready", "timeout": 1},
            "wait_for",
        ),
        (term_tool.term_resize, {"rows": 2, "cols": 3}, "resize"),
        (term_tool.term_cursor, {}, "cursor"),
        (term_tool.term_size, {}, "size"),
    ],
)
async def test_base_wrappers_forward_to_manager(
    monkeypatch, tool, arguments, method
):
    manager = install(monkeypatch)
    result = await tool.ainvoke({"term_id": TERM_ID, **arguments})
    assert manager.calls[-1][0] == method
    assert manager.calls[-1][1][0] == TERM_ID
    assert result is not None


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            term_tool.term_click,
            {"row": 1, "col": 2, "button": "right", "modifiers": ["ctrl"]},
        ),
        (term_tool.term_mouse_down, {"row": 1, "col": 2}),
        (term_tool.term_mouse_up, {"row": 1, "col": 2}),
        (term_tool.term_hover, {"row": 1, "col": 2}),
        (
            term_tool.term_scroll,
            {"row": 1, "col": 2, "delta_y": -1, "delta_x": 1},
        ),
    ],
)
async def test_mouse_wrappers_emit_events(monkeypatch, tool, arguments):
    manager = install(monkeypatch)
    await tool.ainvoke({"term_id": TERM_ID, **arguments})
    call = manager.calls[-1]
    assert call[0] == "mouse_input"
    assert call[1][:3] == (TERM_ID, 1, 2)
    assert call[1][3]


async def test_screen_wait_and_screenshot_contracts(monkeypatch):
    manager = install(monkeypatch)
    assert (
        await term_tool.term_wait_screen.ainvoke({
            "term_id": TERM_ID,
            "condition": "change",
            "bounding_box": [0, 0, 1, 1],
            "include_styling": False,
        })
        == "change"
    )
    assert manager.calls[-1][2] == {
        "condition": "change",
        "bounding_box": (0, 0, 1, 1),
        "include_styling": False,
        "timeout": None,
    }
    result = await term_tool.term_screenshot.coroutine(TERM_ID)
    assert result[0]["text"] == "Screenshot attached."
    assert base64.b64decode(result[1]["base64"]).startswith(b"\x89PNG")
    install(monkeypatch, Manager(screen=False))
    with pytest.raises(ToolException, match="Ghostty"):
        await term_tool.term_screenshot.coroutine(TERM_ID)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (term_tool.term, {"cmd": "x", "shell": ["bash", "-c"]}),
        (term_tool.term_send_bytes, {"term_id": TERM_ID, "data": [256]}),
        (term_tool.term_paste_text, {"term_id": TERM_ID, "text": "bad\n"}),
        (term_tool.term_send_key, {"term_id": TERM_ID, "key": "bad"}),
        (term_tool.term_read, {"term_id": TERM_ID, "offset": -1}),
        (term_tool.term_wait_for, {"term_id": TERM_ID, "pattern": "["}),
        (term_tool.term_wait_screen, {"term_id": TERM_ID, "condition": "bad"}),
        (term_tool.term_resize, {"term_id": TERM_ID, "rows": 0, "cols": 1}),
        (term_tool.term_click, {"term_id": TERM_ID, "row": -1, "col": 0}),
        (
            term_tool.term_scroll,
            {"term_id": TERM_ID, "row": 0, "col": 0, "delta_y": 0},
        ),
    ],
)
async def test_schema_rejects_invalid_inputs(tool, arguments):
    with pytest.raises(ValidationError):
        await tool.ainvoke(arguments)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (term_tool.term_send_bytes, {"data": [1]}),
        (term_tool.term_send_text, {"text": "x"}),
        (term_tool.term_paste_text, {"text": "x"}),
        (term_tool.term_send_line, {"line": "x"}),
        (term_tool.term_send_key, {"key": "x"}),
        (term_tool.term_read, {}),
        (term_tool.term_is_alive, {}),
        (term_tool.term_resources, {}),
        (term_tool.term_close, {}),
        (term_tool.term_wait_for, {"pattern": "x"}),
        (term_tool.term_wait_screen, {}),
        (term_tool.term_resize, {"rows": 1, "cols": 1}),
        (term_tool.term_cursor, {}),
        (term_tool.term_size, {}),
        (term_tool.term_click, {"row": 0, "col": 0}),
        (term_tool.term_mouse_down, {"row": 0, "col": 0}),
        (term_tool.term_mouse_up, {"row": 0, "col": 0}),
        (term_tool.term_hover, {"row": 0, "col": 0}),
        (term_tool.term_scroll, {"row": 0, "col": 0, "delta_y": 1}),
    ],
)
async def test_stale_ids_are_retryable(monkeypatch, tool, arguments):
    class Missing:
        def __getattr__(self, _):
            async def missing(term_id, *_args, **_kwargs):
                raise KeyError(term_id)

            return missing

    monkeypatch.setattr(term_tool, "term_manager", Missing())
    with pytest.raises(ToolException, match="Unknown terminal ID"):
        await tool.ainvoke({"term_id": "Missing1", **arguments})
