import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException
from PIL import Image
from pydantic import ValidationError

import ursa.tools as tools_package
from ursa.tools import term_tool
from ursa.tools.terminal import (
    TerminalRenderSnapshot,
    TerminalSpan,
    TerminalStyle,
    screenshot,
)


def test_term_send_keycode_is_not_publicly_exported():
    assert not hasattr(term_tool, "term_send_keycode")
    assert not hasattr(tools_package, "term_send_keycode")


def test_term_screenshot_is_publicly_exported():
    assert tools_package.term_screenshot is term_tool.term_screenshot


class WrapperTerm:
    term_id = "Ab12Cd34"

    def __init__(self, *, output="output\n", wait=None):
        self.output = output
        self.wait_result = wait
        self.calls = []

    async def wait(self):
        if self.wait_result is not None:
            return await self.wait_result()
        return 0

    async def read(self, **kwargs):
        self.calls.append(("read", kwargs))
        return self.output

    async def contents(self):
        return self.output

    async def send_bytes(self, value):
        self.calls.append(("bytes", value))

    async def send_text(self, value):
        self.calls.append(("text", value))

    async def send_line(self, value):
        self.calls.append(("line", value))

    async def is_alive(self):
        return {"is_alive": True}

    async def resize(self, rows, cols):
        self.calls.append(("resize", rows, cols))

    async def cursor(self):
        return (2, 7)

    async def size(self):
        return (24, 80)


class WrapperManager:
    def __init__(self, terminal):
        self.terminal = terminal
        self.created = []
        self.removed = []

    async def create(self, cmd, **kwargs):
        self.created.append((cmd, kwargs))
        return self.terminal

    def get(self, term_id):
        assert term_id == self.terminal.term_id
        return self.terminal

    async def send_bytes(self, term_id, data):
        await self.get(term_id).send_bytes(data)

    async def send_text(self, term_id, text):
        await self.get(term_id).send_text(text)

    async def send_line(self, term_id, line):
        await self.get(term_id).send_line(line)

    async def read(self, term_id, **kwargs):
        return await self.get(term_id).read(**kwargs)

    async def contents(self, term_id):
        return await self.get(term_id).contents()

    async def wait(self, term_id):
        return await self.get(term_id).wait()

    async def is_alive(self, term_id):
        return await self.get(term_id).is_alive()

    async def resize(self, term_id, rows, cols):
        await self.get(term_id).resize(rows, cols)

    async def cursor(self, term_id):
        return await self.get(term_id).cursor()

    async def size(self, term_id):
        return await self.get(term_id).size()

    async def render_snapshot(self, term_id):
        return await self.get(term_id).render_snapshot()

    async def remove(self, term_id, **kwargs):
        self.removed.append((term_id, kwargs))

    async def wait_for(self, term_id, pattern, timeout):
        return f"{term_id}:{pattern}:{timeout}"


def runtime(tmp_path):
    return SimpleNamespace(context=SimpleNamespace(workspace=str(tmp_path)))


class MissingTerminalManager:
    """Manager double that rejects every well-formed terminal ID."""

    def __getattr__(self, name):
        async def missing(*args, **kwargs):
            raise KeyError(args[0])

        return missing


@pytest.fixture(autouse=True)
def safe_command(monkeypatch):
    async def allow(command, runtime):
        return SimpleNamespace(is_safe=True, reason="safe")

    monkeypatch.setattr(term_tool, "assess_command_safety", allow)


def test_launch_safety_text_contains_complete_benign_configuration(tmp_path):
    result = term_tool._launch_safety_text(
        ["printf", "%s", "hello world"],
        {"ZED": "last", "ALPHA": "first"},
        ["bash", "--noprofile"],
        tmp_path,
    )
    assert result == (
        "COMMAND:\n  argv \"printf %s 'hello world'\"\n"
        "SHELL ARGV:\n  'bash --noprofile'\n"
        f"WORKING DIRECTORY:\n  {str(tmp_path)!r}\n"
        "ENVIRONMENT OVERRIDES:\n"
        "  'ALPHA'='first'\n"
        "  'ZED'='last'"
    )


@pytest.mark.parametrize(
    ("terminal_tool", "arguments"),
    [
        (term_tool.term_send_bytes, {"data": [1]}),
        (term_tool.term_send_text, {"text": "x"}),
        (term_tool.term_send_line, {"line": "x"}),
        (term_tool.term_send_key, {"key": "c", "modifiers": ["ctrl"]}),
        (term_tool.term_read, {}),
        (term_tool.term_is_alive, {}),
        (term_tool.term_wait_for, {"pattern": "ready"}),
        (term_tool.term_resize, {"rows": 24, "cols": 80}),
        (term_tool.term_cursor, {}),
        (term_tool.term_size, {}),
    ],
)
async def test_unknown_terminal_ids_raise_retryable_tool_errors(
    monkeypatch, terminal_tool, arguments
):
    monkeypatch.setattr(term_tool, "term_manager", MissingTerminalManager())

    with pytest.raises(ToolException, match="Unknown terminal ID 'Missing1'"):
        await terminal_tool.ainvoke({"term_id": "Missing1", **arguments})


def test_launch_safety_text_labels_text_command_and_empty_environment(tmp_path):
    result = term_tool._launch_safety_text(
        "echo hello", None, ["bash"], tmp_path
    )
    assert result == (
        "COMMAND:\n  text 'echo hello'\n"
        "SHELL ARGV:\n  'bash'\n"
        f"WORKING DIRECTORY:\n  {str(tmp_path)!r}\n"
        "ENVIRONMENT OVERRIDES:\nnone"
    )


async def test_term_short_command_returns_output_and_removes_session(
    monkeypatch, tmp_path
):
    terminal = WrapperTerm(output="quick\n")
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    result = await term_tool.term.coroutine(
        ["printf", "quick"],
        runtime=runtime(tmp_path),
        env={"A": "B"},
        shell=["bash"],
    )
    assert result == "Terminal contents:\nquick\n"
    assert manager.created == [
        (
            ["printf", "quick"],
            {"env": {"A": "B"}, "shell": ["bash"], "cwd": tmp_path},
        )
    ]
    assert manager.removed == [(terminal.term_id, {"terminate": True})]


async def test_term_session_mode_returns_id_without_waiting(
    monkeypatch, tmp_path
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assert (
        await term_tool.term.coroutine(
            "interactive", runtime=runtime(tmp_path), session=True
        )
        == "Terminal ID: Ab12Cd34"
    )
    assert manager.removed == []


async def test_term_falls_back_to_id_on_timeout(monkeypatch, tmp_path):
    async def slow_wait():
        await asyncio.sleep(1)
        return 0

    terminal = WrapperTerm(wait=slow_wait)
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    monkeypatch.setattr(term_tool, "TERM_TIMEOUT", 0.001)
    assert await term_tool.term.coroutine(
        "slow", runtime=runtime(tmp_path)
    ) == ("Terminal ID: Ab12Cd34")
    assert manager.removed == []


async def test_term_cancellation_while_waiting_cleans_session(
    monkeypatch, tmp_path
):
    waiting = asyncio.Event()

    async def blocked_wait():
        waiting.set()
        await asyncio.Event().wait()

    terminal = WrapperTerm(wait=blocked_wait)
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    task = asyncio.create_task(
        term_tool.term.coroutine("slow", runtime=runtime(tmp_path))
    )
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.removed == [(terminal.term_id, {"terminate": True})]


async def test_term_session_cancellation_before_id_cleans_session(
    monkeypatch, tmp_path
):
    terminal = WrapperTerm()

    class CancellingManager(WrapperManager):
        async def create(self, cmd, **kwargs):
            created = await super().create(cmd, **kwargs)
            asyncio.current_task().cancel()
            return created

    manager = CancellingManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    with pytest.raises(asyncio.CancelledError):
        await term_tool.term.coroutine(
            "interactive", runtime=runtime(tmp_path), session=True
        )
    assert manager.removed == [(terminal.term_id, {"terminate": True})]


@pytest.mark.parametrize(
    "output,max_bytes,max_lines",
    [
        ("x" * 11, 10, 100),
        ("a\nb\nc\n", 100, 2),
        ("é" * 6, 11, 100),
    ],
)
async def test_term_falls_back_to_id_on_output_limits(
    monkeypatch, tmp_path, output, max_bytes, max_lines
):
    terminal = WrapperTerm(output=output)
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    monkeypatch.setattr(term_tool, "TERM_MAX_BYTES", max_bytes)
    monkeypatch.setattr(term_tool, "TERM_MAX_LINES", max_lines)
    assert await term_tool.term.coroutine(
        "large", runtime=runtime(tmp_path)
    ) == ("Terminal ID: Ab12Cd34")
    assert manager.removed == []


@pytest.mark.parametrize(
    "output,max_bytes,max_lines",
    [("x" * 10, 10, 100), ("a\nb\n", 100, 2), ("é" * 5, 10, 100)],
)
async def test_term_falls_back_exactly_at_output_limits(
    monkeypatch, tmp_path, output, max_bytes, max_lines
):
    terminal = WrapperTerm(output=output)
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    monkeypatch.setattr(term_tool, "TERM_MAX_BYTES", max_bytes)
    monkeypatch.setattr(term_tool, "TERM_MAX_LINES", max_lines)
    assert (
        await term_tool.term.coroutine("bounded", runtime=runtime(tmp_path))
        == f"Terminal ID: {terminal.term_id}"
    )
    assert manager.removed == []


async def test_term_blocks_unsafe_command_before_creating_terminal(
    monkeypatch, tmp_path
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assessed = []

    async def reject(command, runtime):
        assessed.append(command)
        return SimpleNamespace(is_safe=False, reason="test rejection")

    monkeypatch.setattr(term_tool, "assess_command_safety", reject)
    result = await term_tool.term.coroutine(
        ["bad", "argument with spaces"], runtime=runtime(tmp_path)
    )
    assert "[UNSAFE]" in result
    assert "test rejection" in result
    assert "COMMAND:\n  argv \"bad 'argument with spaces'\"" in assessed[0]
    assert f"WORKING DIRECTORY:\n  {str(tmp_path)!r}" in assessed[0]
    assert manager.created == []


@pytest.mark.parametrize(
    "flag", ["-c", "--COMMAND", "-Command", "-EncodedCommand"]
)
async def test_term_rejects_shell_execution_flags_before_assessment(
    monkeypatch, tmp_path, flag
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assessed = []

    async def record_assessment(text, runtime):
        assessed.append(text)

    monkeypatch.setattr(term_tool, "assess_command_safety", record_assessment)
    with pytest.raises(ValueError) as error:
        await term_tool.term.coroutine(
            "safe", runtime=runtime(tmp_path), shell=["bash", flag, "payload"]
        )
    assert str(error.value) == (
        "shell must not contain command-execution flags; pass the command "
        f"through cmd instead (found {flag!r})"
    )
    assert assessed == []
    assert manager.created == []


def test_validate_shell_reports_multiple_conflicting_flags():
    with pytest.raises(ValueError) as error:
        term_tool._validate_shell(["bash", "-c", "--COMMAND"])
    assert str(error.value).endswith("(found '-c', '--COMMAND')")


def test_empty_shell_reports_actionable_validation_error():
    with pytest.raises(ValueError) as error:
        term_tool._validate_shell([])
    assert str(error.value) == "shell must contain at least one argument"


@pytest.mark.parametrize(
    ("shell", "env", "expected"),
    [
        (["bash", "--noprofile"], None, "bash --noprofile"),
        (["bash"], {"BASH_ENV": "/tmp/payload"}, "'BASH_ENV'='/tmp/payload'"),
    ],
)
async def test_term_safety_assesses_shell_and_environment_inputs(
    monkeypatch, tmp_path, shell, env, expected
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assessed = []

    async def reject(text, runtime):
        assessed.append(text)
        return SimpleNamespace(is_safe=False, reason="configuration")

    monkeypatch.setattr(term_tool, "assess_command_safety", reject)
    result = await term_tool.term.coroutine(
        "echo okay",
        runtime=runtime(tmp_path),
        shell=shell,
        env=env,
    )
    assert result.startswith("[UNSAFE]")
    assert expected in assessed[0]
    assert "SHELL ARGV:" in assessed[0]
    assert "ENVIRONMENT OVERRIDES:" in assessed[0]
    assert manager.created == []


async def test_send_read_and_state_wrappers(monkeypatch):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    term_id = terminal.term_id
    assert await term_tool.term_send_bytes.coroutine(term_id, b"ab") == (
        "Sent 2 bytes to terminal Ab12Cd34"
    )
    assert await term_tool.term_send_text.coroutine(term_id, "txt") == (
        "Sent text to terminal Ab12Cd34"
    )
    assert await term_tool.term_send_line.coroutine(term_id, "line") == (
        "Sent line to terminal Ab12Cd34"
    )
    assert (
        await term_tool.term_read.coroutine(term_id, offset=2, lines=4)
        == "output\n"
    )
    assert await term_tool.term_read.coroutine(term_id) == "output\n"
    assert await term_tool.term_is_alive.coroutine(term_id) == {
        "is_alive": True
    }
    assert terminal.calls[:5] == [
        ("bytes", b"ab"),
        ("text", "txt"),
        ("line", "line"),
        ("read", {"offset": 2, "lines": 4}),
        ("read", {"offset": 0, "lines": None}),
    ]


async def test_send_bytes_accepts_json_integer_array_and_rejects_bad_values(
    monkeypatch,
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assert (
        await term_tool.term_send_bytes.coroutine(
            terminal.term_id, [0, 127, 255]
        )
        == "Sent 3 bytes to terminal Ab12Cd34"
    )
    assert terminal.calls == [("bytes", b"\x00\x7f\xff")]
    for bad in ([256], [-1], [True], [1.5]):
        with pytest.raises(ValueError, match="integers from 0 through 255"):
            await term_tool.term_send_bytes.coroutine(terminal.term_id, bad)


@pytest.mark.parametrize(
    ("key", "modifiers", "expected"),
    [
        ("c", ["ctrl"], b"\x03"),
        ("_", ["CONTROL"], b"\x1f"),
        ("x", ["Alt"], b"\x1bx"),
        ("a", ["SHIFT"], b"A"),
        ("d", ["super"], b"\x1b[100;9u"),
        ("up", ["ctrl"], b"\x1b[1;5A"),
        ("ArrowLeft", ["option"], b"\x1b[1;3D"),
        ("TAB", ["shift"], b"\x1b[Z"),
        ("tab", None, b"\t"),
        ("escape", ["alt"], b"\x1b\x1b"),
        ("ESC", None, b"\x1b"),
        ("UP", None, b"\x1b[A"),
        ("page_up", None, b"\x1b[5~"),
        ("page-down", ["ctrl"], b"\x1b[6;5~"),
        ("f1", None, b"\x1bOP"),
        ("F1", ["ctrl"], b"\x1b[1;5P"),
    ],
)
def test_encode_term_key_variants(key, modifiers, expected):
    assert term_tool.encode_term_key(key, modifiers) == expected


@pytest.mark.parametrize(
    ("key", "modifiers", "message"),
    [
        ("not-a-key", None, "unknown key"),
        ("a", ["hyper"], "unknown modifier"),
        ("a", ["ctrl", "control"], "duplicate modifier"),
        ("d", ["cmd", "meta"], "duplicate modifier"),
        ("", None, "non-empty string"),
        ("é", ["ctrl"], "no representable ASCII control code"),
    ],
)
def test_encode_term_key_rejects_invalid_inputs(key, modifiers, message):
    with pytest.raises(ValueError, match=message):
        term_tool.encode_term_key(key, modifiers)


def test_encode_term_key_validation_messages_are_specific():
    with pytest.raises(ValueError) as modifier_error:
        term_tool.encode_term_key("a", [1])
    assert str(modifier_error.value) == "modifiers must be strings"
    with pytest.raises(ValueError) as named_error:
        term_tool.encode_term_key("escape", ["ctrl"])
    assert str(named_error.value) == (
        "modifiers are not supported for named key 'escape'"
    )
    with pytest.raises(ValueError) as key_error:
        term_tool.encode_term_key(None)
    assert str(key_error.value) == "key must be a non-empty string"


async def test_term_send_key_encodes_and_forwards_bytes(monkeypatch):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    assert (
        await term_tool.term_send_key.coroutine(terminal.term_id, "c", ["ctrl"])
        == "Sent key 'c' to terminal Ab12Cd34"
    )
    assert terminal.calls == [("bytes", b"\x03")]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            term_tool.term,
            {"cmd": "echo unsafe schema", "shell": ["bash", "-c"]},
        ),
        (
            term_tool.term_send_bytes,
            {"term_id": "Ab12Cd34", "data": [256]},
        ),
        (
            term_tool.term_send_key,
            {"term_id": "Ab12Cd34", "key": "not-a-key"},
        ),
        (
            term_tool.term_send_key,
            {"term_id": "Ab12Cd34", "key": "a", "modifiers": ["hyper"]},
        ),
        (
            term_tool.term_read,
            {"term_id": "Ab12Cd34", "offset": -1},
        ),
        (
            term_tool.term_read,
            {"term_id": "Ab12Cd34", "lines": 0},
        ),
        (
            term_tool.term_wait_for,
            {"term_id": "Ab12Cd34", "pattern": "["},
        ),
        (
            term_tool.term_resize,
            {"term_id": "Ab12Cd34", "rows": 0, "cols": 80},
        ),
    ],
)
async def test_invalid_tool_inputs_raise_pydantic_errors_before_invocation(
    tool, arguments
):
    with pytest.raises(ValidationError):
        await tool.ainvoke(arguments)


async def test_cross_field_key_error_is_a_retryable_pydantic_error():
    with pytest.raises(ValidationError, match="modifiers are not supported"):
        await term_tool.term_send_key.ainvoke({
            "term_id": "Ab12Cd34",
            "key": "enter",
            "modifiers": ["ctrl"],
        })


@pytest.mark.parametrize("key", ["ab", "\n"])
async def test_term_send_key_rejects_non_key_strings(key):
    with pytest.raises(ValidationError) as error:
        await term_tool.term_send_key.ainvoke({
            "term_id": "Ab12Cd34",
            "key": key,
        })
    assert "unknown key" in str(error.value)
    assert error.value.errors()[0]["loc"] == ("key",)


@pytest.mark.parametrize("key", ["page_up", "page-down"])
async def test_term_send_key_schema_accepts_named_key_separators(
    monkeypatch, key
):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    result = await term_tool.term_send_key.ainvoke({
        "term_id": terminal.term_id,
        "key": key,
    })
    assert result == f"Sent key {key!r} to terminal {terminal.term_id}"
    assert terminal.calls[0][0] == "bytes"


async def test_term_send_key_unknown_name_has_actionable_schema_error():
    with pytest.raises(ValidationError) as error:
        await term_tool.term_send_key.ainvoke({
            "term_id": "Ab12Cd34",
            "key": "definitely-unknown",
        })
    assert "unknown key: 'definitely-unknown'" in str(error.value)


async def test_term_send_key_invalid_modifier_is_located_on_modifiers():
    with pytest.raises(ValidationError) as error:
        await term_tool.term_send_key.ainvoke({
            "term_id": "Ab12Cd34",
            "key": "a",
            "modifiers": ["hyper"],
        })
    assert error.value.errors()[0]["loc"] == ("modifiers",)


async def test_wait_for_invalid_regex_has_actionable_schema_error():
    with pytest.raises(ValidationError) as error:
        await term_tool.term_wait_for.ainvoke({
            "term_id": "Ab12Cd34",
            "pattern": "[",
        })
    assert "invalid regular expression" in str(error.value)


async def test_wait_resize_cursor_and_size_wrappers(monkeypatch):
    terminal = WrapperTerm()
    manager = WrapperManager(terminal)
    monkeypatch.setattr(term_tool, "term_manager", manager)
    term_id = terminal.term_id
    assert await term_tool.term_wait_for.coroutine(term_id, "ready", 1.5) == (
        "Ab12Cd34:ready:1.5"
    )
    assert await term_tool.term_resize.coroutine(term_id, 40, 120) == (
        "Resized terminal Ab12Cd34 to 40 rows by 120 columns"
    )
    assert await term_tool.term_cursor.coroutine(term_id) == (2, 7)
    assert await term_tool.term_size.coroutine(term_id) == (24, 80)
    assert ("resize", 40, 120) in terminal.calls


async def test_term_screenshot_returns_styled_png(monkeypatch, tmp_path):
    snapshot = TerminalRenderSnapshot(
        term_id="Ab12Cd34",
        spans=(
            TerminalSpan(
                "styled output",
                TerminalStyle(foreground=(255, 0, 0), bold=True),
            ),
        ),
        rows=4,
        cols=20,
        screen=True,
    )

    class ScreenshotManager:
        async def render_snapshot(self, term_id):
            assert term_id == "Ab12Cd34"
            return snapshot

    monkeypatch.setattr(term_tool, "term_manager", ScreenshotManager())
    result = await term_tool.term_screenshot.coroutine("Ab12Cd34")

    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Screenshot attached."
    assert result[1]["type"] == "image"
    assert result[1]["mime_type"] == "image/png"
    png = base64.b64decode(result[1]["base64"])
    with Image.open(BytesIO(png)) as image:
        assert image.width > 0
        assert image.height > 0
        assert any(
            red > green * 2 and red > blue * 2
            for red, green, blue in image.convert("RGB").getdata()
        )
    assert not list(tmp_path.glob("term-*.png"))


async def test_term_screenshot_rejects_process_terminal(monkeypatch, tmp_path):
    snapshot = TerminalRenderSnapshot(
        term_id="Ab12Cd34",
        spans=(TerminalSpan("plain output"),),
    )

    class ScreenshotManager:
        async def render_snapshot(self, term_id):
            assert term_id == "Ab12Cd34"
            return snapshot

    monkeypatch.setattr(term_tool, "term_manager", ScreenshotManager())
    with pytest.raises(ToolException, match="require the Ghostty backend"):
        await term_tool.term_screenshot.coroutine("Ab12Cd34")


async def test_screenshot_settle_waits_for_new_stable_content(monkeypatch):
    blank = TerminalRenderSnapshot(
        "Ab12Cd34",
        (TerminalSpan(" " * 80),),
        rows=24,
        cols=80,
        screen=True,
    )
    content = TerminalRenderSnapshot(
        "Ab12Cd34",
        (TerminalSpan("Python 3.11 >>>"),),
        rows=24,
        cols=80,
        screen=True,
    )

    class EvolvingManager:
        calls = 0

        async def render_snapshot(self, term_id):
            assert term_id == "Ab12Cd34"
            self.calls += 1
            return blank if self.calls == 1 else content

    manager = EvolvingManager()
    result = await screenshot.settled_screen_snapshot(
        manager, "Ab12Cd34", interval=0
    )

    assert result is content
    assert manager.calls == 3


async def test_screenshot_settle_bounds_truly_blank_screen(monkeypatch):
    blank = TerminalRenderSnapshot(
        "Ab12Cd34", (TerminalSpan(" " * 80),), rows=24, cols=80, screen=True
    )

    class BlankManager:
        async def render_snapshot(self, term_id):
            assert term_id == "Ab12Cd34"
            return blank

    manager = BlankManager()
    loop = asyncio.get_running_loop()
    started = loop.time()

    result = await screenshot.settled_screen_snapshot(
        manager, "Ab12Cd34", timeout=0.01, interval=0.001
    )

    assert result is blank
    assert loop.time() - started < 0.2


@pytest.mark.parametrize(
    ("screen", "expected_names"),
    [
        (
            False,
            {
                "term",
                "term_send_bytes",
                "term_send_text",
                "term_send_line",
                "term_send_key",
                "term_read",
                "term_is_alive",
                "term_wait_for",
            },
        ),
        (
            True,
            {
                "term",
                "term_send_bytes",
                "term_send_text",
                "term_send_line",
                "term_send_key",
                "term_read",
                "term_is_alive",
                "term_wait_for",
                "term_resize",
                "term_cursor",
                "term_size",
                "term_screenshot",
            },
        ),
    ],
)
def test_get_supported_term_tools_filters_screen_capabilities(
    monkeypatch, screen, expected_names
):
    monkeypatch.setattr(
        term_tool.term_manager, "supports_screen", lambda: screen
    )
    first = term_tool.get_supported_term_tools()
    second = term_tool.get_supported_term_tools()
    assert {tool.name for tool in first} == expected_names
    assert first is not second
    assert first == second
    assert term_tool.term_send_key in term_tool.TERM_TOOLS
    assert all(
        tool.name != "term_send_keycode" for tool in term_tool.TERM_TOOLS
    )
