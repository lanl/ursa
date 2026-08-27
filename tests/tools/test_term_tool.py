import asyncio
from types import SimpleNamespace

import pytest

from ursa.tools import term_tool


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

    async def send_keycode(self, value):
        self.calls.append(("keycode", value))

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

    async def send_keycode(self, term_id, keycode):
        await self.get(term_id).send_keycode(keycode)

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

    async def remove(self, term_id, **kwargs):
        self.removed.append((term_id, kwargs))

    async def wait_for(self, term_id, pattern, timeout):
        return f"{term_id}:{pattern}:{timeout}"


def runtime(tmp_path):
    return SimpleNamespace(context=SimpleNamespace(workspace=str(tmp_path)))


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
    assert await term_tool.term_send_keycode.coroutine(term_id, 3) == (
        "Sent keycode 3 to terminal Ab12Cd34"
    )
    assert (
        await term_tool.term_read.coroutine(term_id, offset=2, lines=4)
        == "output\n"
    )
    assert await term_tool.term_read.coroutine(term_id) == "output\n"
    assert await term_tool.term_is_alive.coroutine(term_id) == {
        "is_alive": True
    }
    assert terminal.calls[:6] == [
        ("bytes", b"ab"),
        ("text", "txt"),
        ("line", "line"),
        ("keycode", 3),
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
                "term_send_keycode",
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
                "term_send_keycode",
                "term_send_key",
                "term_read",
                "term_is_alive",
                "term_wait_for",
                "term_resize",
                "term_cursor",
                "term_size",
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
