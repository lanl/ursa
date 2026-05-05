import json
import textwrap
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ursa.util.events import DEFAULT_EVENT_NAME


class HITLLogEventHandler(AsyncCallbackHandler):
    """Render structured agent/tool progress events to the REPL console."""

    def __init__(self, console: Console, workspace: Path):
        self.console = console
        self.workspace = Path(workspace).resolve()
        self._last_agent: str | None = None
        self._inflight_tools: dict[Any, dict[str, Any]] = {}
        self.emitted_any = False

    @property
    def ignore_llm(self) -> bool:
        return True

    def _clean(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _agent_key(self, value: Any) -> str:
        raw = self._clean(value).lower().replace("_", "").replace(" ", "")
        return {
            "planningagent": "planner",
            "plan": "planner",
            "planner": "planner",
            "hypothesizeragent": "hypothesizer",
            "hypothesize": "hypothesizer",
            "hypothesizer": "hypothesizer",
            "acquisitionagent": "acquisition",
            "acquisition": "acquisition",
            "executor": "executor",
            "executionagent": "executor",
            "execution": "executor",
        }.get(raw, raw)

    def _agent_title(self, agent: str) -> str:
        return {
            "planner": "Plan",
            "hypothesizer": "Hypothesize",
            "acquisition": "Acquire",
            "executor": "Execute",
        }.get(agent, agent or "Progress")

    def _agent_icon(self, agent: str, stage: str, data: dict[str, Any]) -> str:
        if agent == "planner":
            if stage == "reflect_result":
                return "✅" if data.get("approved") else "🔁"
            return {
                "generate": "📐",
                "reflect": "🔍",
            }.get(stage, "🗺️")
        if agent == "hypothesizer":
            return {
                "generate": "✨",
                "generate_result": "💡",
                "critique": "🔬",
                "critique_result": "🧪",
                "competitor": "🧭",
                "competitor_result": "🗣️",
                "finalize": "🛠️",
                "finalize_result": "⭐",
                "summarize": "📝",
                "summarize_result": "📚",
            }.get(stage, "💡")
        if agent == "acquisition":
            return "📚"
        if agent == "executor":
            return "⚙️"
        return "🔹"

    def _tool_icon(
        self,
        tool: str,
        stage: str,
        phase: str | None,
        data: dict[str, Any],
    ) -> str:
        if tool == "read_file":
            return "📖"
        if tool == "write_code":
            if phase == "error":
                return "✖"
            return "✏️" if phase == "start" else "✅"
        if tool == "run_command":
            if stage == "safety_check":
                return "🛡️" if data.get("safe") else "⚠️"
            if phase == "error":
                return "✖"
            return "▶" if phase == "start" else "✔"
        return "🔧"

    def _display_path(self, value: Any) -> str | None:
        raw = self._clean(value)
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            return str(resolved.relative_to(self.workspace))
        except ValueError:
            return str(resolved)

    def _wrap(self, value: Any, *, max_lines: int = 3) -> list[str]:
        text = str(value or "").strip()
        if not text or max_lines <= 0:
            return []

        width = max(24, self.console.width - 8)
        wrapped: list[str] = []
        for line in text.splitlines():
            pieces = textwrap.wrap(
                line.strip(),
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped.extend(pieces or [""])

        trimmed = [line for line in wrapped if line]
        if len(trimmed) <= max_lines:
            return trimmed
        return trimmed[: max_lines - 1] + [trimmed[max_lines - 1] + " …"]

    def _print_agent_rule(self, agent: str) -> None:
        title = self._agent_title(agent)
        self.console.print(Rule(f"[emph]{title}[/]", style="cyan"))

    def _print_agent_event(self, data: dict[str, Any]) -> None:
        agent = self._agent_key(data.get("agent"))
        message = self._clean(data.get("message"))
        if not agent or not message:
            return

        if self._last_agent != agent:
            self._print_agent_rule(agent)
            self._last_agent = agent

        stage = self._clean(data.get("stage"))
        phase = self._clean(data.get("phase"))
        style = "error" if phase == "error" else "emph"
        icon = self._agent_icon(agent, stage, data)
        self.console.print(f"[{style}]{icon} {message}[/]")
        self.emitted_any = True

        preview = data.get("reason") if stage == "reflect_result" else None
        if preview is None:
            preview = data.get("preview")
        for line in self._wrap(preview, max_lines=4):
            self.console.print(f"[dim]  {line}[/]")
        if output_path := self._display_path(data.get("output_path")):
            self.console.print(f"[dim]  {output_path}[/]")

    def _print_tool_event(self, data: dict[str, Any]) -> None:
        tool = self._clean(data.get("tool"))
        message = self._clean(data.get("message"))
        if not tool or not message:
            return

        stage = self._clean(data.get("stage"))
        phase = self._clean(data.get("phase")) or None
        icon = self._tool_icon(tool, stage, phase, data)
        style = "emph"
        detail: str | None = None

        if tool == "read_file":
            detail = self._display_path(data.get("path"))
            style = "success"
        elif tool == "write_code":
            detail = self._display_path(data.get("path") or data.get("filename"))
            style = "error" if phase == "error" else "success"
        elif tool == "run_command":
            detail = self._clean(data.get("query"))
            if stage == "safety_check":
                style = "success" if data.get("safe") else "warn"
            elif phase == "error":
                style = "error"
            else:
                style = "success" if phase == "end" else "emph"
        elif phase == "error":
            style = "error"

        if detail:
            self.console.print(f"[{style}]{icon} {message}:[/] {detail}")
        else:
            self.console.print(f"[{style}]{icon} {message}[/]")
        self.emitted_any = True

        if tool == "run_command" and stage == "execute" and phase == "end":
            parts: list[str] = []
            if isinstance(data.get("returncode"), int):
                parts.append(f"exit {data['returncode']}")
            if isinstance(data.get("stdout_chars"), int):
                parts.append(f"stdout {data['stdout_chars']} chars")
            if isinstance(data.get("stderr_chars"), int):
                parts.append(f"stderr {data['stderr_chars']} chars")
            if parts:
                self.console.print(f"[dim]  {', '.join(parts)}[/]")
        elif tool == "run_command" and stage == "safety_check" and not data.get(
            "safe", True
        ):
            for line in self._wrap(data.get("reason"), max_lines=2):
                self.console.print(f"[dim]  {line}[/]")

    def _tool_name(self, serialized: Any) -> str:
        if isinstance(serialized, dict):
            return self._clean(serialized.get("name")) or "tool"
        return self._clean(serialized) or "tool"

    def _tool_input_payload(self, data: Any, input_str: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            return data
        if not isinstance(input_str, str):
            return {}
        stripped = input_str.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw_input": input_str}
        return parsed if isinstance(parsed, dict) else {"raw_input": input_str}

    def _truncate_block(self, value: Any, *, max_lines: int = 8) -> str:
        lines = str(value or "").splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join([*lines[: max_lines - 1], "..."])

    def _coerce_run_command_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            if payload.startswith("STDOUT:\n") and "\nSTDERR:\n" in payload:
                stdout, stderr = payload[len("STDOUT:\n") :].split(
                    "\nSTDERR:\n", 1
                )
                return {"stdout": stdout, "stderr": stderr}
            return {"result": payload}
        if isinstance(payload, dict):
            return payload
        return {"result": payload}

    def _print_tool_panel(
        self,
        body: Any,
        *,
        title: str,
        border_style: str,
    ) -> None:
        self.console.print(
            Panel(
                Text(self._truncate_block(body), no_wrap=False),
                title=title,
                border_style=border_style,
            )
        )

    def _print_run_command_start(self, query: str | None) -> None:
        if cleaned := self._clean(query):
            self.console.print(f"[emph]▶ Running command:[/] {cleaned}")
        else:
            self.console.print("[emph]▶ Running command[/]")
        self.emitted_any = True

    def _print_run_command_end(
        self,
        payload: Any,
        *,
        query: str | None = None,
        success: bool = True,
    ) -> None:
        query_text = self._clean(query)
        result = self._coerce_run_command_payload(payload)
        raw_result = self._clean(result.get("result"))
        if raw_result.startswith("[UNSAFE]"):
            prefix = (
                f"[warn]⚠ Command blocked:[/] {query_text}"
                if query_text
                else "[warn]⚠ Command blocked[/]"
            )
            self.console.print(prefix)
            for line in self._wrap(raw_result, max_lines=3):
                self.console.print(f"[dim]  {line}[/]")
            self.emitted_any = True
            return

        prefix = (
            f"[success]✔ Command finished:[/] {query_text}"
            if success and query_text
            else "[success]✔ Command finished[/]"
            if success
            else f"[error]✖ Command failed:[/] {query_text}"
            if query_text
            else "[error]✖ Command failed[/]"
        )
        self.console.print(prefix)
        self.emitted_any = True

        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if self._clean(stdout):
            self._print_tool_panel(
                stdout,
                title="stdout",
                border_style="green",
            )
        if self._clean(stderr):
            self._print_tool_panel(
                stderr,
                title="stderr",
                border_style="red",
            )

    def _print_read_file_start(self, path: str | None) -> None:
        detail = self._display_path(path)
        if detail:
            self.console.print(f"[success]📖 Reading file:[/] {detail}")
        else:
            self.console.print("[success]📖 Reading file[/]")
        self.emitted_any = True

    def _print_read_file_end(self, payload: Any, *, path: str | None = None) -> None:
        detail = self._display_path(path)
        if detail:
            self.console.print(f"[success]📖 File read:[/] {detail}")
        else:
            self.console.print("[success]📖 File read[/]")
        self.emitted_any = True
        if text := self._clean(payload):
            self._print_tool_panel(
                text,
                title="read_file",
                border_style="green",
            )

    def _print_write_code_start(self, filename: str | None, path: str | None) -> None:
        detail = self._display_path(path or filename)
        if detail:
            self.console.print(f"[success]✏️ Writing file:[/] {detail}")
        else:
            self.console.print("[success]✏️ Writing file[/]")
        self.emitted_any = True

    def _print_write_code_end(
        self,
        payload: Any,
        *,
        filename: str | None,
        path: str | None,
        success: bool,
    ) -> None:
        detail = self._display_path(path or filename)
        message = self._clean(payload)
        if success and "written successfully" in message.lower():
            if detail:
                self.console.print(f"[success]✅ File written:[/] {detail}")
            else:
                self.console.print("[success]✅ File written[/]")
            self.emitted_any = True
            return

        if detail:
            self.console.print(f"[error]✖ Failed to write file:[/] {detail}")
        else:
            self.console.print("[error]✖ Failed to write file[/]")
        self.emitted_any = True
        if message:
            for line in self._wrap(message, max_lines=2):
                self.console.print(f"[dim]  {line}[/]")

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        tool_name = self._tool_name(serialized)
        payload = self._tool_input_payload(inputs, input_str)
        self._inflight_tools[run_id] = {
            "name": tool_name,
            "input": payload,
        }
        if tool_name == "run_command":
            self._print_run_command_start(payload.get("query"))
        elif tool_name == "read_file":
            self._print_read_file_start(
                payload.get("path") or payload.get("filename")
            )
        elif tool_name == "write_code":
            self._print_write_code_start(
                payload.get("filename"),
                payload.get("path"),
            )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs,
    ) -> None:
        tool_info = self._inflight_tools.pop(run_id, {})
        tool_name = tool_info.get("name")
        payload = output.content if isinstance(output, ToolMessage) else output
        success = (
            output.status == "success" if isinstance(output, ToolMessage) else True
        )
        if tool_name == "run_command":
            self._print_run_command_end(
                payload,
                query=tool_info.get("input", {}).get("query"),
                success=success,
            )
        elif tool_name == "read_file":
            self._print_read_file_end(
                payload,
                path=tool_info.get("input", {}).get("path")
                or tool_info.get("input", {}).get("filename"),
            )
        elif tool_name == "write_code":
            self._print_write_code_end(
                payload,
                filename=tool_info.get("input", {}).get("filename"),
                path=tool_info.get("input", {}).get("path"),
                success=success,
            )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs,
    ) -> None:
        tool_info = self._inflight_tools.pop(run_id, {})
        tool_name = tool_info.get("name", "tool")
        self.console.print(f"[error]✖ {tool_name} failed.[/]")
        self.emitted_any = True
        for line in self._wrap(error, max_lines=2):
            self.console.print(f"[dim]  {line}[/]")

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        if name != DEFAULT_EVENT_NAME or not isinstance(data, dict):
            return
        if "agent" in data:
            self._print_agent_event(data)


__all__ = ["HITLLogEventHandler"]
