import json
from pathlib import Path
from typing import Any, ClassVar

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from pygments.util import ClassNotFound
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from ursa.util.diff_renderer import DiffRenderer
from ursa.util.events import DEFAULT_EVENT_NAME

FILE_TOOL_NAMES = {
    "edit_code",
    "read_file",
    "write_code",
    "write_code_with_repo",
}

TOOL_STYLES = {
    "edit_code": "green",
    "read_file": "green",
    "write_code": "green",
    "write_code_with_repo": "green",
}


def has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, set, tuple)):
        return bool(value)
    return True


class CallbackRenderingMixin:
    """Rendering helpers for the HITL callback handler.

    This intentionally covers only the event families shown in the CLI:
    read/write/edit file tools, plus execute/plan/hypothesis agent progress.
    """

    AGENT_KEYS: ClassVar[dict[str, str]] = {
        "ExecutionAgent": "executor",
        "HypothesizerAgent": "hypothesizer",
        "PlanningAgent": "planner",
        "executor": "executor",
        "hypothesizer": "hypothesizer",
        "planner": "planner",
    }
    AGENT_RULE_TITLES: ClassVar[dict[str, str]] = {
        "executor": "⚙️ Execute",
        "hypothesizer": "💡 Hypothesize",
        "planner": "🗺️ Plan",
    }

    def _clean(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _agent_key(self, value: Any) -> str | None:
        raw = self._clean(value)
        return self.AGENT_KEYS.get(raw)

    def _tool_style(self, tool_name: str | None) -> str:
        return TOOL_STYLES.get(str(tool_name or ""), "green")

    def _workspace_path(self, value: Any) -> Path | None:
        raw = self._clean(value)
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path

    def _display_path(self, value: Any) -> str | None:
        path = self._workspace_path(value)
        if path is None:
            return None
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            return str(resolved.relative_to(self.workspace))
        except ValueError:
            return str(resolved)

    def _print_panel(
        self,
        renderable: Any,
        *,
        title: str,
        subtitle: str | None = None,
        border_style: str = "green",
    ) -> None:
        self.console.print(
            Panel(
                renderable,
                title=title,
                subtitle=subtitle,
                border_style=border_style,
            )
        )

    def _render_value(
        self,
        value: Any,
        *,
        language: str | None = None,
        line_numbers: bool = False,
    ) -> Syntax:
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, indent=2, default=str)
            language = language or "json5"
        else:
            text = str(value or "(empty)")
            language = language or "text"
        return Syntax(
            text,
            language,
            line_numbers=line_numbers,
            word_wrap=not line_numbers,
        )

    def _event_text(self, value: Any) -> str | None:
        text = self._clean(value)
        return text or None

    def _display_lines(self, value: Any) -> list[str]:
        text = str(value or "").strip()
        if not text:
            return []
        return [line for line in text.splitlines() if line]

    def _plan_step_lines(self, steps: Any) -> list[str]:
        if not isinstance(steps, list):
            return []
        lines: list[str] = []
        total_steps = 0
        for total_steps, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                model_dump = getattr(step, "model_dump", None)
                step = (
                    model_dump()
                    if callable(model_dump)
                    else {"name": str(step)}
                )
            name = self._event_text(step.get("name")) or f"Step {total_steps}"
            description = self._event_text(step.get("description")) or ""
            text = f"{total_steps}. {name}"
            if description:
                text += f": {description}"
            lines.append(text)
        return lines

    def _plan_steps_markdown(self, steps: Any) -> str | None:
        lines = self._plan_step_lines(steps)
        if not lines:
            return None
        lines.append("\n")
        return "\n".join(lines)

    def _agent_icon(self, agent: str, stage: str, data: dict[str, Any]) -> str:
        if agent == "planner":
            if stage == "reflect_result":
                return "✅" if data.get("approved") else "🔁"
            return {"generate": "📐", "plan_ready": "🗺️"}.get(stage, "📋")
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
        return "⚙️"

    def _print_agent_rule(self, agent: str) -> None:
        title = self.AGENT_RULE_TITLES.get(agent, agent)
        self.console.print(Rule(f"[blue]{title}[/]", style="blue"))

    def _print_agent_event(self, data: dict[str, Any]) -> None:
        agent = self._agent_key(data.get("agent"))
        if agent is None:
            return

        message = self._clean(data.get("message"))
        if not message:
            return

        if self._last_agent not in (None, agent):
            self._print_agent_rule(agent)
        self._last_agent = agent

        stage = self._clean(data.get("stage"))
        phase = self._clean(data.get("phase"))
        style = "error" if phase == "error" else "blue"
        self.console.print(
            f"[{style}]{self._agent_icon(agent, stage, data)} {message}[/]"
        )
        if agent == "planner" and (
            plan_markdown := self._plan_steps_markdown(data.get("steps"))
        ):
            self.console.print(Markdown(plan_markdown))
        elif agent == "executor" and stage == "step":
            for line in self._display_lines(data.get("preview")):
                self.console.print(line)
        elif stage == "reflect_result":
            if reason := data.get("reason"):
                self.console.print(Markdown(reason))
        else:
            for line in self._display_lines(data.get("preview")):
                self.console.print(line)

        self.emitted_any = True

        if output_path := self._display_path(data.get("output_path")):
            self.console.print(f"[dim]  {output_path}[/]")

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

    def _has_inflight_tool(self, *tool_names: str) -> bool:
        names = set(tool_names)
        return any(
            self._clean(tool.get("name")) in names
            for tool in self._inflight_tools.values()
        )

    def _is_file_tool(self, tool_name: str | None) -> bool:
        return str(tool_name or "") in FILE_TOOL_NAMES

    def _tool_path(self, data: dict[str, Any]) -> str | None:
        return self._display_path(data.get("path") or data.get("filename"))

    def _read_file_language(self, display_path: str | None, text: str) -> str:
        suffix = Path(display_path or "file.txt").suffix.lower()
        try:
            lexer_name = str(
                Syntax.guess_lexer(Path(display_path or "file.txt").name, text)
            ).lower()
        except (ClassNotFound, TypeError, ValueError):
            lexer_name = ""

        if suffix == ".py" or lexer_name == "python":
            return "python"
        if suffix == ".json" or lexer_name == "json":
            return "json5"
        return "text"

    def _print_read_file_start(self, path: str | None) -> None:
        detail = self._display_path(path)
        self.console.print(
            f"[success]📖 Reading file:[/] {detail}"
            if detail
            else "[success]📖 Reading file[/]"
        )
        self.emitted_any = True

    def _print_read_file_end(
        self, payload: Any, *, path: str | None = None
    ) -> None:
        detail = self._display_path(path)
        self.console.print(
            f"[success]📖 File read:[/] {detail}"
            if detail
            else "[success]📖 File read[/]"
        )
        self.emitted_any = True
        if not has_content(payload):
            return

        language = self._read_file_language(detail, str(payload))
        self._print_panel(
            self._render_value(
                payload,
                language=language,
            ),
            title="read_file output",
            border_style=self._tool_style("read_file"),
        )

    def _print_write_code_preview(
        self, filename: str | None, code: Any
    ) -> None:
        if not has_content(code):
            return
        try:
            lexer_name = Syntax.guess_lexer(
                str(filename or "file.txt"), str(code)
            )
        except (ClassNotFound, TypeError, ValueError):
            lexer_name = "text"
        self._print_panel(
            Syntax(
                str(code),
                lexer_name,
                line_numbers=True,
            ),
            title="File Preview",
            border_style=self._tool_style("write_code"),
        )

    def _print_write_code_start(
        self, filename: str | None, path: str | None, code: Any = None
    ) -> None:
        detail = self._display_path(path or filename)
        self.console.print(
            f"[success]✏️ Writing file:[/] {detail}"
            if detail
            else "[success]✏️ Writing file[/]"
        )
        self.emitted_any = True
        self._print_write_code_preview(filename or path, code)

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
        lower_message = message.lower()
        if success and "written successfully" in lower_message:
            self.console.print(
                f"[success]✅ File written:[/] {detail}"
                if detail
                else "[success]✅ File written[/]"
            )
        else:
            self.console.print(
                f"[error]✖ Failed to write file:[/] {detail}"
                if detail
                else "[error]✖ Failed to write file[/]"
            )
            for line in self._display_lines(message):
                self.console.print(f"[dim]  {line}[/]")
        self.emitted_any = True

    def _print_edit_diff(
        self,
        *,
        filename: str | None,
        old_code: Any,
        new_code: Any,
    ) -> None:
        if not has_content(old_code) and not has_content(new_code):
            return
        display_path = self._display_path(filename) or str(filename or "file")
        self._print_panel(
            DiffRenderer(
                str(old_code or ""), str(new_code or ""), display_path
            ),
            title="Edit Diff",
            border_style=self._tool_style("edit_code"),
        )

    def _print_edit_code_start(
        self,
        filename: str | None,
        path: str | None,
        old_code: Any = None,
        new_code: Any = None,
    ) -> None:
        detail = self._display_path(path or filename)
        self.console.print(
            f"[success]✏️ Editing file:[/] {detail}"
            if detail
            else "[success]✏️ Editing file[/]"
        )
        self.emitted_any = True
        self._print_edit_diff(
            filename=path or filename,
            old_code=old_code,
            new_code=new_code,
        )

    def _print_edit_code_end(
        self,
        payload: Any,
        *,
        filename: str | None,
        path: str | None,
        success: bool,
    ) -> None:
        detail = self._display_path(path or filename)
        message = self._clean(payload)
        lower_message = message.lower()
        no_change = lower_message.startswith("no changes made")
        failed = not success or lower_message.startswith("failed")
        if failed:
            self.console.print(
                f"[error]✖ Failed to edit file:[/] {detail}"
                if detail
                else "[error]✖ Failed to edit file[/]"
            )
            for line in self._display_lines(message):
                self.console.print(f"[dim]  {line}[/]")
        elif no_change:
            self.console.print(
                f"[warn]⚠ No changes made:[/] {detail}"
                if detail
                else "[warn]⚠ No changes made[/]"
            )
            for line in self._display_lines(message):
                self.console.print(f"[dim]  {line}[/]")
        else:
            self.console.print(
                f"[success]✅ File edited:[/] {detail}"
                if detail
                else "[success]✅ File edited[/]"
            )
        self.emitted_any = True

    def _print_tool_event(self, data: dict[str, Any]) -> None:
        tool = self._clean(data.get("tool"))
        if not self._is_file_tool(tool):
            return

        message = self._clean(data.get("message"))
        if not message:
            return

        phase = self._clean(data.get("phase")) or None
        detail = self._tool_path(data)
        style = "error" if phase == "error" else "success"
        icon = "📖" if tool == "read_file" else "✏️"
        if phase == "end" and tool in {
            "edit_code",
            "write_code",
            "write_code_with_repo",
        }:
            icon = "✅"
        elif phase == "error":
            icon = "✖"

        self.console.print(
            f"[{style}]{icon} {message}:[/] {detail}"
            if detail
            else f"[{style}]{icon} {message}[/]"
        )
        self.emitted_any = True

        if phase == "error":
            for line in self._display_lines(data.get("error")):
                self.console.print(f"[dim]  {line}[/]")
        elif reason := data.get("reason"):
            for line in self._display_lines(reason):
                self.console.print(f"[dim]  {line}[/]")
        if tool == "edit_code":
            self._print_edit_diff(
                filename=data.get("path") or data.get("filename"),
                old_code=data.get("old_code"),
                new_code=data.get("new_code"),
            )


class HITLLogEventHandler(CallbackRenderingMixin, AsyncCallbackHandler):
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

    async def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        return None

    async def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        return None

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
        if tool_name == "read_file":
            self._print_read_file_start(
                payload.get("path") or payload.get("filename")
            )
        elif tool_name in {"write_code", "write_code_with_repo"}:
            self._print_write_code_start(
                payload.get("filename"),
                payload.get("path"),
                payload.get("code"),
            )
        elif tool_name == "edit_code":
            self._print_edit_code_start(
                payload.get("filename"),
                payload.get("path"),
                payload.get("old_code"),
                payload.get("new_code"),
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
            output.status == "success"
            if isinstance(output, ToolMessage)
            else True
        )
        if tool_name == "read_file":
            self._print_read_file_end(
                payload,
                path=tool_info.get("input", {}).get("path")
                or tool_info.get("input", {}).get("filename"),
            )
        elif tool_name in {"write_code", "write_code_with_repo"}:
            self._print_write_code_end(
                payload,
                filename=tool_info.get("input", {}).get("filename"),
                path=tool_info.get("input", {}).get("path"),
                success=success,
            )
        elif tool_name == "edit_code":
            tool_input = tool_info.get("input", {})
            self._print_edit_code_end(
                payload,
                filename=tool_input.get("filename"),
                path=tool_input.get("path"),
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
        if not self._is_file_tool(tool_name):
            return
        self.console.print(f"[error]✖ {tool_name} failed.[/]")
        self.emitted_any = True
        for line in self._display_lines(error):
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
        elif "tool" in data:
            tool = self._clean(data.get("tool"))
            if not self._is_file_tool(tool):
                return
            if tool == "read_file" and self._has_inflight_tool("read_file"):
                return
            if tool == "write_code" and self._has_inflight_tool(
                "write_code",
                "write_code_with_repo",
            ):
                return
            if tool == "edit_code" and self._has_inflight_tool("edit_code"):
                return
            self._print_tool_event(data)


__all__ = ["HITLLogEventHandler"]
