# src/ursa/util/tool_utils.py
from __future__ import annotations

import re
import textwrap
from typing import Any, Mapping


def _get_tool_mapping(agent: Any) -> dict[str, Any]:
    """
    Return {name: tool} from AgentWithTools-like objects.
    Prefers public .tools, falls back to ._tools.
    """
    if hasattr(agent, "tools"):
        t = agent.tools
        # tools property returns a dict copy in your mixin; good.
        if isinstance(t, Mapping):
            return dict(t)
    if hasattr(agent, "_tools") and isinstance(agent._tools, Mapping):
        return dict(agent._tools)
    return {}


def debug_tools(agent: Any) -> dict[str, dict[str, Any]]:
    tools = _get_tool_mapping(agent)

    out: dict[str, dict[str, Any]] = {}
    for name, tool in sorted(tools.items(), key=lambda kv: kv[0]):
        schema = getattr(tool, "args_schema", None)
        schema_name = (
            getattr(schema, "__name__", None)
            or getattr(schema.__class__, "__name__", None)
            if schema
            else None
        )
        out[name] = {
            "class": tool.__class__.__name__,
            "description": getattr(tool, "description", None),
            "args_schema": schema_name or str(schema) if schema else None,
            "return_direct": getattr(tool, "return_direct", None),
        }
    return out


def list_tools(agent: Any) -> list[str]:
    return sorted(_get_tool_mapping(agent).keys())


def _normalize_docstring(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\t", "    ")
    text = textwrap.dedent(text).strip()
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _wrap_preserve_newlines(text: str, width: int) -> str:
    if not text:
        return ""
    out: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        leading = re.match(r"^\s*", line).group(0)
        body = line.strip()
        out.append(
            textwrap.fill(
                body,
                width=width,
                initial_indent=leading,
                subsequent_indent=leading,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(out)


def print_tool_report(
    agent: Any, *, width: int = 100, rich: bool = True
) -> None:
    info = debug_tools(agent)
    if not rich:
        # plain
        if not info:
            print("Tools: (none)")
            return
        for name, meta in info.items():
            print("=" * width)
            print(f"{name}  [{meta.get('class')}]")
            print("-" * min(width, len(name) + 4 + len(meta.get("class", ""))))
            desc = _wrap_preserve_newlines(
                _normalize_docstring(meta.get("description") or ""), width=width
            )
            print(desc or "(no description)")
            print()
            print(f"args_schema   : {meta.get('args_schema')}")
            print(f"return_direct : {meta.get('return_direct')}")
        print("=" * width)
        return

    # rich rendering
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        # fall back automatically
        print_tool_report(agent, width=width, rich=False)
        return

    console = Console(width=width)
    if not info:
        console.print("[bold]Tools:[/bold] (none)")
        return

    for name in sorted(info.keys()):
        meta = info[name]
        desc = _normalize_docstring(meta.get("description") or "")
        desc_md = desc
        desc_md = re.sub(r"^Args:\s*$", "### Args", desc_md, flags=re.MULTILINE)
        desc_md = re.sub(
            r"^Arguments:\s*$", "### Args", desc_md, flags=re.MULTILINE
        )
        desc_md = re.sub(
            r"^Returns:\s*$", "### Returns", desc_md, flags=re.MULTILINE
        )

        wrapped = _wrap_preserve_newlines(desc_md, width=width - 10)

        footer = Text()
        footer.append(f"args_schema: {meta.get('args_schema')}\n", style="dim")
        footer.append(
            f"return_direct: {meta.get('return_direct')}", style="dim"
        )

        console.print(
            Panel(
                Markdown(wrapped)
                if wrapped
                else Text("(no description)", style="dim"),
                title=f"[bold]{name}[/bold]  [dim]{meta.get('class')}[/dim]",
                subtitle=footer,
                border_style="cyan",
                padding=(1, 2),
            )
        )
