"""
SummarizingAgent example runner (CLI-first, human-friendly).

- Self-contained: no external example harness helpers
- CLI args with --help (no env var dependency for config)
- Uses ursa.util.llm_factory.setup_llm for provider wiring
- Rich console/progress when available; plain prints otherwise
- Reads inputs from ./summarizing_agent_example_inputs/ by default
- Writes output to ./summarizing_agent_example_output/summary.txt by default

Example usage:
  python summarizing_agent_example.py
  python summarizing_agent_example.py --mode synthesis
  python summarizing_agent_example.py --input-dir ./docs --recurse --max-files 50
  python summarizing_agent_example.py --output-path ./out/summary.txt

Using a non-default gateway/base URL:
  python summarizing_agent_example.py --base-url https://your.gateway.example/v1 --api-key-env YOUR_GATEWAY_API_KEY
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final, Optional, Sequence

from ursa.agents.summarizing_agent import SummarizingAgent
from ursa.util.llm_factory import setup_llm

# ----------------------------
# Task presets
# ----------------------------

_TASK_SUMMARY: Final[str] = (
    "Produce: (1) Executive Summary (<=200 words), (2) Required Actions (5–10 bullets), "
    "then a main synthesis (600–900 words). "
    "Do not segment by input."
)

_TASK_SYNTHESIS: Final[str] = (
    "Write one integrated synthesis that merges all material into a single coherent narrative. "
    "Organize by themes, tensions, and shared ideas, and highlight meaningful contrasts where they exist—"
    "but do NOT structure the output as separate summaries of each input. "
    "Keep one consistent voice throughout."
)

# Default provider wiring: OpenAI-compatible settings.
_DEFAULT_MODEL_CHOICE: Final[str] = "openai:gpt-4o-mini"
_DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"
_DEFAULT_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"


# ----------------------------
# Rich helpers (optional)
# ----------------------------


def _get_console():
    """Return a rich console if installed; otherwise None."""
    try:
        from rich import get_console  # type: ignore

        return get_console()
    except Exception:
        return None


def _panel(console, title: str, lines: Sequence[str]) -> None:
    """Render a small info panel; fallback to plain prints if Rich is unavailable."""
    body = "\n".join(lines)
    if console is None:
        print(f"\n== {title} ==\n{body}\n")
        return
    try:
        from rich.panel import Panel  # type: ignore

        console.print(Panel.fit(body, title=title))
    except Exception:
        print(f"\n== {title} ==\n{body}\n")


class _ProgressTracker:
    """
    Receives progress events emitted by SummarizingAgent via the `on_event` callback.
    Uses Rich progress bars if available; otherwise prints a small set of milestones.
    """

    _PRINT_EVENTS: Final[set[str]] = {
        "start",
        "discover_done",
        "read_start",
        "chunking_done",
        "reduce_round",
        "rewrite_start",
        "rewrite_done",
        "done",
    }

    def __init__(self, console):
        self.console = console
        self.progress = None
        self._tasks: dict[str, Any] = {}

        if console is None:
            return

        try:
            from rich.progress import (  # type: ignore
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
        except Exception:
            self.progress = None

    def __enter__(self):
        if self.progress is not None:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.progress is not None:
            return self.progress.__exit__(exc_type, exc, tb)
        return False

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        """Dispatch progress events from the agent."""
        if self.progress is None:
            if event in self._PRINT_EVENTS:
                print(f"[{event}] {data}")
            return

        if event == "discover_done":
            total = int(data.get("count", 0))
            self._tasks["read"] = self.progress.add_task(
                "Reading files", total=total
            )
            return

        if event == "read_start":
            f = data.get("file")
            if f:
                self.progress.console.print(f"[READING]:  {f}")
            return

        if event in {"read_ok", "read_skip"}:
            task_id = self._tasks.get("read")
            if task_id is not None:
                self.progress.advance(task_id, 1)
            return

        if event == "chunking_done":
            total = int(data.get("chunks", 0))
            self._tasks["map"] = self.progress.add_task(
                "Summarizing chunks (map)", total=total
            )
            return

        if event == "map_done":
            task_id = self._tasks.get("map")
            if task_id is not None:
                self.progress.advance(task_id, 1)
            return

        if event == "reduce_start":
            self._tasks["reduce"] = self.progress.add_task(
                "Reducing summaries", total=1
            )
            return

        if event == "reduce_round":
            batches = int(data.get("batches", 1))
            task_id = self._tasks.get("reduce")
            if task_id is not None:
                self.progress.reset(task_id, total=batches)

            self.progress.console.print(
                f"[dim]Reduce round {data.get('round')}: {data.get('items')} items → {batches} merges[/dim]"
            )
            return

        if event == "reduce_batch_done":
            task_id = self._tasks.get("reduce")
            if task_id is not None:
                self.progress.advance(task_id, 1)
            return

        if event == "rewrite_start":
            self.progress.console.print("[dim]Rewrite pass...[/dim]")
            return

        if event == "rewrite_done":
            self.progress.console.print(
                f"[dim]Rewrite done (changed={data.get('changed')}).[/dim]"
            )
            return

        if event == "done":
            self.progress.console.print("[bold green]Done.[/bold green]")
            return


# ----------------------------
# CLI parsing
# ----------------------------


def _comma_list(value: str) -> list[str]:
    """Parse comma-separated list values."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _normalize_exts(exts: Sequence[str]) -> tuple[str, ...]:
    """Normalize extensions to begin with '.'."""
    out: list[str] = []
    for e in exts:
        e = e.strip()
        if not e:
            continue
        out.append(e if e.startswith(".") else f".{e}")
    return tuple(out)


def _build_parser(
    default_input_dir: Path, default_output_path: Path
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarizing_agent_example",
        description="Run SummarizingAgent over a directory of input documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing input documents to summarize.",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recurse into subdirectories under --input-dir.",
    )
    parser.add_argument(
        "--allowed-exts",
        type=_comma_list,
        default=[".txt", ".md", ".rst", ".pdf"],
        help="Comma-separated list of allowed extensions.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files (deterministic after sorting).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path,
        help="File path for the generated summary.",
    )

    # Task / behavior
    parser.add_argument(
        "--mode",
        choices=["summary", "synthesis"],
        default="summary",
        help="Select built-in task behavior; can be overridden by --task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Override the task text entirely (if set, ignores --mode presets).",
    )
    parser.add_argument(
        "--show-tool-output",
        action="store_true",
        help="Do not silence tool stdout/stderr.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat tool errors/empty reads as hard failures.",
    )

    # Provider wiring (OpenAI-compatible by default)
    parser.add_argument(
        "--model-choice",
        type=str,
        default=_DEFAULT_MODEL_CHOICE,
        help=(
            "Model choice string used by setup_llm (e.g., provider_alias:model_id). "
            "Default targets an OpenAI-compatible provider."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=_DEFAULT_BASE_URL,
        help="Base URL for an OpenAI-compatible endpoint (can be a gateway).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=_DEFAULT_API_KEY_ENV,
        help="Name of env var that contains the API key (read by setup_llm).",
    )

    # LLM params
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Retry attempts."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2500,
        help="Max tokens for completion (ChatOpenAI style).",
    )

    # Summarizer knobs
    parser.add_argument(
        "--chunk-size-chars",
        type=int,
        default=10000,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=800,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=200,
        help="Cap chunks produced per file.",
    )
    parser.add_argument(
        "--reduce-batch-size",
        type=int,
        default=8,
        help="Batch size per reduce round.",
    )

    # Agent framework knobs
    parser.add_argument(
        "--thread-id",
        type=str,
        default="summarizer_example",
        help="Thread/run identifier passed to the agent.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory used by tools; default is ./workspace_summarizer next to this script.",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable agent metrics collection.",
    )

    return parser


def _resolve_task(mode: str, override: str) -> str:
    """Choose the task prompt string from mode/override."""
    if override and override.strip():
        return override.strip()
    return _TASK_SYNTHESIS if mode == "synthesis" else _TASK_SUMMARY


def _build_models_cfg(
    base_url: str,
    api_key_env: str,
    temperature: float,
    max_retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the inline models_cfg consumed by setup_llm."""
    return {
        "providers": {
            "openai": {
                "model_provider": "openai",
                "base_url": base_url,
                "api_key_env": api_key_env,
            }
        },
        "defaults": {
            "params": {
                "temperature": float(temperature),
                "max_retries": int(max_retries),
                "max_tokens": int(max_tokens),
                "use_responses_api": False,
            }
        },
        "agents": {"summarizer": {"params": {}}},
    }


# ----------------------------
# Main
# ----------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    console = _get_console()

    here = Path(__file__).resolve().parent
    default_input_dir = here / "summarizing_agent_example_inputs"
    default_output_path = (
        here / "summarizing_agent_example_output" / "summary.txt"
    )

    parser = _build_parser(
        default_input_dir=default_input_dir,
        default_output_path=default_output_path,
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_path: Path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = (
        args.workspace
        if args.workspace is not None
        else (here / "workspace_summarizer")
    )
    allowed_exts = _normalize_exts(args.allowed_exts)
    task = _resolve_task(args.mode, args.task)

    models_cfg = _build_models_cfg(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )

    _panel(
        console,
        "SummarizingAgent Example",
        [
            f"Input: {input_dir}",
            f"Output: {output_path}",
            f"Mode: {args.mode}",
            f"Recurse: {bool(args.recurse)}",
            f"Allowed exts: {', '.join(allowed_exts)}",
            f"Max files: {args.max_files}",
            f"Tool output: {'ON' if args.show_tool_output else 'OFF'}",
            f"Strict: {bool(args.strict)}",
            f"Model choice: {args.model_choice}",
            f"Base URL: {args.base_url}",
            f"API key env: {args.api_key_env}",
        ],
    )

    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"input-dir is not a directory: {input_dir}"
        if console is not None:
            console.print(f"[bold red]✖ Error:[/bold red] {msg}")
        else:
            print(f"Error: {msg}")
        return 2

    try:
        llm = setup_llm(
            model_choice=args.model_choice,
            models_cfg=models_cfg,
            agent_name="summarizer",
            base_llm_kwargs={
                "max_tokens": int(args.max_tokens),
                # Prevent older defaults from winning if both exist.
                "max_completion_tokens": None,
            },
            console=console,
        )

        agent = SummarizingAgent(
            llm=llm,
            thread_id=args.thread_id,
            workspace=Path(workspace),
            enable_metrics=(not args.disable_metrics),
        )

        with _ProgressTracker(console) as tracker:
            result = agent.invoke({
                "input_docs_dir": str(input_dir),
                "recurse": bool(args.recurse),
                "allowed_extensions": allowed_exts,
                "max_files": args.max_files,
                "task": task,
                "silent_tools": (not args.show_tool_output),
                "strict": bool(args.strict),
                "chunk_size_chars": int(args.chunk_size_chars),
                "chunk_overlap_chars": int(args.chunk_overlap_chars),
                "max_chunks_per_file": int(args.max_chunks_per_file),
                "reduce_batch_size": int(args.reduce_batch_size),
                "on_event": tracker.on_event,
            })

        summary = (result.get("summary") or "").strip()
        output_path.write_text(summary + "\n", encoding="utf-8")

        if console is not None:
            console.print(f"[bold green]✔ Wrote[/bold green] {output_path}")
        else:
            print(f"OK: wrote {output_path}")
        return 0

    except Exception as e:
        if console is not None:
            console.print(
                f"[bold red]✖ Error:[/bold red] {type(e).__name__}: {e}"
            )
            try:
                from rich.traceback import Traceback  # type: ignore

                console.print(
                    Traceback.from_exception(
                        type(e), e, e.__traceback__, show_locals=False
                    )
                )
            except Exception:
                pass
        else:
            print(f"Error: {type(e).__name__}: {e}")

        try:
            output_path.write_text(
                f"[Error] {type(e).__name__}: {e}\n", encoding="utf-8"
            )
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
