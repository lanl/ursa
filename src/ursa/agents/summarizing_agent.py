from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, Sequence, TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages

from ursa.agents.base import BaseAgent
from ursa.prompt_library.summarizing_prompts import (
    FINAL_COVERAGE_INSTRUCTION,
    MAP_USER_INSTRUCTIONS,
    SYSTEM_MAP_PROMPT,
    SYSTEM_NOTES_REDUCE_PROMPT,
    SYSTEM_REDUCE_PROMPT,
    SYSTEM_REWRITE_PROMPT,
)
from ursa.tools import read_file

_DENY_RE = re.compile(
    r"\b(doc|docs|document|documents|file|files|filename|filenames|source|sources|excerpt|excerpts|chunk|chunks)\b"
    r"|\b(the text above|the passage above|this text|this document|this file|above text)\b"
    r"|\b(url|urls|link|links)\b"
    r"|https?://\S+"
    r"|\bwww\.\S+",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+")
_RULE_RE = re.compile(r"(?m)^\s*(-{3,}|={3,})\s*$")


def _needs_rewrite(text: str) -> bool:
    """Return True if the generated text likely violates formatting/meta constraints."""
    if not text.strip():
        return False
    return bool(
        _DENY_RE.search(text)
        or _HEADING_RE.search(text)
        or _RULE_RE.search(text)
    )


def _normalize_whitespace(s: str) -> str:
    """Normalize line endings and collapse excessive blank lines for stable prompting/output."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk_text(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
    max_chunks: int,
) -> list[str]:
    """
    Deterministic, paragraph-aware chunking with optional overlap.

    - Keeps paragraphs intact where possible.
    - Splits oversized paragraphs into fixed-size segments.
    - Adds overlap between consecutive chunks to preserve continuity.
    """
    text = _normalize_whitespace(text)
    if len(text) <= chunk_size_chars:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    n = 0

    def flush() -> None:
        """Flush current buffered paragraphs to a new chunk."""
        nonlocal buf, n
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            n = 0

    for p in paras:
        if len(p) > chunk_size_chars:
            flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_size_chars)
                seg = p[start:end].strip()
                if seg:
                    chunks.append(seg)
                if end >= len(p):
                    break
                start = max(0, end - overlap_chars)
            continue

        add = len(p) + (2 if buf else 0)
        if n + add <= chunk_size_chars:
            buf.append(p)
            n += add
        else:
            flush()
            buf.append(p)
            n = len(p)

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            out.append((prev[-overlap_chars:] + "\n\n" + nxt).strip())
        chunks = out

    return chunks[:max_chunks]


@dataclass(frozen=True)
class _Doc:
    """In-memory representation of an input document."""

    name: str
    text: str


@contextlib.contextmanager
def _silence_stdio(enabled: bool):
    """
    Optionally suppress stdout/stderr.

    Some tools can be chatty; silencing keeps logs and notebooks clean.
    """
    if not enabled:
        yield
        return
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


class SummarizingState(TypedDict, total=False):
    """
    Agent state schema.

    Inputs control selection, reading, chunking, and summarization behavior.
    Outputs include the final summary plus file selection diagnostics.
    """

    messages: Annotated[list, add_messages]

    input_docs_dir: str
    recurse: bool
    allowed_extensions: Sequence[str]
    max_files: Optional[int]
    task: Optional[str]
    silent_tools: bool
    strict: bool
    chunk_size_chars: int
    chunk_overlap_chars: int
    max_chunks_per_file: int
    reduce_batch_size: int
    on_event: Optional[Callable[[str, dict[str, Any]], None]]

    summary: str
    selected_files: list[str]
    skipped_files: list[str]


class SummarizingAgent(BaseAgent[SummarizingState]):
    """
    Summarize a directory of documents using a balanced map-reduce strategy.

    Strategy:
    - Per-file MAP: summarize each chunk into compact notes.
    - Per-file REDUCE: merge chunk notes into a single “doc notes” artifact per file.
    - Global REDUCE: merge doc notes into a single structured summary.
    - Optional REWRITE: scrub meta-language if the model violates constraints.

    This two-level approach prevents long files from dominating the final output purely by chunk count.
    """

    state_type = SummarizingState

    def __init__(self, llm: BaseChatModel, **kwargs):
        super().__init__(llm=llm, **kwargs)

    def format_result(self, state: SummarizingState) -> str:
        """Return the final summary text for the framework's result handling."""
        return state.get("summary", "")

    def _build_graph(self):
        """Single-node graph: summarize and finish."""
        self.add_node(self._summarize_node, "summarize")
        self.graph.set_entry_point("summarize")
        self.graph.add_edge("summarize", END)
        self.graph.set_finish_point("summarize")

    def _summarize_node(self, state: SummarizingState) -> SummarizingState:
        """Read, chunk, summarize per file, then globally synthesize into one output."""

        def emit(event: str, **data: Any) -> None:
            cb = state.get("on_event")
            if cb is not None:
                cb(event, data)

        input_dir = Path(str(state.get("input_docs_dir", "")))
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"input_docs_dir is not a directory: {input_dir}")

        recurse = bool(state.get("recurse", False))
        allowed_exts = tuple(
            str(e).lower()
            for e in (
                state.get("allowed_extensions")
                or (".txt", ".md", ".rst", ".pdf")
            )
        )
        max_files = state.get("max_files")
        silent_tools = bool(state.get("silent_tools", True))
        strict = bool(state.get("strict", False))

        chunk_size = int(state.get("chunk_size_chars", 10_000))
        chunk_overlap = int(state.get("chunk_overlap_chars", 800))
        max_chunks_per_file = int(state.get("max_chunks_per_file", 200))
        reduce_batch = int(state.get("reduce_batch_size", 8))

        task = state.get("task") or (
            "Write a single unified summary in one consistent voice. "
            "Do not segment the output by input, and do not produce separate per-item summaries."
        )

        emit("start", input=str(input_dir), recurse=recurse)

        selected, skipped = self._select_files(
            root=input_dir,
            recurse=recurse,
            allowed_exts=allowed_exts,
            max_files=max_files,
        )
        emit("discover_done", count=len(selected))

        docs: list[_Doc] = []
        tool_state = {"workspace": str(input_dir)}

        for rel in selected:
            emit("read_start", file=rel)
            try:
                with _silence_stdio(silent_tools):
                    txt = (
                        read_file.func(filename=rel, state=tool_state)  # type: ignore[attr-defined]
                        if hasattr(read_file, "func")
                        else read_file(filename=rel, state=tool_state)  # type: ignore[misc]
                    )

                s = _normalize_whitespace("" if txt is None else str(txt))

                if s.startswith("[Error]") or s.startswith("[Error]:"):
                    if strict:
                        raise RuntimeError(f"read_file error for {rel}: {s}")
                    emit("read_skip", file=rel, reason="tool_error")
                    continue

                if not s:
                    if strict:
                        raise RuntimeError(f"Empty file content for {rel}")
                    emit("read_skip", file=rel, reason="empty")
                    continue

                docs.append(_Doc(name=rel, text=s))
                emit("read_ok", file=rel)

            except Exception as e:
                if strict:
                    raise
                emit(
                    "read_skip",
                    file=rel,
                    reason=f"exception:{type(e).__name__}",
                )
                continue

        if not docs:
            return {
                "summary": "",
                "selected_files": selected,
                "skipped_files": skipped,
            }

        # --- Per-file processing ---
        # Each file produces exactly one doc-notes artifact, preventing long files from dominating.
        doc_notes: list[str] = []
        total_chunks = 0

        for d in docs:
            chunks = _chunk_text(
                d.text,
                chunk_size_chars=chunk_size,
                overlap_chars=chunk_overlap,
                max_chunks=max_chunks_per_file,
            )
            chunks = [c for c in chunks if c.strip()]
            total_chunks += len(chunks)

            emit("file_chunking_done", file=d.name, chunks=len(chunks))

            if not chunks:
                continue

            # MAP: each chunk -> compact bullet notes.
            partials: list[str] = []
            for i, ch in enumerate(chunks, start=1):
                prompt_user = (
                    f"Task:\n{task}\n\n"
                    f"{MAP_USER_INSTRUCTIONS}\n"
                    "Passage:\n"
                    "```\n"
                    f"{ch}\n"
                    "```"
                )

                msg = self.llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_MAP_PROMPT),
                        HumanMessage(content=prompt_user),
                    ],
                    self.build_config(tags=["summarizer", "map"]),
                )
                partials.append(msg.text.strip())
                emit("map_done", file=d.name, i=i, total=len(chunks))

            # Per-file REDUCE: merge chunk-notes into one bullet list (doc-notes).
            note_text = self._reduce_notes_list(
                partials,
                task=task,
                reduce_batch=reduce_batch,
                emit=emit,
                scope=f"file:{d.name}",
            )

            if note_text.strip():
                doc_notes.append(note_text.strip())
                emit("file_notes_done", file=d.name)

        if not doc_notes:
            return {
                "summary": "",
                "selected_files": selected,
                "skipped_files": skipped,
            }

        emit("chunking_done", chunks=total_chunks, files=len(docs))

        # --- Global REDUCE: doc-notes -> final structured output ---
        summary = self._reduce_final(
            doc_notes,
            task=task,
            reduce_batch=reduce_batch,
            emit=emit,
        )

        summary = _normalize_whitespace(summary)

        if _needs_rewrite(summary):
            emit("rewrite_start")
            prompt_user = (
                "Rewrite the text below to remove forbidden references and meta-language.\n\n"
                "Text:\n"
                "```\n"
                f"{summary}\n"
                "```"
            )
            msg = self.llm.invoke(
                [
                    SystemMessage(content=SYSTEM_REWRITE_PROMPT),
                    HumanMessage(content=prompt_user),
                ],
                self.build_config(tags=["summarizer", "rewrite"]),
            )
            summary2 = _normalize_whitespace(msg.text.strip())
            if summary2:
                summary = summary2
                emit("rewrite_done", changed=True)
            else:
                emit("rewrite_done", changed=False)

        emit("done", chars=len(summary))

        return {
            "summary": summary,
            "selected_files": selected,
            "skipped_files": skipped,
        }

    def _reduce_notes_list(
        self,
        items: list[str],
        *,
        task: str,
        reduce_batch: int,
        emit: Callable[..., None],
        scope: str,
    ) -> str:
        """
        Reduce a list of note blocks down to a single bullet-list note artifact.

        Used for per-file reduction so each file contributes one “doc-notes” unit.
        """
        current = [x for x in items if (x or "").strip()]
        if not current:
            return ""

        round_i = 0
        emit(
            "notes_reduce_start",
            scope=scope,
            items=len(current),
            batch_size=reduce_batch,
        )

        while len(current) > 1:
            round_i += 1
            num_batches = math.ceil(len(current) / reduce_batch)
            emit(
                "notes_reduce_round",
                scope=scope,
                round=round_i,
                items=len(current),
                batches=num_batches,
            )

            nxt: list[str] = []
            batch_idx = 0

            for i in range(0, len(current), reduce_batch):
                batch_idx += 1
                batch = current[i : i + reduce_batch]
                material = "\n\n".join(batch).strip()

                prompt_user = (
                    f"Task:\n{task}\n\n"
                    "Merge the notes below into a single bullet list that preserves distinct, concrete details.\n\n"
                    f"Notes:\n{material}"
                )

                msg = self.llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_NOTES_REDUCE_PROMPT),
                        HumanMessage(content=prompt_user),
                    ],
                    self.build_config(tags=["summarizer", "notes_reduce"]),
                )
                nxt.append(msg.text.strip())
                emit(
                    "notes_reduce_batch_done",
                    scope=scope,
                    round=round_i,
                    batch=batch_idx,
                    batches=num_batches,
                )

            current = nxt

        emit("notes_reduce_done", scope=scope, rounds=round_i)
        return current[0].strip()

    def _reduce_final(
        self,
        doc_notes: list[str],
        *,
        task: str,
        reduce_batch: int,
        emit: Callable[..., None],
    ) -> str:
        """
        Reduce per-file notes into a single final structured summary.
        """
        current = [x for x in doc_notes if (x or "").strip()]
        round_i = 0

        emit("reduce_start", items=len(current), batch_size=reduce_batch)

        while len(current) > 1:
            round_i += 1
            num_batches = math.ceil(len(current) / reduce_batch)
            emit(
                "reduce_round",
                round=round_i,
                items=len(current),
                batches=num_batches,
            )

            nxt: list[str] = []
            batch_idx = 0

            for i in range(0, len(current), reduce_batch):
                batch_idx += 1
                batch = current[i : i + reduce_batch]

                emit(
                    "reduce_batch_start",
                    round=round_i,
                    batch=batch_idx,
                    batches=num_batches,
                    size=len(batch),
                )

                material = "\n\n".join(batch).strip()
                prompt_user = (
                    f"Task:\n{task}\n\n"
                    f"{FINAL_COVERAGE_INSTRUCTION}\n"
                    f"Material:\n{material}"
                )

                msg = self.llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_REDUCE_PROMPT),
                        HumanMessage(content=prompt_user),
                    ],
                    self.build_config(tags=["summarizer", "reduce"]),
                )
                nxt.append(msg.text.strip())
                emit(
                    "reduce_batch_done",
                    round=round_i,
                    batch=batch_idx,
                    batches=num_batches,
                )

            current = nxt

        emit("reduce_done", rounds=round_i)
        return current[0].strip()

    def _select_files(
        self,
        *,
        root: Path,
        recurse: bool,
        allowed_exts: tuple[str, ...],
        max_files: Optional[int],
    ) -> tuple[list[str], list[str]]:
        """
        Select eligible files under `root`.

        - Skips hidden paths (any segment starting with '.').
        - Filters by allowed file extensions.
        - Applies `max_files` deterministically after sorting.
        """

        def eligible(p: Path) -> bool:
            if not p.is_file():
                return False
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                return False
            return p.suffix.lower() in allowed_exts

        paths = [
            p
            for p in (root.rglob("*") if recurse else root.iterdir())
            if eligible(p)
        ]
        rels = sorted(
            [p.relative_to(root).as_posix() for p in paths],
            key=lambda s: s.lower(),
        )

        skipped: list[str] = []
        if max_files is not None and len(rels) > int(max_files):
            skipped.extend(rels[int(max_files) :])
            rels = rels[: int(max_files)]

        return rels, sorted(set(skipped))
