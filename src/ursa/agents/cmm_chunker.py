from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from ursa.agents.cmm_taxonomy import (
    detect_commodity_tags,
    detect_subdomain_tags,
    first_temporal_indicator,
    has_numerical_data,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*[:\-]+\s*(\|\s*[:\-]+\s*)+\|?\s*$")


@dataclass
class _Section:
    section_path: str
    text: str


class CMMChunker:
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_tokens: int = 50,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def chunk_document(self, text: str, metadata: dict[str, Any]) -> list[Document]:
        if not text.strip():
            return []

        source_doc_id = str(
            metadata.get("source_doc_id")
            or metadata.get("id")
            or metadata.get("doc_id")
            or metadata.get("source")
            or "unknown_doc"
        )
        source_doc_title = str(
            metadata.get("source_doc_title")
            or metadata.get("title")
            or metadata.get("filename")
            or source_doc_id
        )
        sensitivity = str(metadata.get("sensitivity_level", "public"))
        doc_level_commodity = detect_commodity_tags(text)
        doc_level_subdomain = detect_subdomain_tags(text)
        doc_temporal = str(
            metadata.get("temporal_indicator") or first_temporal_indicator(text)
        )
        data_vintage = str(metadata.get("data_vintage") or doc_temporal)

        sections = (
            self._chunk_markdown(text)
            if "#" in text
            else [_Section(section_path="", text=text)]
        )
        docs: list[Document] = []
        chunk_index = 0
        for section in sections:
            blocks = self._extract_tables(section.text)
            for block_text, block_type in blocks:
                guarded_blocks = self._apply_size_guard([block_text])
                for chunk in guarded_blocks:
                    if not chunk.strip():
                        continue
                    commodity_tags = detect_commodity_tags(
                        chunk, fallback=doc_level_commodity
                    )
                    subdomain_tags = detect_subdomain_tags(
                        chunk, fallback=doc_level_subdomain
                    )
                    temporal_indicator = first_temporal_indicator(chunk) or doc_temporal
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source_doc_id": source_doc_id,
                            "source_doc_title": source_doc_title,
                            "section_path": section.section_path,
                            "chunk_index": chunk_index,
                            "chunk_type": block_type,
                            "commodity_tags": commodity_tags,
                            "subdomain_tags": subdomain_tags,
                            "temporal_indicator": temporal_indicator,
                            "data_vintage": data_vintage,
                            "sensitivity_level": sensitivity,
                            "char_count": len(chunk),
                            "has_numerical_data": has_numerical_data(chunk),
                        },
                    )
                    docs.append(doc)
                    chunk_index += 1
        return docs

    def _chunk_markdown(self, text: str) -> list[_Section]:
        sections: list[_Section] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_path = ""

        def flush():
            nonlocal current_lines, current_path
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(_Section(section_path=current_path, text=body))
            current_lines = []

        for raw_line in text.splitlines():
            m = _HEADING_RE.match(raw_line.strip())
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
                current_path = " > ".join(heading_stack)
            else:
                current_lines.append(raw_line)
        flush()
        if not sections:
            return [_Section(section_path="", text=text)]
        return sections

    def _chunk_plain_text(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paras:
            return [text]
        return paras

    def _extract_tables(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        blocks: list[tuple[str, str]] = []
        prose_buf: list[str] = []
        i = 0

        def flush_prose():
            nonlocal prose_buf
            prose = "\n".join(prose_buf).strip()
            if prose:
                for para in self._chunk_plain_text(prose):
                    blocks.append((para, "prose"))
            prose_buf = []

        while i < len(lines):
            line = lines[i]
            if "|" in line and line.count("|") >= 2:
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if _TABLE_SEPARATOR_RE.match(next_line.strip()):
                    flush_prose()
                    table_lines = [line, next_line]
                    i += 2
                    while i < len(lines):
                        if "|" in lines[i] and lines[i].count("|") >= 2:
                            table_lines.append(lines[i])
                            i += 1
                        else:
                            break
                    blocks.append(("\n".join(table_lines).strip(), "table"))
                    continue
            prose_buf.append(line)
            i += 1

        flush_prose()
        return blocks or [(text, "prose")]

    def _apply_size_guard(self, chunks: list[str]) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            words = chunk.split()
            if len(words) <= self.max_tokens:
                out.append(chunk)
                continue

            sentences = _SENTENCE_SPLIT_RE.split(chunk)
            cur: list[str] = []
            cur_tokens = 0
            for sentence in sentences:
                s_tokens = len(sentence.split())
                if s_tokens > self.max_tokens:
                    if cur:
                        out.append(" ".join(cur).strip())
                        cur = []
                        cur_tokens = 0
                    long_words = sentence.split()
                    step = max(1, self.max_tokens - self.overlap_tokens)
                    start = 0
                    while start < len(long_words):
                        end = min(len(long_words), start + self.max_tokens)
                        out.append(" ".join(long_words[start:end]).strip())
                        if end >= len(long_words):
                            break
                        start += step
                    continue
                if cur and cur_tokens + s_tokens > self.max_tokens:
                    out.append(" ".join(cur).strip())
                    overlap = " ".join(cur).split()[-self.overlap_tokens :]
                    cur = [" ".join(overlap), sentence]
                    cur_tokens = len(overlap) + s_tokens
                else:
                    cur.append(sentence)
                    cur_tokens += s_tokens
            if cur:
                out.append(" ".join(cur).strip())

        if not out:
            return []

        merged: list[str] = []
        for chunk in out:
            tokens = len(chunk.split())
            if merged and tokens < self.min_tokens:
                merged[-1] = merged[-1].rstrip() + "\n\n" + chunk.lstrip()
            else:
                merged.append(chunk)
        return merged
