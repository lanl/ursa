#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ursa.agents.cmm_chunker import CMMChunker
from ursa.agents.cmm_embeddings import init_embeddings
from ursa.agents.cmm_vectorstore import init_vectorstore
from ursa.util.parse import (
    OFFICE_EXTENSIONS,
    SPECIAL_TEXT_FILENAMES,
    TEXT_EXTENSIONS,
    read_text_from_file,
)

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    "node_modules",
    "site-packages",
}


def _normalize_extension(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return ""
    return value if value.startswith(".") else f".{value}"


def _normalize_extensions(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {ext for ext in (_normalize_extension(v) for v in values) if ext}


def _iter_ingestible_files(
    corpus_path: Path,
    include_extensions: set[str] | None,
    exclude_extensions: set[str],
) -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(corpus_path):
        dirnames[:] = [
            d
            for d in dirnames
            if d.lower() not in EXCLUDED_DIR_NAMES
        ]
        root_path = Path(root)
        for filename_raw in filenames:
            filename = filename_raw.lower()
            path = root_path / filename_raw
            ext = path.suffix.lower()
            ingestible = (
                ext == ".pdf"
                or ext in TEXT_EXTENSIONS
                or filename in SPECIAL_TEXT_FILENAMES
                or ext in OFFICE_EXTENSIONS
            )
            if not ingestible:
                continue
            if ext in exclude_extensions:
                continue
            if include_extensions is not None and ext not in include_extensions:
                if filename not in SPECIAL_TEXT_FILENAMES:
                    continue
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backend-agnostic CMM corpus reindex utility."
    )
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--vectorstore-path", default="cmm_vectorstore")
    parser.add_argument("--backend", default="chroma")
    parser.add_argument(
        "--embedding-model",
        default="openai:text-embedding-3-large",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=3072)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=20,
    )
    parser.add_argument("--collection-name", default="cmm_chunks")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument(
        "--include-extension",
        action="append",
        dest="include_extensions",
        default=None,
    )
    parser.add_argument(
        "--exclude-extension",
        action="append",
        dest="exclude_extensions",
        default=[],
    )
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=0,
        help="Cap chunks ingested from each source document (0 = no cap).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip source files already present in _ingested_ids.txt.",
    )
    parser.add_argument(
        "--flush-docs",
        type=int,
        default=50,
        help="Number of source documents to batch before vectorstore insert.",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path).expanduser().resolve()
    vectorstore_path = Path(args.vectorstore_path).expanduser().resolve()
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    include_extensions = (
        _normalize_extensions(args.include_extensions)
        if args.include_extensions
        else None
    )
    exclude_extensions = _normalize_extensions(args.exclude_extensions)

    embedding = init_embeddings(
        args.embedding_model,
        dimensions=args.embedding_dimensions,
        batch_size=max(1, args.embedding_batch_size),
    )
    vectorstore = init_vectorstore(
        backend=args.backend,
        persist_directory=vectorstore_path,
        embedding_model=embedding,
        collection_name=args.collection_name,
    )
    manifest_path = vectorstore_path / "_ingested_ids.txt"
    if args.reset:
        vectorstore.delete_collection()
        if manifest_path.exists():
            manifest_path.unlink()

    chunker = CMMChunker(
        max_tokens=max(64, args.chunk_size // 2),
        overlap_tokens=max(0, args.chunk_overlap // 4),
        min_tokens=max(20, min(args.chunk_size // 6, 120)),
    )

    files = _iter_ingestible_files(
        corpus_path=corpus_path,
        include_extensions=include_extensions,
        exclude_extensions=exclude_extensions,
    )
    existing_ids: set[str] = set()
    if args.skip_existing and manifest_path.exists():
        existing_ids = {
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        files = [path for path in files if str(path) not in existing_ids]
    if args.max_docs > 0:
        files = files[: args.max_docs]

    commodity_counts: Counter[str] = Counter()
    subdomain_counts: Counter[str] = Counter()
    docs_indexed = 0
    chunks_indexed = 0
    ingested_doc_ids: set[str] = set()
    batched_docs = []
    batched_source_ids: list[str] = []

    def flush_batch() -> None:
        nonlocal batched_docs
        nonlocal batched_source_ids
        if not batched_docs:
            return
        vectorstore.add_documents(batched_docs)
        batched_docs = []
        batched_source_ids = []

    for path in tqdm(files, desc="Reindex corpus"):
        text = read_text_from_file(path)
        if len(text) < args.min_chars:
            continue
        docs = chunker.chunk_document(
            text,
            metadata={
                "source_doc_id": str(path),
                "source_doc_title": path.name,
            },
        )
        if not docs:
            continue
        if args.max_chunks_per_doc > 0:
            docs = docs[: args.max_chunks_per_doc]
        batched_docs.extend(docs)
        batched_source_ids.append(str(path))
        if len(batched_source_ids) >= max(1, args.flush_docs):
            flush_batch()
        docs_indexed += 1
        chunks_indexed += len(docs)
        ingested_doc_ids.add(str(path))
        for doc in docs:
            commodity_counts.update(doc.metadata.get("commodity_tags", []))
            subdomain_counts.update(doc.metadata.get("subdomain_tags", []))

    flush_batch()

    if manifest_path.exists():
        existing_ids = {
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    merged_ids = sorted(existing_ids.union(ingested_doc_ids))
    manifest_path.write_text(
        "\n".join(merged_ids) + ("\n" if merged_ids else ""),
        encoding="utf-8",
    )

    print(f"Docs indexed: {docs_indexed}")
    print(f"Chunks indexed: {chunks_indexed}")
    print(f"Vectorstore count: {vectorstore.count()}")
    print("Commodity tag counts:")
    for tag, count in sorted(commodity_counts.items()):
        print(f"  {tag}: {count}")
    print("Subdomain tag counts:")
    for tag, count in sorted(subdomain_counts.items()):
        print(f"  {tag}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
