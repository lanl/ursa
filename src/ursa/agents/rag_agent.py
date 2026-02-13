import os
import re
import statistics
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import TypedDict

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings, init_embeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from ursa.agents.base import BaseAgent
from ursa.util.parse import (
    read_docx,
    read_odf,
    read_pdf,
    read_pptx,
    read_text_file,
)

# Curate this for your environment. Start broad, tighten later.
TEXT_EXTENSIONS = {
    # plain text & docs
    ".txt",
    ".md",
    ".rst",
    ".rtf",
    ".tex",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".log",
    ".xml",
    ".html",
    ".htm",
    # source code (common)
    ".py",
    ".pyi",
    ".ipynb",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".java",
    ".kt",
    ".scala",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
}

SPECIAL_TEXT_FILENAMES = {
    "makefile",
    "readme",
    "license",
}

OFFICE_EXTENSIONS = {".docx", ".pptx", ".odt", ".odp"}

# Set a minimum number of characters in a file to
#     to ingest it. Avoids files with minimal content
#     that would be unlikely to give meaningful
#     information to perform RAG on.
MIN_CHARS = 30


class RAGMetadata(TypedDict):
    k: int
    num_results: int
    relevance_scores: list[float]


class RAGState(TypedDict, total=False):
    context: str
    doc_texts: list[str]
    doc_ids: list[str]
    summary: str
    rag_metadata: RAGMetadata


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


def _is_meaningful(text: str) -> bool:
    return len(text) >= MIN_CHARS


class RAGAgent(BaseAgent[RAGState]):
    agent_state = RAGState

    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings | None = None,
        return_k: int = 10,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        database_path: str = "database",
        summaries_path: str = "database",
        vectorstore_path: str = "vectorstore",
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.retriever = None
        self._vs_lock = Lock()
        self.return_k = return_k
        self.embedding = embedding or init_embeddings(
            "openai:text-embedding-3-small"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_path = self.workspace / database_path
        self.summaries_path = self.workspace / summaries_path
        self.vectorstore_path = self.workspace / vectorstore_path

        self.vectorstore_path.mkdir(exist_ok=True, parents=True)
        self.vectorstore = self._open_global_vectorstore()

    @cached_property
    def graph(self):
        return self._build_graph()

    @property
    def _action(self):
        return self.graph

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.vectorstore_path, "_ingested_ids.txt")

    @property
    def manifest_exists(self) -> bool:
        return os.path.exists(self.manifest_path)

    def _open_global_vectorstore(self) -> Chroma:
        return Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embedding,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _paper_exists_in_vectorstore(self, doc_id: str) -> bool:
        try:
            col = self.vectorstore._collection
            res = col.get(where={"id": doc_id}, limit=1)
            return len(res.get("ids", [])) > 0
        except Exception:
            if not self.manifest_exists:
                return False
            with open(self.manifest_path, "r") as f:
                return any(line.strip() == doc_id for line in f)

    def _mark_paper_ingested(self, arxiv_id: str) -> None:
        with open(self.manifest_path, "a") as f:
            f.write(f"{arxiv_id}\n")

    def _ensure_doc_in_vectorstore(self, paper_text: str, doc_id: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs = splitter.create_documents(
            [paper_text], metadatas=[{"id": doc_id}]
        )
        with self._vs_lock:
            if not self._paper_exists_in_vectorstore(doc_id):
                ids = [f"{doc_id}::{i}" for i, _ in enumerate(docs)]
                self.vectorstore.add_documents(docs, ids=ids)
                self._mark_paper_ingested(doc_id)

    def _get_global_retriever(self, k: int = 5):
        return self.vectorstore, self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def _read_docs_node(self, state: RAGState) -> RAGState:
        print("[RAG Agent] Reading Documents....")
        new_state = state.copy()

        base_dir = Path(self.database_path)
        ingestible_paths: list[Path] = []

        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue

            ext = p.suffix.lower()
            is_special_text = p.name.lower() in SPECIAL_TEXT_FILENAMES

            if (
                ext == ".pdf"
                or ext in TEXT_EXTENSIONS
                or is_special_text
                or ext in OFFICE_EXTENSIONS
            ):
                ingestible_paths.append(p)

        candidates: list[tuple[Path, str]] = []
        for p in ingestible_paths:
            doc_id = str(p)
            if not self._paper_exists_in_vectorstore(doc_id):
                candidates.append((p, doc_id))

        papers: list[str] = []
        doc_ids: list[str] = []
        for path, doc_id in tqdm(candidates, desc="RAG parsing text"):
            full_text = ""
            ext = path.suffix.lower()

            try:
                match ext:
                    case ".pdf":
                        full_text = read_pdf(path)
                    case ".odt" | ".odp":
                        full_text = read_odf(path)
                    case ".docx":
                        full_text = read_docx(path)
                    case ".pptx":
                        full_text = read_pptx(path)
                    case _:
                        if (
                            ext in TEXT_EXTENSIONS
                            or path.name.lower() in SPECIAL_TEXT_FILENAMES
                        ):
                            full_text = read_text_file(path)
                        else:
                            # Maybe add a warning here?
                            full_text = f"Unsupported file type: {path.name}"
                # skip files with very few characters to
                #    avoid parsing/rag ingestion problems
                if not _is_meaningful(full_text):
                    continue
            except Exception as e:
                full_text = f"Error loading {path.name}: {e}"

            papers.append(full_text)
            doc_ids.append(doc_id)

        new_state["doc_texts"] = papers
        new_state["doc_ids"] = doc_ids

        return new_state

    def _ingest_docs_node(self, state: RAGState) -> RAGState:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        if "doc_texts" not in state:
            raise RuntimeError("Unexpected error: doc_ids not in state!")

        if "doc_ids" not in state:
            raise RuntimeError("Unexpected error: doc_texts not in state!")

        batch_docs, batch_ids = [], []

        for paper, id in tqdm(
            zip(state["doc_texts"], state["doc_ids"]),
            total=len(state["doc_texts"]),
            desc="RAG Ingesting",
        ):
            cleaned_text = remove_surrogates(paper)
            docs = splitter.create_documents(
                [cleaned_text], metadatas=[{"id": id}]
            )
            ids = [f"{id}::{i}" for i, _ in enumerate(docs)]
            batch_docs.extend(docs)
            batch_ids.extend(ids)

        if state["doc_texts"]:
            print("[RAG Agent] Ingesting Documents Into RAG Database....")
            with self._vs_lock:
                self.vectorstore.add_documents(batch_docs, ids=batch_ids)
                for id in batch_ids:
                    self._mark_paper_ingested(id)

        return state

    def _retrieve_and_summarize_node(self, state: RAGState) -> RAGState:
        print(
            "[RAG Agent] Retrieving Contextually Relevant Information From Database..."
        )
        prompt = ChatPromptTemplate.from_template("""
        You are a scientific assistant responsible for summarizing extracts from research papers, in the context of the following task: {context}

        Summarize the retrieved scientific content below.
        Cite sources by ID when relevant: {source_ids}

        {retrieved_content}
        """)
        chain = prompt | self.llm | StrOutputParser()

        # 2) One retrieval over the global DB with the task context
        try:
            if "context" not in state:
                raise RuntimeError("Unexpected error: context not in state!")

            results = self.vectorstore.similarity_search_with_relevance_scores(
                state["context"], k=self.return_k
            )

            relevance_scores = [score for _, score in results]
        except Exception as e:
            print(f"RAG failed due to: {e}")
            return {**state, "summary": ""}

        source_ids_list = []
        for doc, _ in results:
            aid = doc.metadata.get("id")
            if aid and aid not in source_ids_list:
                source_ids_list.append(aid)
        source_ids = ", ".join(source_ids_list)

        retrieved_content = (
            "\n\n".join(doc.page_content for doc, _ in results)
            if results
            else ""
        )

        print("[RAG Agent] Summarizing Retrieved Information From Database...")
        # 3) One summary based on retrieved chunks
        rag_summary = chain.invoke({
            "retrieved_content": retrieved_content,
            "context": state["context"],
            "source_ids": source_ids,
        })

        # Persist a single file for the batch (optional)
        batch_name = "RAG_summary.txt"
        os.makedirs(self.summaries_path, exist_ok=True)
        with open(os.path.join(self.summaries_path, batch_name), "w") as f:
            f.write(rag_summary)

        # Diagnostics
        if relevance_scores:
            print(f"\nMax Relevance Score: {max(relevance_scores):.4f}")
            print(f"Min Relevance Score: {min(relevance_scores):.4f}")
            print(
                f"Median Relevance Score: {statistics.median(relevance_scores):.4f}\n"
            )
        else:
            print("\nNo RAG results retrieved (score list empty).\n")

        # Return a single-element list by default (preferred)
        return {
            **state,
            "summary": rag_summary,
            "rag_metadata": {
                "k": self.return_k,
                "num_results": len(results),
                "relevance_scores": relevance_scores,
            },
        }

    def _build_graph(self):
        self.add_node(self._read_docs_node)
        self.add_node(self._ingest_docs_node)
        self.add_node(self._retrieve_and_summarize_node)

        self.graph.add_edge("_read_docs_node", "_ingest_docs_node")
        self.graph.add_edge("_ingest_docs_node", "_retrieve_and_summarize_node")

        self.graph.set_entry_point("_read_docs_node")
        self.graph.set_finish_point("_retrieve_and_summarize_node")


# NOTE: Run test in `tests/agents/test_rag_agent/test_rag_agent.py` via:
#
# pytest -s tests/agents/test_rag_agent
#
# OR
#
# uv run pytest -s tests/agents/test_rag_agent
#
# NOTE: You may need to `rm -rf workspace/rag-agent` to remove the vectorstore.
