import os
import re
import statistics
from threading import Lock
from typing import List, Optional, TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

from ursa.agents.base import BaseAgent


class RAGState(TypedDict, total=False):
    context: str
    doc_texts: List[str]
    doc_ids: List[str]
    summary: str


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


class RAGAgent(BaseAgent):
    def __init__(
        self,
        llm="openai/o3-mini",
        embedding: Optional[Embeddings] = None,
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
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_path = database_path
        self.summaries_path = summaries_path
        self.vectorstore_path = vectorstore_path
        self.graph = self._build_graph()

        os.makedirs(self.vectorstore_path, exist_ok=True)
        self.vectorstore = self._open_global_vectorstore()

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

    def _read_docs(self, state: RAGState) -> RAGState:
        print("[RAG Agent] Reading Documents....")
        papers = []
        new_state = state.copy()

        pdf_files = [
            f
            for f in os.listdir(self.database_path)
            if f.lower().endswith(".pdf")
        ]

        doc_ids = [
            pdf_filename.rsplit(".pdf", 1)[0] for pdf_filename in pdf_files
        ]
        pdf_files = [
            pdf_filename
            for pdf_filename, id in zip(pdf_files, doc_ids)
            if not self._paper_exists_in_vectorstore(id)
        ]

        for pdf_filename in pdf_files:
            full_text = ""

            try:
                loader = PyPDFLoader(
                    os.path.join(self.database_path, pdf_filename)
                )
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])

            except Exception as e:
                full_text = f"Error loading paper: {e}"

            papers.append(full_text)

        new_state["doc_texts"] = papers
        new_state["doc_ids"] = doc_ids

        return new_state

    def _ingest_docs(self, state: RAGState) -> RAGState:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        batch_docs, batch_ids = [], []
        for paper, id in zip(state["doc_texts"], state["doc_ids"]):
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
                for id in ids:
                    self._mark_paper_ingested(id)

        return state

    def _summarize_node(self, state: RAGState) -> RAGState:
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
            results = self.vectorstore.similarity_search_with_score(
                state["context"], k=self.return_k
            )
        except Exception as e:
            print(f"RAG failed due to: {e}")
            return {**state, "summary": ""}

        source_ids_list = []
        for doc, _ in results:
            aid = doc.metadata.get("id")
            if aid and aid not in source_ids_list:
                source_ids_list.append(aid)
        source_ids = ", ".join(source_ids_list)

        # Compute a simple similarity-based quality score
        relevancy_scores = []
        if results:
            distances = [score for _, score in results]
            sims = [1.0 / (1.0 + d) for d in distances]  # map distance -> [0,1)
            relevancy_scores = sims

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
        if relevancy_scores:
            print(f"\nMax Relevancy Score: {max(relevancy_scores):.4f}")
            print(f"Min Relevancy Score: {min(relevancy_scores):.4f}")
            print(
                f"Median Relevancy Score: {statistics.median(relevancy_scores):.4f}\n"
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
                "relevancy_scores": relevancy_scores,
            },
        }

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("Read Documents", self._read_docs)
        builder.add_node("Ingest Documents", self._ingest_docs)
        builder.add_node("Retrieve and Summarize", self._summarize_node)
        builder.add_edge("Read Documents", "Ingest Documents")
        builder.add_edge("Ingest Documents", "Retrieve and Summarize")

        builder.set_entry_point("Read Documents")
        builder.set_finish_point("Retrieve and Summarize")

        graph = builder.compile()
        return graph

    def run(self, context: str) -> str:
        result = self.graph.invoke({"context": context})

        return result.get("summary", "No summary generated.")


if __name__ == "__main__":
    agent = RAGAgent(database_path="workspace/arxiv_papers_neutron_star")
    result = agent.run(
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )

    print(result)
