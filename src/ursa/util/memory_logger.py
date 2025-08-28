import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document


class AgentMemory:
    """
    Simple wrapper around a persistent Chroma vector-store for agent-conversation memory.

    Parameters
    ----------
    path : str | Path | None
        Where to keep the on-disk Chroma DB.  If *None*, a folder called
        ``agent_memory_db`` is created in the package’s base directory.
    collection_name : str
        Name of the Chroma collection.
    embedding_model : <TODO> | None
        the embedding model

    Notes
    -----
    * Requires `langchain-chroma`, and `chromadb`.
    """

    @classmethod
    def get_db_path(cls, path: Optional[str | Path]) -> Path:
        match path:
            case None:
                return Path.home() / ".cache" / "ursa" / "rag" / "db"
            case str():
                return Path(str)
            case Path():
                return path
            case _:
                raise TypeError(
                    f"Type of path is `{type(path)}` "
                    "but `Optional[str | Path]` was expected."
                )

    def __init__(
        self,
        embedding_model,
        path: Optional[str | Path] = None,
        collection_name: str = "agent_memory",
    ) -> None:
        self.path = self.get_db_path(path)
        self.collection_name = collection_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.embeddings = embedding_model

        # If a DB already exists, load it; otherwise defer creation until `build_index`.
        self.vectorstore: Optional[Chroma] = None
        if any(self.path.iterdir()):
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.path),
            )

    # --------------------------------------------------------------------- #
    # ❶ Build & index a brand-new database                                   #
    # --------------------------------------------------------------------- #
    def build_index(
        self,
        chunks: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Create a fresh vector store from ``chunks``.  Existing data (if any)
        are overwritten.

        Parameters
        ----------
        chunks : Sequence[str]
            Text snippets (already chunked) to embed.
        metadatas : Sequence[dict] | None
            Optional metadata dict for each chunk, same length as ``chunks``.
        """
        docs = [
            Document(
                page_content=text, metadata=metadatas[i] if metadatas else {}
            )
            for i, text in enumerate(chunks)
        ]

        # Create (or overwrite) the collection
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.path),
        )

    # --------------------------------------------------------------------- #
    # ❷ Add new chunks and re-index                                          #
    # --------------------------------------------------------------------- #
    def add_memories(
        self,
        new_chunks: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Append new text chunks to the existing store (must call `build_index`
        first if the DB is empty).

        Raises
        ------
        RuntimeError
            If the vector store is not yet initialised.
        """
        if self.vectorstore is None:
            self.build_index(new_chunks, metadatas)
            print("----- Vector store initialised -----")

        docs = []
        for i, text in enumerate(new_chunks):
            if len(text) > 0:  # only add non-empty documents
                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadatas[i] if metadatas else {},
                    )
                )
        self.vectorstore.add_documents(docs)

    # --------------------------------------------------------------------- #
    # ❸ Retrieve relevant chunks (RAG query)                                 #
    # --------------------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        k: int = 4,
        with_scores: bool = False,
        **search_kwargs,
    ):
        """
        Return the *k* most similar chunks for `query`.

        Parameters
        ----------
        query : str
            Natural-language question or statement.
        k : int
            How many results to return.
        with_scores : bool
            If True, also return similarity scores.
        **search_kwargs
            Extra kwargs forwarded to Chroma’s ``similarity_search*`` helpers.

        Returns
        -------
        list[Document] | list[tuple[Document, float]]
        """
        if self.vectorstore is None:
            return ["None"]

        if with_scores:
            return self.vectorstore.similarity_search_with_score(
                query, k=k, **search_kwargs
            )
        return self.vectorstore.similarity_search(query, k=k, **search_kwargs)


def delete_database(path: Optional[str | Path] = None):
    """
    Simple wrapper around a persistent Chroma vector-store for agent-conversation memory.

    Parameters
    ----------
    path : str | Path | None
        Where the on-disk Chroma DB is for deleting.  If *None*, a folder called
        ``agent_memory_db`` is created in the package’s base directory.
    """
    db_path = AgentMemory.get_db_path(path)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Database: {db_path} has been deleted.")
    else:
        print("No database found to delete.")
