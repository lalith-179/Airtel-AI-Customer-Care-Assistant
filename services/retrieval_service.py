"""
Retrieval Service.

Responsible ONLY for:
    1. Receiving a user query.
    2. Embedding the query (via Ollama / nomic-embed-text).
    3. Searching the already-built ChromaDB collection.
    4. Returning top-K chunks + metadata above a similarity threshold.

This module NEVER re-chunks documents, never rebuilds embeddings for
existing content, and never touches the offline indexing pipeline. It opens
the ChromaDB collection once (module-level singleton) and reuses it for
every request - the vector store is not re-initialized per query.
"""
import logging
import time
from typing import List, Optional

import chromadb

from config import settings
from services.ollama_service import OllamaError, ollama_service

logger = logging.getLogger("voicerag.retrieval")


class KnowledgeBaseUnavailableError(RuntimeError):
    """Raised when the pre-built ChromaDB collection cannot be opened."""


class RetrievalService:
    def __init__(self, chroma_path: str = None, collection_name: str = None):
        self.chroma_path = chroma_path or settings.CHROMA_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self._client = None
        self._collection = None
        self._load_error: Optional[str] = None
        self._load_collection()

    # ------------------------------------------------------------------
    def _load_collection(self) -> None:
        """Open the existing persistent ChromaDB collection ONCE.

        This intentionally does not create embeddings or documents - if the
        collection does not exist yet, that means `scripts/build_knowledge_base.py`
        has not been run, and the app should surface a clear status rather
        than silently creating an empty collection.
        """
        try:
            self._client = chromadb.PersistentClient(path=self.chroma_path)
            existing = {c.name for c in self._client.list_collections()}
            if self.collection_name not in existing:
                self._load_error = (
                    f"Collection '{self.collection_name}' not found at "
                    f"'{self.chroma_path}'. Run scripts/build_knowledge_base.py first."
                )
                logger.warning(self._load_error)
                return
            self._collection = self._client.get_collection(self.collection_name)
            logger.info(
                "Loaded ChromaDB collection '%s' (%d chunks) from %s",
                self.collection_name, self._collection.count(), self.chroma_path,
            )
        except Exception as exc:  # noqa: BLE001 - surface as status, not a crash
            self._load_error = f"Failed to open ChromaDB at '{self.chroma_path}': {exc}"
            logger.error(self._load_error)

    def reload(self) -> None:
        """Re-open the collection (e.g. after an offline rebuild)."""
        self._client = None
        self._collection = None
        self._load_error = None
        self._load_collection()

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._collection is not None

    def status(self) -> dict:
        if not self.is_ready():
            return {
                "ready": False,
                "collection": self.collection_name,
                "chroma_path": self.chroma_path,
                "error": self._load_error,
                "chunk_count": 0,
            }
        count = self._collection.count()
        meta = self._collection.metadata or {}
        return {
            "ready": True,
            "collection": self.collection_name,
            "chroma_path": self.chroma_path,
            "chunk_count": count,
            "document_count": meta.get("document_count", "unknown"),
            "embedding_model": meta.get("embedding_model", settings.EMBEDDING_MODEL),
            "last_indexed": meta.get("last_indexed", "unknown"),
            "error": None,
        }

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> List[dict]:
        """Embed the query and search the existing vector store.

        Only the incoming query is embedded here - document embeddings were
        produced once, offline, by scripts/build_knowledge_base.py.
        """
        top_k = top_k or settings.TOP_K
        score_threshold = (
            settings.SIMILARITY_THRESHOLD if score_threshold is None else score_threshold
        )

        if not self.is_ready():
            raise KnowledgeBaseUnavailableError(
                self._load_error or "Knowledge base collection is not loaded."
            )

        start = time.time()
        try:
            query_embedding = ollama_service.embed(query, model=settings.EMBEDDING_MODEL)
        except OllamaError as exc:
            raise KnowledgeBaseUnavailableError(f"Query embedding failed: {exc}") from exc

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise KnowledgeBaseUnavailableError(f"ChromaDB query failed: {exc}") from exc

        elapsed = time.time() - start

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: List[dict] = []
        for text, meta, distance in zip(docs, metas, distances):
            # Chroma's default space is L2 distance on normalized embeddings
            # for nomic-embed-text this approximates cosine distance; convert
            # to a 0..1 "similarity-like" score for threshold comparisons.
            similarity = max(0.0, 1.0 - distance)
            if similarity < score_threshold:
                continue
            matches.append(
                {
                    "text": text,
                    "score": round(similarity, 4),
                    "source_url": meta.get("url", ""),
                    "source_title": meta.get("title", "Untitled"),
                    "category": meta.get("category", "general"),
                    "document_type": meta.get("document_type", "faq"),
                    "chunk_index": meta.get("chunk_index"),
                }
            )

        logger.info(
            "retrieval query=%r top_k=%d threshold=%.2f matched=%d/%d took=%.2fs",
            query[:80], top_k, score_threshold, len(matches), len(docs), elapsed,
        )
        return matches


# Module-level singleton - collection is opened once at import/startup time.
retrieval_service = RetrievalService()
