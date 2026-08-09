"""
RAG Service.

Sits above RetrievalService and turns raw retrieved chunks into:
  - a single context string suitable for the response-generation prompt
  - a deduplicated, UI-friendly list of sources

Retrieval logic itself lives in retrieval_service.py; this module is purely
about shaping retrieved data for the LLM agents and the frontend.
"""
import logging
from typing import List, Tuple

from services.retrieval_service import KnowledgeBaseUnavailableError, retrieval_service

logger = logging.getLogger("voicerag.rag")


def get_context_and_sources(
    query: str, top_k: int = None, score_threshold: float = None
) -> Tuple[str, List[dict], List[dict]]:
    """Returns (context_text, sources, raw_matches).

    Raises KnowledgeBaseUnavailableError if the pre-built collection can't
    be queried - callers should turn that into a safe fallback response,
    never a stack trace shown to the user.
    """
    matches = retrieval_service.retrieve(query, top_k=top_k, score_threshold=score_threshold)

    if not matches:
        return "", [], []

    context_parts = []
    for i, m in enumerate(matches, start=1):
        context_parts.append(
            f"[Document {i}] (source: {m['source_title']}, category: {m['category']})\n{m['text']}"
        )
    context_text = "\n\n".join(context_parts)

    seen = set()
    sources = []
    for m in matches:
        key = (m["source_title"], m["source_url"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": m["source_title"],
                "url": m["source_url"],
                "category": m["category"],
            }
        )

    return context_text, sources, matches


def knowledge_base_status() -> dict:
    return retrieval_service.status()
