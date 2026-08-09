"""
ONE-TIME / MANUAL knowledge-base indexing pipeline.

    SOURCE DOCUMENTS
        |
        v
    Document Loading      (PDF via PyMuPDF, .txt/.md)
        |
        v
    Text Extraction
        |
        v
    Cleaning
        |
        v
    Chunking               (config.settings.CHUNK_SIZE / CHUNK_OVERLAP)
        |
        v
    Metadata attachment
        |
        v
    Embedding               (nomic-embed-text via Ollama)
        |
        v
    Vector Storage           (ChromaDB, persisted to knowledge_base/chroma)

Run this manually whenever the Airtel source documents change:

    python scripts/build_knowledge_base.py --source data/sample_docs

This script is NEVER imported or invoked by the runtime Flask app - the
app (services/retrieval_service.py) only ever opens the collection this
script produces and queries it.
"""
import argparse
import hashlib
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from config import settings  # noqa: E402
from services.ollama_service import OllamaError, ollama_service  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("build_knowledge_base")


# ---------------------------------------------------------------------------
# 1. Document loading + text extraction
# ---------------------------------------------------------------------------
def load_pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_text(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported source file type: {suffix}")


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Chunking (configurable via config/settings.py)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# 4. Metadata
# ---------------------------------------------------------------------------
def infer_category(filename: str) -> str:
    name = filename.lower()
    for key in ("prepaid", "postpaid", "broadband", "wifi", "recharge", "billing",
                "sim", "5g", "porting", "dth", "app", "plans", "complaints"):
        if key in name:
            return key
    return "general"


def build_metadata(source_path: Path, chunk_index: int, content: str, title: str, url: str) -> dict:
    return {
        "source": source_path.name,
        "url": url,
        "title": title,
        "category": infer_category(source_path.stem),
        "document_type": "faq" if "faq" in source_path.stem.lower() else "support_doc",
        "chunk_index": chunk_index,
        "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------
# 5. Embedding + 6. Vector storage
# ---------------------------------------------------------------------------
def get_or_create_collection(client: "chromadb.PersistentClient"):
    existing = {c.name for c in client.list_collections()}
    if settings.COLLECTION_NAME in existing:
        client.delete_collection(settings.COLLECTION_NAME)
        logger.info("Existing collection '%s' dropped for a fresh rebuild.", settings.COLLECTION_NAME)
    return client.create_collection(
        name=settings.COLLECTION_NAME,
        metadata={
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "last_indexed": datetime.now(timezone.utc).isoformat(),
            "document_count": 0,  # updated after ingestion
        },
    )


def build_knowledge_base(source_dir: Path, base_url_prefix: str = "https://www.airtel.in/support") -> None:
    if not ollama_service.is_available():
        logger.error(
            "Ollama is not reachable at %s. Start Ollama and ensure '%s' is pulled.",
            settings.OLLAMA_BASE_URL, settings.EMBEDDING_MODEL,
        )
        sys.exit(1)

    source_files = sorted(
        [p for p in source_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    )
    if not source_files:
        logger.error("No .pdf/.txt/.md source documents found under %s", source_dir)
        sys.exit(1)

    logger.info("Found %d source document(s) under %s", len(source_files), source_dir)

    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = get_or_create_collection(client)

    total_chunks = 0
    start_time = time.time()

    for doc_index, path in enumerate(source_files, start=1):
        logger.info("[%d/%d] Processing %s", doc_index, len(source_files), path.name)
        try:
            raw_text = load_document(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (extraction failed: %s)", path.name, exc)
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            logger.warning("Skipping %s (no extractable text)", path.name)
            continue

        chunks = chunk_text(cleaned)
        title = path.stem.replace("_", " ").replace("-", " ").title()
        url = f"{base_url_prefix.rstrip('/')}/{path.stem}"

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            try:
                embedding = ollama_service.embed(chunk, model=settings.EMBEDDING_MODEL)
            except OllamaError as exc:
                logger.warning("Embedding failed for %s chunk %d: %s", path.name, i, exc)
                continue

            meta = build_metadata(path, i, chunk, title, url)
            chunk_id = f"{path.stem}-{i}-{meta['content_hash']}"

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append(meta)

        if ids:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            total_chunks += len(ids)
            logger.info("  -> indexed %d chunks from %s", len(ids), path.name)

    elapsed = time.time() - start_time
    collection.modify(
        metadata={
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "last_indexed": datetime.now(timezone.utc).isoformat(),
            "document_count": len(source_files),
        }
    )

    logger.info(
        "Knowledge base build complete: %d documents, %d chunks, %.1fs, collection='%s', path='%s'",
        len(source_files), total_chunks, elapsed, settings.COLLECTION_NAME, settings.CHROMA_PATH,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="One-time Airtel knowledge base builder")
    parser.add_argument(
        "--source", type=str, default="data/sample_docs",
        help="Directory containing source .pdf/.txt/.md documents",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_directory = Path(args.source)
    if not source_directory.is_absolute():
        source_directory = Path(__file__).resolve().parent.parent / source_directory
    build_knowledge_base(source_directory)
