"""
vectorstore.py — Build and query a ChromaDB vector store from filing chunks.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — free, fast,
runs locally. Persists to disk so re-runs skip re-embedding entirely.
"""

import hashlib
from pathlib import Path
from typing import Callable
import chromadb
from chromadb.utils import embedding_functions


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = Path("data/chromadb")   # All collections saved here on disk
BATCH_SIZE = 50                        # Smaller batches = more frequent progress updates


def _get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client, creating the directory if needed."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def _collection_name(chunks: list[str]) -> str:
    """Stable collection name derived from file content — same file = same name."""
    content_hash = hashlib.md5("".join(chunks[:10]).encode()).hexdigest()[:12]
    return f"filing_{content_hash}"


def build_vectorstore(
    chunks: list[str],
    on_progress: Callable[[float, str], None] | None = None,
) -> chromadb.Collection:
    """
    Embed chunks and persist to disk. If the collection already exists on disk
    (same file content hash), skip embedding entirely and return immediately.

    Args:
        chunks: Text chunks to embed.
        on_progress: Optional callback(fraction_0_to_1, message) for progress UI.

    Returns:
        ChromaDB Collection ready for querying.
    """
    def progress(frac: float, msg: str):
        if on_progress:
            on_progress(frac, msg)

    client = _get_client()
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection_name = _collection_name(chunks)

    # ── Check if already persisted ────────────────────────────────────────────
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=emb_fn,
        )
        # Verify it has the expected number of docs (not a partial write)
        if collection.count() == len(chunks):
            progress(1.0, f"✅ Loaded from disk cache ({len(chunks)} chunks)")
            print(f"[vectorstore] Cache hit: '{collection_name}' ({len(chunks)} chunks)")
            return collection
        else:
            # Partial — delete and rebuild
            client.delete_collection(collection_name)

    # ── Fresh build ───────────────────────────────────────────────────────────
    progress(0.0, "Creating vector index…")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
        )
        frac = (batch_idx + 1) / total_batches
        chunks_done = min(i + BATCH_SIZE, len(chunks))
        progress(frac, f"Embedding chunks… {chunks_done}/{len(chunks)}")

    progress(1.0, f"✅ Embedded & saved to disk ({len(chunks)} chunks)")
    print(f"[vectorstore] Built & persisted '{collection_name}' ({len(chunks)} chunks)")
    return collection


def is_cached(chunks: list[str]) -> bool:
    """Return True if this exact file is already embedded on disk."""
    client = _get_client()
    name = _collection_name(chunks)
    existing = [c.name for c in client.list_collections()]
    if name not in existing:
        return False
    col = client.get_collection(name)
    return col.count() == len(chunks)


def query_vectorstore(collection: chromadb.Collection, query: str, k: int = 6) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns:
        Concatenated string of relevant chunks, separated by newlines.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
    )
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs)