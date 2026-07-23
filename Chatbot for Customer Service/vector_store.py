# # SAVING (Pipeline 1):
# function build_vector_store(embeddings, chunks):
#     index = faiss.IndexFlatL2(embedding_dimension)  # 384 for MiniLM
#     index.add(embeddings)                           # add all vectors
#     faiss.write_index(index, "vector_store.faiss")  # save to disk
#     save chunks to "chunks.pkl"                     # save text separately

# # SEARCHING (Pipeline 2):
# function search(query_embedding, top_k=3):
#     index = faiss.read_index("vector_store.faiss")
#     distances, indices = index.search(query_embedding, top_k)
#     return [chunks[i] for i in indices[0]]          # return matching text

"""
vector_store.py
===============
STAGE 1 - STEP 4: FAISS Vector Store

What this does:
- Stores all chunk embeddings in a FAISS index
- Saves the index + chunks to disk so you don't re-process every time
- Searches for the most relevant chunks given a query embedding

How it works internally:
- FAISS (Facebook AI Similarity Search) is a specialized vector database
- IndexFlatL2 does exact nearest-neighbor search using L2 (Euclidean) distance
- Lower distance = more similar
- FAISS only stores vectors — we save the original chunk text separately as a .pkl file
- When you search, FAISS returns indices → we use those to look up the original text

Files saved to disk:
- vector_store.faiss  → the FAISS index (the vectors)
- chunks.pkl          → the original text chunks (the words)
- metadata.pkl        → source info per chunk (for citations)
"""

import os
import pickle
import numpy as np
import faiss
from embedder import get_embedding_dimension

# Default paths for saved files
FAISS_INDEX_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"
METADATA_PATH = "metadata.pkl"


def build_vector_store(
    embeddings: np.ndarray,
    chunks: list[str],
    metadata: list[dict] = None,
    index_path: str = FAISS_INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
    metadata_path: str = METADATA_PATH,
) -> None:
    """
    Build a FAISS index from embeddings and save everything to disk.

    Args:
        embeddings:    numpy array shape (num_chunks, 384) from embedder.py
        chunks:        list of original text strings from chunker.py
        metadata:      list of dicts with source info per chunk (optional)
                       e.g. [{"source": "doc.pdf", "chunk_id": 0}, ...]
        index_path:    where to save the FAISS index file
        chunks_path:   where to save the text chunks
        metadata_path: where to save the metadata
    """
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks"
        )

    dimension = embeddings.shape[1]
    print(f"[vector_store] Building FAISS index with {len(chunks)} vectors (dim={dimension})")

    # Create a flat L2 index (exact search, good for up to ~100k chunks)
    index = faiss.IndexFlatL2(dimension)

    # FAISS requires float32
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)

    print(f"[vector_store] Index built. Total vectors stored: {index.ntotal}")

    # Save the FAISS index
    faiss.write_index(index, index_path)
    print(f"[vector_store] FAISS index saved to: {index_path}")

    # Save the text chunks (FAISS doesn't store text, only vectors)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[vector_store] Chunks saved to: {chunks_path}")

    # Save metadata (source citations)
    if metadata is None:
        metadata = [{"source": "unknown", "chunk_id": i} for i in range(len(chunks))]

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[vector_store] Metadata saved to: {metadata_path}")


def load_vector_store(
    index_path: str = FAISS_INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple:
    """
    Load a previously saved FAISS index, chunks, and metadata from disk.

    Returns:
        Tuple of (faiss_index, chunks_list, metadata_list)
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Did you run build_vector_store first?"
        )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found at '{chunks_path}'.")

    index = faiss.read_index(index_path)
    print(f"[vector_store] Loaded FAISS index with {index.ntotal} vectors")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    metadata = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

    return index, chunks, metadata


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[str],
    metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Search the FAISS index for the most relevant chunks.

    Args:
        query_embedding: shape (1, 384) from embedder.embed_query()
        index:           loaded FAISS index
        chunks:          list of original text strings
        metadata:        list of metadata dicts
        top_k:           how many results to return (default 3)

    Returns:
        List of dicts, each with:
        - "text":       the matching chunk text
        - "score":      relevance score (0-100, higher = more relevant)
        - "source":     source filename
        - "chunk_id":   chunk index
    """
    query_f32 = query_embedding.astype(np.float32)

    # FAISS returns distances (L2) and indices
    distances, indices = index.search(query_f32, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 if fewer results than top_k

        # Convert L2 distance to a 0-100 relevance score
        # Lower distance = more similar = higher score
        # We cap at 100 and floor at 0
        relevance_score = max(0, round(100 - float(dist) * 25, 1))

        result = {
            "text": chunks[idx],
            "score": relevance_score,
            "source": metadata[idx].get("source", "unknown") if metadata else "unknown",
            "chunk_id": int(idx),
        }
        results.append(result)

    return results


def vector_store_exists(
    index_path: str = FAISS_INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
) -> bool:
    """Check if a saved vector store already exists on disk."""
    return os.path.exists(index_path) and os.path.exists(chunks_path)


# ── Quick test — run: python vector_store.py ────────────────────────────────
if __name__ == "__main__":
    from embedder import embed_chunks, embed_query

    # Test data
    test_chunks = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is great for data science and machine learning.",
        "FAISS is a library for efficient similarity search.",
        "Transformers are powerful models for NLP tasks.",
    ]

    test_metadata = [
        {"source": "test_doc.pdf", "chunk_id": i}
        for i in range(len(test_chunks))
    ]

    print("=== Building test vector store ===")
    embeddings = embed_chunks(test_chunks)
    build_vector_store(
        embeddings, test_chunks, test_metadata,
        index_path="test_store.faiss",
        chunks_path="test_chunks.pkl",
        metadata_path="test_metadata.pkl",
    )

    print("\n=== Loading and searching ===")
    index, chunks, metadata = load_vector_store(
        index_path="test_store.faiss",
        chunks_path="test_chunks.pkl",
        metadata_path="test_metadata.pkl",
    )

    query_emb = embed_query("What is machine learning?")
    results = search(query_emb, index, chunks, metadata, top_k=3)

    print("\nQuery: 'What is machine learning?'")
    print("Top 3 results:")
    for i, r in enumerate(results):
        print(f"\n  Result {i+1} (score: {r['score']})")
        print(f"  Source: {r['source']}")
        print(f"  Text: {r['text']}")

    # Clean up test files
    import os
    for f in ["test_store.faiss", "test_chunks.pkl", "test_metadata.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    print("\n[Test passed] Vector store works correctly!")