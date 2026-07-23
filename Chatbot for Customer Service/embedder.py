# load model: SentenceTransformer("all-MiniLM-L6-v2")

# function embed_chunks(chunks):
#     embeddings = model.encode(chunks)   # returns a 2D numpy array
#     return embeddings                   # shape: (num_chunks, 384)

"""
embedder.py
===========
STAGE 1 - STEP 3: Text Embeddings

What this does:
- Loads a local embedding model (no API key needed — runs on your machine)
- Converts text chunks into vectors (lists of numbers)
- Also converts user queries into vectors for search

How it works internally:
- Model: all-MiniLM-L6-v2 (free, fast, 384-dimensional vectors)
- Each chunk becomes a numpy array of 384 floats
- Similar chunks produce vectors that are "close" in space
- This is what makes semantic search possible

Why all-MiniLM-L6-v2?
- Small (80MB), fast, works on CPU
- Good quality for English text
- Perfect for a beginner project
- Downloads automatically on first run
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model instance — loaded once, reused across calls
# This avoids reloading the 80MB model every time
_model = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (only once, cached globally)."""
    global _model
    if _model is None:
        print("[embedder] Loading embedding model (first time may take ~30 seconds)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] Model loaded successfully")
    return _model


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Convert a list of text chunks into a 2D numpy array of embeddings.

    Args:
        chunks: List of text strings from chunker.py

    Returns:
        numpy array of shape (num_chunks, 384)
        Each row = embedding vector for one chunk
    """
    if not chunks:
        raise ValueError("No chunks provided to embed.")

    model = _get_model()

    print(f"[embedder] Embedding {len(chunks)} chunks...")

    # batch_size=32: process 32 chunks at a time (memory efficient)
    # show_progress_bar=True: shows a progress bar during embedding
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"[embedder] Done. Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Convert a single user query into an embedding vector.

    Args:
        query: The user's question string

    Returns:
        numpy array of shape (1, 384) — ready for FAISS search
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = _get_model()

    # encode returns shape (384,) for a single string
    # We reshape to (1, 384) because FAISS expects 2D input
    embedding = model.encode([query], convert_to_numpy=True)

    return embedding  # shape: (1, 384)


def get_embedding_dimension() -> int:
    """Returns the vector size — needed when creating the FAISS index."""
    return _get_model().get_sentence_embedding_dimension()


# ── Quick test — run: python embedder.py ────────────────────────────────────
if __name__ == "__main__":
    test_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language for data science.",
    ]

    print("=== Testing chunk embedding ===")
    embeddings = embed_chunks(test_chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First vector (first 5 values): {embeddings[0][:5]}")

    print("\n=== Testing query embedding ===")
    query_emb = embed_query("What is machine learning?")
    print(f"Query embedding shape: {query_emb.shape}")

    print(f"\n=== Embedding dimension: {get_embedding_dimension()} ===")