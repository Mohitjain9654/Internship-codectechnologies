# function split_into_chunks(text, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start = start + chunk_size - overlap  # slide with overlap

#     return chunks

# RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

"""
chunker.py
==========
STAGE 1 - STEP 2: Text Chunking

What this does:
- Takes the big raw text string from document_loader
- Splits it into smaller, overlapping pieces (chunks)
- Returns a list of chunk strings

How it works internally:
- Uses LangChain's RecursiveCharacterTextSplitter
- It tries to split at paragraph breaks first (\n\n)
- Then sentence breaks (\n)
- Then word spaces ( )
- This way chunks never cut mid-sentence if possible
- Overlap ensures boundary sentences appear in two chunks so nothing is lost

Why 500 chars / 100 overlap?
- 500 chars ≈ 1-2 paragraphs — enough context for embedding
- 100 char overlap — safety net for boundary sentences
- You can tune these for your documents
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Merge small splits into chunks up to chunk_size, with overlap (stdlib fallback)."""
    docs: list[str] = []
    current: list[str] = []
    total = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)
        extra = sep_len if current else 0
        if total + piece_len + extra > chunk_size and current:
            doc = separator.join(current)
            if doc.strip():
                docs.append(doc)
            while total > chunk_overlap and current:
                removed = current.pop(0)
                total -= len(removed) + (sep_len if current else 0)
        current.append(piece)
        total += piece_len + (sep_len if len(current) > 1 else 0)

    if current:
        doc = separator.join(current)
        if doc.strip():
            docs.append(doc)
    return docs


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Split like RecursiveCharacterTextSplitter when LangChain is not installed."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[-1]
    next_seps = separators[1:] if len(separators) > 1 else [""]

    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            next_seps = [""]
            break
        if sep in text:
            separator = sep
            next_seps = separators[i + 1:] if i + 1 < len(separators) else [""]
            break

    if separator:
        splits = text.split(separator)
        good = []
        for j, s in enumerate(splits):
            if j < len(splits) - 1:
                s = s + separator
            if s:
                good.append(s)
    else:
        good = list(text)

    merged: list[str] = []
    for s in good:
        if len(s) < chunk_size:
            merged.append(s)
        else:
            merged.extend(_split_text_recursive(s, chunk_size, chunk_overlap, next_seps))
    return _merge_splits(merged, separator if separator else "", chunk_size, chunk_overlap)


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Split raw text into overlapping chunks.

    Args:
        text:          Raw extracted text from document_loader
        chunk_size:    Max characters per chunk (default 500)
        chunk_overlap: Characters shared between consecutive chunks (default 100)

    Returns:
        List of text chunk strings
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty. Cannot chunk an empty document.")

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # tries these in order
            length_function=len,
        )
        chunks = splitter.split_text(text)
    else:
        chunks = _split_text_recursive(
            text,
            chunk_size,
            chunk_overlap,
            ["\n\n", "\n", " ", ""],
        )

    # Filter out empty or whitespace-only chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    print(f"[chunker] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return chunks


def get_chunk_stats(chunks: list[str]) -> dict:
    """
    Returns statistics about your chunks.
    Useful for debugging — if avg size is too small, increase chunk_size.
    """
    if not chunks:
        return {}

    lengths = [len(c) for c in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_characters": sum(lengths),
    }


# ── Quick test — run: python chunker.py ─────────────────────────────────────
if __name__ == "__main__":
    # Sample text to test with
    sample_text = """
    Machine learning is a branch of artificial intelligence. It focuses on building
    systems that learn from data. Instead of being explicitly programmed, these systems
    improve their performance through experience.

    Supervised learning is one type of machine learning. In supervised learning, the
    model is trained on labeled data. Each training example has an input and a known
    correct output. The model learns to map inputs to outputs.

    Unsupervised learning is another type. Here, the model works with unlabeled data.
    It tries to find patterns and structure on its own. Clustering is a common
    unsupervised technique.

    Reinforcement learning is a third type. An agent learns by interacting with an
    environment. It receives rewards for good actions and penalties for bad ones.
    Over time it learns to maximize rewards.
    """ * 5  # repeat to simulate a real document

    chunks = split_into_chunks(sample_text)
    stats = get_chunk_stats(chunks)

    print("\n=== Chunk Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== First 3 Chunks ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
