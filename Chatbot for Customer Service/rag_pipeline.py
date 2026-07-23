# function ask(question):
#     1. query_embedding = embed(question)
#     2. chunks = faiss_search(query_embedding, top_k=3)
#     3. answer = get_answer(question, chunks)
#     return answer

"""
rag_pipeline.py
===============
STAGE 2 - STEP 7: The Full RAG Pipeline (Orchestrator)

What this does:
- Ties ALL steps together into two clean functions:
  1. process_document() → Pipeline 1 (build knowledge base)
  2. ask()             → Pipeline 2 (answer a question)

How it works internally:
PIPELINE 1 (process_document):
  Document → load_document → split_into_chunks → embed_chunks
          → build_vector_store → saved to disk

PIPELINE 2 (ask):
  Question → embed_query → search FAISS → get_answer from Ollama
           → return answer + sources

This is the ONLY file that app.py needs to import.
Everything else is hidden behind these two functions.
"""

import os
import tempfile
from document_loader import load_document, get_document_info
from chunker import split_into_chunks, get_chunk_stats
from embedder import embed_chunks, embed_query
from vector_store import (
    build_vector_store,
    load_vector_store,
    search,
    vector_store_exists,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    METADATA_PATH,
)
from llm_handler import get_answer, summarize_document


# In-memory cache so we don't reload FAISS index on every question
_cache = {
    "index": None,
    "chunks": None,
    "metadata": None,
    "doc_info": None,
    "all_chunks": None,   # for summarization
}


def process_document(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source_name: str = None,
) -> dict:
    """
    PIPELINE 1: Process a document into the vector store.

    Call this when the user uploads a document.

    Args:
        file_path:     Path to the uploaded PDF or DOCX
        chunk_size:    Characters per chunk (default 500)
        chunk_overlap: Overlap between chunks (default 100)
        source_name:   Display name for citations (defaults to filename)

    Returns:
        Dict with processing stats:
        {
            "success": True/False,
            "doc_info": {...},
            "num_chunks": int,
            "chunk_stats": {...},
            "message": "..."
        }
    """
    global _cache

    try:
        # ── Step 1: Get document info ────────────────────────────────────────
        doc_info = get_document_info(file_path)
        if source_name:
            doc_info["file_name"] = source_name

        print(f"\n{'='*50}")
        print(f"Processing: {doc_info['file_name']}")
        print(f"{'='*50}")

        # ── Step 2: Extract text ─────────────────────────────────────────────
        print("\n[Pipeline 1] Step 1/4: Extracting text...")
        raw_text = load_document(file_path)

        # ── Step 3: Chunk the text ───────────────────────────────────────────
        print("\n[Pipeline 1] Step 2/4: Splitting into chunks...")
        chunks = split_into_chunks(raw_text, chunk_size, chunk_overlap)
        chunk_stats = get_chunk_stats(chunks)

        # ── Step 4: Create embeddings ────────────────────────────────────────
        print("\n[Pipeline 1] Step 3/4: Creating embeddings...")
        embeddings = embed_chunks(chunks)

        # ── Step 5: Build and save FAISS index ───────────────────────────────
        print("\n[Pipeline 1] Step 4/4: Building vector store...")
        metadata = [
            {"source": doc_info["file_name"], "chunk_id": i}
            for i in range(len(chunks))
        ]
        build_vector_store(embeddings, chunks, metadata)

        # ── Cache the loaded index for immediate use ─────────────────────────
        _cache["index"], _cache["chunks"], _cache["metadata"] = load_vector_store()
        _cache["doc_info"] = doc_info
        _cache["all_chunks"] = chunks

        print(f"\n✓ Document processed successfully!")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Ready to answer questions.\n")

        return {
            "success": True,
            "doc_info": doc_info,
            "num_chunks": len(chunks),
            "chunk_stats": chunk_stats,
            "message": f"Successfully processed '{doc_info['file_name']}' into {len(chunks)} chunks.",
        }

    except Exception as e:
        print(f"[Pipeline 1] ERROR: {e}")
        return {
            "success": False,
            "doc_info": None,
            "num_chunks": 0,
            "chunk_stats": {},
            "message": f"Error processing document: {str(e)}",
        }


def ask(
    question: str,
    top_k: int = 3,
    chat_history: list[dict] = None,
) -> dict:
    """
    PIPELINE 2: Answer a question using the vector store + Ollama.

    Call this when the user submits a question.

    Args:
        question:     The user's question string
        top_k:        How many chunks to retrieve (default 3)
        chat_history: List of past Q&A pairs for conversation memory
                      Format: [{"question": "...", "answer": "..."}, ...]

    Returns:
        Dict with answer and sources:
        {
            "success": True/False,
            "answer": "...",
            "sources": [...],
            "message": "..."
        }
    """
    global _cache

    if not question or not question.strip():
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "message": "Question cannot be empty.",
        }

    # ── Load index (from cache or disk) ──────────────────────────────────────
    if _cache["index"] is None:
        if not vector_store_exists():
            return {
                "success": False,
                "answer": "",
                "sources": [],
                "message": "No document loaded. Please upload a document first.",
            }
        print("[Pipeline 2] Loading vector store from disk...")
        _cache["index"], _cache["chunks"], _cache["metadata"] = load_vector_store()

    try:
        # ── Step 1: Embed the query ───────────────────────────────────────────
        print(f"\n[Pipeline 2] Question: '{question}'")
        print("[Pipeline 2] Step 1/3: Embedding query...")
        query_embedding = embed_query(question)

        # ── Step 2: Retrieve relevant chunks ─────────────────────────────────
        print("[Pipeline 2] Step 2/3: Searching FAISS...")
        results = search(
            query_embedding,
            _cache["index"],
            _cache["chunks"],
            _cache["metadata"],
            top_k=top_k,
        )
        print(f"  Retrieved {len(results)} chunks (top scores: {[r['score'] for r in results]})")

        # ── Step 3: Generate answer ───────────────────────────────────────────
        print("[Pipeline 2] Step 3/3: Generating answer with Ollama...")
        answer = get_answer(question, results, chat_history)

        return {
            "success": True,
            "answer": answer,
            "sources": results,
            "message": "OK",
        }

    except Exception as e:
        print(f"[Pipeline 2] ERROR: {e}")
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "message": f"Error answering question: {str(e)}",
        }


def summarize(doc_name: str = "document") -> dict:
    """
    Summarize the currently loaded document.
    Used by the Summarize button in the UI.
    """
    global _cache

    if _cache["all_chunks"] is None:
        return {
            "success": False,
            "summary": "",
            "message": "No document loaded. Please upload a document first.",
        }

    try:
        print("[Pipeline] Generating document summary...")
        summary = summarize_document(_cache["all_chunks"], doc_name)
        return {
            "success": True,
            "summary": summary,
            "message": "OK",
        }
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "message": f"Error generating summary: {str(e)}",
        }


def clear_pipeline():
    """Reset the pipeline — clears cache and saved files."""
    global _cache
    _cache = {
        "index": None, "chunks": None,
        "metadata": None, "doc_info": None, "all_chunks": None,
    }

    for path in [FAISS_INDEX_PATH, CHUNKS_PATH, METADATA_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"[Pipeline] Deleted: {path}")

    print("[Pipeline] Cache and files cleared.")


def get_pipeline_status() -> dict:
    """Returns current pipeline status — used by the UI."""
    return {
        "has_document": _cache["index"] is not None or vector_store_exists(),
        "doc_info": _cache.get("doc_info"),
        "num_chunks": len(_cache["chunks"]) if _cache["chunks"] else 0,
    }


# ── Quick end-to-end test — run: python rag_pipeline.py ─────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <path_to_pdf_or_docx>")
        print("\nRunning with sample text test instead...")

        # Create a temporary test file
        sample_text = """
Machine learning (ML) is a type of artificial intelligence that allows software
applications to become more accurate at predicting outcomes without being explicitly
programmed to do so. Machine learning algorithms use historical data as input to
predict new output values.

Supervised learning is the most common type. It uses labeled datasets to train
algorithms that classify data or predict outcomes accurately.

Unsupervised learning uses machine learning algorithms to analyze and cluster
unlabeled datasets. These algorithms discover hidden patterns in data without human
intervention.

Deep learning is part of a broader family of machine learning methods. It uses
neural networks with many layers to model complex patterns in data.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(sample_text)
            test_path = f.name

        # For testing, we'll use a proper temp docx
        print("Please provide a real .pdf or .docx file to test.")
        sys.exit(0)

    file_path = sys.argv[1]

    print("=== PIPELINE 1: Processing Document ===")
    result = process_document(file_path)
    print(f"\nResult: {result['message']}")

    if result["success"]:
        print("\n=== PIPELINE 2: Asking a Question ===")
        response = ask("What is the main topic of this document?")
        print(f"\nAnswer: {response['answer']}")

        if response["sources"]:
            print("\nSources used:")
            for src in response["sources"]:
                print(f"  - {src['source']} (score: {src['score']}%)")