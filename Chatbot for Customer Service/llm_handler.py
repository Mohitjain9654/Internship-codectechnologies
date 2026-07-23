# function get_answer(question, retrieved_chunks):

#     context = join all retrieved_chunks with "\n\n"

#     prompt = f"""
#     You are a helpful assistant. Answer the question using ONLY
#     the context provided below. If the answer is not in the context,
#     say "I don't know based on the document."

#     Context:
#     {context}

#     Question: {question}
#     Answer:
#     """

#     response = llm.chat(prompt)
#     return response.text

"""
llm_handler.py
==============
STAGE 2 - STEP 6: LLM Answer Generation (Ollama - local)

What this does:
- Connects to a local Ollama server (no API key required)
- Builds a prompt combining retrieved chunks + user question
- Sends the prompt to Ollama
- Returns the generated answer

How it works internally:
- We build a "grounded" prompt: we paste the relevant document chunks
  into the prompt as "Context", then ask the question
- This forces the local model to answer FROM the document context
- This is the core idea of RAG — the LLM reads your document in real time

Install Ollama from: https://ollama.com
"""

import os
import json
from urllib import error, request
from dotenv import load_dotenv

# Load optional OLLAMA settings from .env file
load_dotenv()


def _format_llm_error(error: Exception) -> str:
    """Convert Ollama/client errors into short, user-friendly messages."""
    error_msg = str(error)
    upper_msg = error_msg.upper()

    if "CONNECTION REFUSED" in upper_msg or "FAILED TO ESTABLISH A NEW CONNECTION" in upper_msg:
        return (
            "Error: Ollama server is not running. Start Ollama first "
            "(example: `ollama serve`) and ensure it's reachable."
        )
    if "MODEL" in upper_msg and "NOT FOUND" in upper_msg:
        return "Error: Ollama model not found. Pull it first (example: `ollama pull llama3.2`)."
    return f"Error generating answer: {error_msg}"


def _get_ollama_config() -> tuple[str, str]:
    """Read Ollama model/base URL from environment."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return model, base_url


def _ollama_generate(prompt: str) -> str:
    """Call Ollama /api/generate and return plain text output."""
    model, base_url = _get_ollama_config()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    req = request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("response", "").strip()
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {e.code}: {details}") from e
    except error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e.reason}") from e


def get_answer(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] = None,
) -> str:
    """
    Generate an answer to the question using retrieved chunks as context.

    Args:
        question:         The user's question string
        retrieved_chunks: List of result dicts from vector_store.search()
                          Each has "text", "score", "source", "chunk_id"
        chat_history:     Optional list of previous Q&A pairs for memory
                          Format: [{"question": "...", "answer": "..."}, ...]

    Returns:
        Answer string from Ollama
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I could not find any relevant information in the uploaded document."

    # Build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source: {chunk['source']} | Relevance: {chunk['score']}%]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build chat history block (for memory feature)
    history_block = ""
    if chat_history:
        # Only use last 4 exchanges to avoid context overload
        recent = chat_history[-4:]
        history_lines = []
        for exchange in recent:
            history_lines.append(f"User: {exchange['question']}")
            history_lines.append(f"Assistant: {exchange['answer']}")
        history_block = "\n".join(history_lines)

    # Build the full prompt
    # We explicitly tell the model to ONLY use the context provided
    prompt = f"""You are a helpful document assistant. Your job is to answer questions
based ONLY on the document context provided below.

RULES:
1. Answer using ONLY the information in the context below.
2. If the answer is not in the context, say exactly: "I couldn't find that information in the uploaded document."
3. Be concise and clear.
4. If quoting from the document, mention which source it came from.

{"PREVIOUS CONVERSATION:" + chr(10) + history_block + chr(10) if history_block else ""}
DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

    print(f"[llm_handler] Sending prompt to Ollama ({len(prompt)} chars)...")

    try:
        answer = _ollama_generate(prompt)
        print(f"[llm_handler] Received answer ({len(answer)} chars)")
        return answer

    except Exception as e:
        return _format_llm_error(e)


def summarize_document(chunks: list[str], doc_name: str = "document") -> str:
    """
    Generate a summary of the entire document.
    Used by the 'Summarize' button in Stage 4.

    Args:
        chunks:   All text chunks from the document
        doc_name: Name of the document (for display)

    Returns:
        Summary string from Ollama
    """
    # Use first 10 chunks + evenly spaced samples for large docs
    if len(chunks) <= 10:
        sample_chunks = chunks
    else:
        # Take first 5, last 3, and 2 from the middle
        indices = list(range(5)) + [len(chunks)//2, len(chunks)//2+1] + list(range(len(chunks)-3, len(chunks)))
        sample_chunks = [chunks[i] for i in sorted(set(indices))]

    content = "\n\n".join(sample_chunks)

    prompt = f"""Please provide a clear, structured summary of the following document.

Include:
1. Main topic / purpose
2. Key points (3-5 bullet points)
3. Any important conclusions or findings

DOCUMENT CONTENT:
{content}

SUMMARY:"""

    try:
        return _ollama_generate(prompt)
    except Exception as e:
        return _format_llm_error(e).replace("generating answer", "generating summary")


def test_api_connection() -> bool:
    """Test if Ollama server/model works. Used in app.py startup check."""
    try:
        response = _ollama_generate("Say exactly: API connection successful")
        return "successful" in response.lower()
    except Exception as e:
        print(f"[llm_handler] Ollama test failed: {e}")
        return False


# ── Quick test — run: python llm_handler.py ─────────────────────────────────
if __name__ == "__main__":
    print("=== Testing Ollama Connection ===")
    ok = test_api_connection()
    print(f"Connection: {'OK' if ok else 'FAILED'}")

    if ok:
        print("\n=== Testing Answer Generation ===")
        test_chunks = [
            {
                "text": "Machine learning is a branch of AI that allows systems to learn from data.",
                "score": 92.0,
                "source": "test.pdf",
                "chunk_id": 0,
            },
            {
                "text": "Supervised learning uses labeled data to train models.",
                "score": 85.0,
                "source": "test.pdf",
                "chunk_id": 1,
            },
        ]

        answer = get_answer("What is machine learning?", test_chunks)
        print(f"Question: What is machine learning?")
        print(f"Answer: {answer}")