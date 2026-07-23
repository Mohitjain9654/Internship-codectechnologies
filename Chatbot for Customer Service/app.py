# st.title("Document Q&A Chatbot")

# # Sidebar: upload
# uploaded_file = st.sidebar.file_uploader("Upload PDF or DOCX")

# if uploaded_file:
#     text = load_document(uploaded_file)
#     chunks = split_into_chunks(text)
#     embeddings = embed_chunks(chunks)
#     build_vector_store(embeddings, chunks)
#     st.success("Document processed!")

# # Main area: chat
# question = st.text_input("Ask a question about your document")

# if question:
#     answer = ask(question)
#     st.write(answer)

"""
app.py
======
STAGE 3 + 4: Complete Streamlit UI

Features included:
✓ Document upload (PDF/DOCX)
✓ Ask questions
✓ Source citations (which chunk answered)
✓ Chat memory (remembers last 4 exchanges)
✓ Multi-document support
✓ Document summarization
✓ Confidence scores
✓ Clear/reset functionality

Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import process_document, ask, summarize, clear_pipeline, get_pipeline_status

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d2f3e;
    }

    /* Chat message bubbles */
    .user-message {
        background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
        border: 1px solid #2d4a6e;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8f0fe;
        font-size: 15px;
    }
    .assistant-message {
        background: linear-gradient(135deg, #1e2d1e, #1a2a1a);
        border: 1px solid #2d4a2d;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8f5e8;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Source citation cards */
    .source-card {
        background: #1e2030;
        border: 1px solid #363852;
        border-left: 3px solid #5865f2;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #9ca3af;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: #2d3748;
        color: #68d391;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 8px;
    }

    /* Processing status */
    .status-ready {
        color: #68d391;
        font-weight: bold;
    }
    .status-empty {
        color: #fc8181;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
# Session state persists across Streamlit reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of {"question", "answer", "sources"}

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []       # track multi-document uploads


# ── Helper Functions ──────────────────────────────────────────────────────────
def process_uploaded_file(uploaded_file, append: bool = False):
    """Process an uploaded file and update session state."""
    # Save uploaded file to a temp location (Streamlit gives us a buffer, not a path)
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner(f"Processing '{uploaded_file.name}'... This may take ~30 seconds."):
        result = process_document(tmp_path, source_name=uploaded_file.name)

    os.unlink(tmp_path)  # clean up temp file

    if result["success"]:
        st.session_state.document_loaded = True
        st.session_state.doc_info = result["doc_info"]
        if not append:
            st.session_state.chat_history = []
        if uploaded_file.name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(uploaded_file.name)
        st.success(f"✓ {result['message']}")
        st.info(f"📊 {result['num_chunks']} chunks created | Avg chunk: {result['chunk_stats'].get('avg_length', 0)} chars")
    else:
        st.error(f"✗ {result['message']}")

    return result["success"]


def render_sources(sources: list):
    """Render source citation cards below an answer."""
    if not sources:
        return

    with st.expander("📎 Sources used to generate this answer", expanded=False):
        for i, src in enumerate(sources):
            score_color = "#68d391" if src["score"] >= 70 else "#f6c90e" if src["score"] >= 40 else "#fc8181"
            st.markdown(f"""
<div class="source-card">
    <strong>Chunk #{src['chunk_id']}</strong> from <em>{src['source']}</em>
    <span class="score-badge" style="color: {score_color}">
        {src['score']}% match
    </span>
    <br><br>
    <span style="color: #d1d5db; font-size: 13px;">{src['text'][:300]}{'...' if len(src['text']) > 300 else ''}</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.markdown("*RAG-powered document assistant*")
    st.divider()

    # ── Document Status ───────────────────────────────────────────────────────
    st.markdown("### 📄 Document Status")
    if st.session_state.document_loaded:
        st.markdown('<p class="status-ready">● Document ready</p>', unsafe_allow_html=True)
        if st.session_state.loaded_files:
            for fname in st.session_state.loaded_files:
                st.markdown(f"  - `{fname}`")
    else:
        st.markdown('<p class="status-empty">● No document loaded</p>', unsafe_allow_html=True)

    st.divider()

    # ── Upload Document ───────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Upload a document to start asking questions about it",
    )

    if uploaded_file:
        # Only process if this is a new file (avoid reprocessing on every rerun)
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state.get("processed_keys", set()):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process", type="primary", use_container_width=True):
                    success = process_uploaded_file(uploaded_file, append=False)
                    if success:
                        if "processed_keys" not in st.session_state:
                            st.session_state.processed_keys = set()
                        st.session_state.processed_keys.add(file_key)
                        st.rerun()
            with col2:
                if st.button("Add to existing", use_container_width=True,
                             help="Add this doc to the current knowledge base"):
                    success = process_uploaded_file(uploaded_file, append=True)
                    if success:
                        if "processed_keys" not in st.session_state:
                            st.session_state.processed_keys = set()
                        st.session_state.processed_keys.add(file_key)
                        st.rerun()

    st.divider()

    # ── Advanced Features ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Options")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=6, value=3,
        help="More chunks = more context but slower. 3 is usually best."
    )

    show_scores = st.toggle("Show confidence scores", value=True)
    show_sources = st.toggle("Show source citations", value=True)
    use_memory = st.toggle("Chat memory (last 4 Q&As)", value=True)

    st.divider()

    # ── Summarize Button ──────────────────────────────────────────────────────
    if st.session_state.document_loaded:
        if st.button("📋 Summarize Document", use_container_width=True):
            with st.spinner("Generating summary..."):
                doc_name = st.session_state.loaded_files[0] if st.session_state.loaded_files else "document"
                result = summarize(doc_name)
            if result["success"]:
                st.session_state.chat_history.append({
                    "question": "📋 Summarize this document",
                    "answer": result["summary"],
                    "sources": [],
                    "is_summary": True,
                })
                st.rerun()
            else:
                st.error(result["message"])

    # ── Clear Button ──────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Everything", use_container_width=True):
        clear_pipeline()
        st.session_state.chat_history = []
        st.session_state.document_loaded = False
        st.session_state.doc_info = None
        st.session_state.loaded_files = []
        st.session_state.processed_keys = set()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind</h1>
    <p style="color: #9ca3af; font-size: 16px;">
        Upload any PDF or DOCX — then ask questions about it
    </p>
</div>
""", unsafe_allow_html=True)

# ── Welcome Screen ────────────────────────────────────────────────────────────
if not st.session_state.document_loaded and not st.session_state.chat_history:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📤 1. Upload
        Drop any PDF or DOCX in the sidebar. The system will extract text and build a searchable knowledge base.
        """)
    with col2:
        st.markdown("""
        ### 🔍 2. Ask
        Type any question about your document. The AI retrieves the most relevant sections and answers precisely.
        """)
    with col3:
        st.markdown("""
        ### 📎 3. Cite
        Every answer shows which part of the document it came from, with confidence scores.
        """)
    st.markdown("---")
    st.info("👈 Start by uploading a document in the sidebar")

# ── Chat History Display ──────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for exchange in st.session_state.chat_history:
        # User message
        st.markdown(
            f'<div class="user-message">💬 {exchange["question"]}</div>',
            unsafe_allow_html=True
        )

        # Assistant answer
        if show_scores and exchange.get("sources") and not exchange.get("is_summary"):
            avg_score = round(sum(s["score"] for s in exchange["sources"]) / len(exchange["sources"]), 1)
            score_indicator = "🟢" if avg_score >= 70 else "🟡" if avg_score >= 40 else "🔴"
            score_text = f'<span style="float:right; font-size:12px; color:#9ca3af">{score_indicator} {avg_score}% avg confidence</span>'
        else:
            score_text = ""

        st.markdown(
            f'<div class="assistant-message">{score_text}🤖 {exchange["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Sources
        if show_sources and exchange.get("sources"):
            render_sources(exchange["sources"])

        st.markdown("")

# ── Question Input ────────────────────────────────────────────────────────────
st.markdown("---")

if st.session_state.document_loaded:
    # Example questions
    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        example_cols = st.columns(3)
        examples = [
            "What is the main topic?",
            "Summarize the key points",
            "What are the conclusions?",
        ]
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state["prefill_question"] = example
                    st.rerun()

    # Text input
    prefill = st.session_state.pop("prefill_question", "")
    question = st.chat_input(
        "Ask anything about your document...",
    )

    # Handle prefill from example buttons
    if prefill and not question:
        question = prefill

    if question:
        # Add user message to display immediately
        st.session_state.chat_history.append({
            "question": question,
            "answer": "⏳ Thinking...",
            "sources": [],
        })
        st.rerun()

else:
    st.chat_input("Upload a document first...", disabled=True)


# ── Process pending "Thinking..." messages ────────────────────────────────────
# If last message is still "Thinking...", process it now
if (st.session_state.chat_history and
        st.session_state.chat_history[-1]["answer"] == "⏳ Thinking..."):

    last = st.session_state.chat_history[-1]
    question = last["question"]

    history_for_memory = (
        st.session_state.chat_history[:-1]  # exclude the current pending one
        if use_memory else []
    )

    with st.spinner("Searching document and generating answer..."):
        result = ask(question, top_k=top_k, chat_history=history_for_memory)

    if result["success"]:
        st.session_state.chat_history[-1]["answer"] = result["answer"]
        st.session_state.chat_history[-1]["sources"] = result["sources"]
    else:
        st.session_state.chat_history[-1]["answer"] = f"❌ {result['message']}"

    st.rerun()
