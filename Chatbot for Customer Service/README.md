# 📄 AI Document Assistant using RAG (Retrieval-Augmented Generation)

An AI-powered Document Question Answering system that enables users to upload PDF or DOCX documents and ask questions in natural language. The application extracts document content, creates semantic embeddings, stores them in a FAISS vector database, retrieves the most relevant information, and generates accurate answers using a Large Language Model (LLM).

---

## 🚀 Features

- 📂 Upload PDF and DOCX documents
- 📖 Automatic text extraction
- ✂️ Intelligent text chunking with overlap
- 🧠 Semantic embeddings using Sentence Transformers
- 🗄️ FAISS vector database for fast similarity search
- 🤖 AI-powered question answering using LLM
- ⚡ Fast document retrieval with Retrieval-Augmented Generation (RAG)
- 🎨 Interactive Streamlit web interface

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit

**Backend**
- Python

**AI & NLP**
- Sentence Transformers
- LangChain
- Retrieval-Augmented Generation (RAG)

**Vector Database**
- FAISS

**LLM**
- Ollama (Llama 3.2)
- Easily configurable for Groq or Google Gemini

---

## 📂 Project Structure

```
AI-Document-Assistant/
│
├── app.py
├── document_loader.py
├── chunker.py
├── embedder.py
├── vector_store.py
├── llm_handler.py
├── rag_pipeline.py
├── test_retrieval.py
│
├── vector_store.faiss
├── chunks.pkl
├── metadata.pkl
│
├── requirements.txt
├── Dockerfile
├── README.md
└── SETUP_GUIDE.txt
```

---

# ⚙️ Workflow

```
             Upload PDF / DOCX
                     │
                     ▼
            Document Loader
                     │
                     ▼
            Text Extraction
                     │
                     ▼
        Recursive Text Chunking
                     │
                     ▼
      Sentence Transformer Embeddings
                     │
                     ▼
           FAISS Vector Database
                     │
                     ▼
          Semantic Similarity Search
                     │
                     ▼
      Retrieved Context + User Question
                     │
                     ▼
          Large Language Model (LLM)
                     │
                     ▼
               AI Generated Answer
```

---

# 📋 How It Works

### Step 1
Upload any PDF or DOCX document.

### Step 2
The system extracts all textual content from the document.

### Step 3
The extracted text is divided into smaller overlapping chunks for better retrieval.

### Step 4
Each chunk is converted into vector embeddings using Sentence Transformers.

### Step 5
The embeddings are stored inside a FAISS vector database.

### Step 6
When the user asks a question:

- The query is converted into an embedding.
- FAISS retrieves the most relevant chunks.
- Retrieved context and user query are sent to the LLM.
- The LLM generates an accurate answer grounded in the document.

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI-Document-Assistant.git

cd AI-Document-Assistant
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# 📚 Libraries Used

- Streamlit
- PyPDF2
- python-docx
- Sentence Transformers
- LangChain
- FAISS
- NumPy

---

# 💡 Example Use Cases

- Academic Research Assistant
- Company Policy Assistant
- Legal Document Question Answering
- Medical Report Analysis
- Resume & CV Query Assistant
- Technical Documentation Assistant
- Customer Support Knowledge Base

---

# 📈 Future Improvements

- OCR support for scanned PDFs
- Multi-document retrieval
- Chat history
- Source citation for generated answers
- Voice-based question answering
- Cloud deployment
- Multi-language support

---

# 👨‍💻 Author

**Mohit**

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ If you found this project useful, don't forget to star the repository.
