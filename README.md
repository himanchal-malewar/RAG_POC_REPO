# 📘 Agentic RAG Evaluation System

This project implements an **Agentic Retrieval-Augmented Generation
(RAG)** system with:

-   🔎 Document ingestion (PDF, TXT, DOCX, CSV)
-   ✂️ Text chunking with overlap
-   🧠 Embedding generation (OpenAI via GitHub Models endpoint)
-   📌 Cosine similarity retrieval
-   🤖 Autonomous agent loop (Planner → Generator → Reflector)
-   📊 Full evaluation pipeline:
    -   Retrieval Recall
    -   Exact Match
    -   Semantic Similarity
    -   Hallucination Rate
    -   Tool Misuse Rate
    -   Latency
    -   Agent Loop Count

The system uses Azure AI Inference SDK with GitHub-hosted OpenAI models.

------------------------------------------------------------------------

# 📂 Project Structure

    RAG_CODE/
    │
    ├── .env
    ├── .gitignore
    ├── doc_ai.txt
    ├── golden_set.json
    ├── ITI_AgenticAI_Final.pdf
    ├── RAG_Test.py
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

# ⚙️ Setup Instructions

## 1️⃣ Create Virtual Environment

``` bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Mac/Linux
```

## 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### requirements.txt

    numpy==1.26.4
    pandas==2.2.2
    PyMuPDF==1.24.1
    python-docx==1.1.0
    python-dotenv==1.0.1
    azure-ai-inference==1.0.0b3
    azure-core==1.30.1
    faiss-cpu==1.7.4

## 3️⃣ Configure Environment Variables

Create a `.env` file:

    GITHUB_TOKEN=YOUR_API_TOKEN

Replace `YOUR_API_TOKEN` with your actual OpenAI API
token.

------------------------------------------------------------------------

# 🚀 How to Run

``` bash
python RAG_Test.py
```

The system will:

1.  Load document (`ITI_AgenticAI_Final.pdf`)
2.  Chunk text
3.  Generate embeddings
4.  Run evaluation using `golden_set.json`
5.  Enter interactive Q&A mode

To exit interactive mode:

    exit

------------------------------------------------------------------------

# 🧠 System Architecture

## Document Loader

Supports: - `.pdf` - `.txt` - `.docx` - `.csv`

## Chunking

-   Chunk size: 800
-   Overlap: 200

## Models Used

  Task         Model
  ------------ -------------------------------
  Embeddings   openai/text-embedding-3-small
  Chat         openai/gpt-4o-mini

Endpoint:

    https://models.github.ai/inference

------------------------------------------------------------------------

# 📊 Evaluation Metrics

Each test case in `golden_set.json`:

    {
      "question": "...",
      "expected_answer": "..."
    }

Metrics reported:

-   Retrieval Recall
-   Exact Match
-   Average Semantic Score
-   Hallucination Rate
-   Tool Misuse Rate
-   Average Latency
-   Average Agent Loops

------------------------------------------------------------------------

# 🎯 Key Features

✅ Autonomous multi-step reasoning\
✅ Retrieval + generation evaluation\
✅ Grounding verification\
✅ Semantic similarity scoring\
✅ Tool misuse detection\
✅ Full performance report

------------------------------------------------------------------------

# 🏗️ Future Improvements

-   Replace brute-force similarity with FAISS index
-   Add citation highlighting
-   Add structured tool calling
-   Add streaming responses
-   Add vector caching
-   Add re-ranking layer

------------------------------------------------------------------------

# 📌 Summary

This project demonstrates a complete **Agentic RAG pipeline with
evaluation**, combining:

-   Retrieval
-   Planning
-   Controlled generation
-   Reflection
-   Metric-driven validation

Suitable for academic research, RAG benchmarking, and agent architecture
experimentation.
