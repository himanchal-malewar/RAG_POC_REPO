# 📘 Agentic RAG Evaluation System

This project implements an **Agentic Retrieval-Augmented Generation
(RAG)** system with:

-   🔎 Document ingestion (PDF, TXT, DOCX, CSV)
-   ✂️ Text chunking with overlap
-   🧠 Embedding generation (OpenAI via GitHub Models endpoint)
-   📌 Cosine similarity retrieval
-   🤖 Autonomous agent loop (Planner → Generator → Reflector)
-   📊 Full evaluation pipeline

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

## 3️⃣ Configure Environment Variables

Create a `.env` file:

    GITHUB_TOKEN=YOUR_API_TOKEN

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

------------------------------------------------------------------------

# 📊 Evaluation Metrics

-   Retrieval Recall
-   Exact Match
-   Average Semantic Score
-   Hallucination Rate
-   Tool Misuse Rate
-   Average Latency
-   Average Agent Loops

------------------------------------------------------------------------

# 🏗️ Architecture Summary

**Core Components:**

1.  Document Loader\
2.  Chunking Module\
3.  Embedding Generator\
4.  Vector Retrieval\
5.  Planner\
6.  Generator\
7.  Reflector\
8.  Evaluation Engine

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
