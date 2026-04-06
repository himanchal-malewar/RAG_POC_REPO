# 🧠 Agentic RAG System with Evaluation Metrics

## 📌 Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** that:

* Ingests documents (PDF, TXT, DOCX, CSV)
* Splits them into chunks
* Generates embeddings
* Retrieves relevant context for queries
* Uses an **agent loop (Planner → Generator → Reflector)** for answer refinement
* Evaluates performance using **true RAG metrics**

---

## ⚙️ Features

* ✅ Multi-format document support
* ✅ Embedding caching for performance
* ✅ Autonomous reasoning loop (Agentic AI)
* ✅ Retry-safe API calls
* ✅ Proper evaluation metrics:

  * Recall@K (fixed)
  * Precision@K
  * Semantic similarity
  * Hallucination detection
  * Latency tracking
  * Agent loop iterations

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / Mac
.venv\Scripts\activate        # Windows
```

---

### 3. Install Requirements

Create a `requirements.txt` file with:

```txt
numpy
pymupdf
python-docx
python-dotenv
tenacity
azure-ai-inference
azure-core
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in root:

```env
GITHUB_TOKEN=your_token_here
```

---

## 📂 Project Structure

```
.
├── RAG_Test.py
├── embeddings.pkl
├── ITI_AgenticAI_Final.pdf
├── golden_set.json
├── requirements.txt
└── .env
```

---

## 📊 Input Data Format

### 1. Document Input

Supported formats:

* `.pdf`
* `.txt`
* `.docx`
* `.csv`

---

### 2. Golden Dataset (for Evaluation)

`golden_set.json`

```json
[
  {
    "question": "What is X?",
    "expected_answer": "Definition of X",
    "relevant_chunks": ["Actual chunk text from document"]
  }
]
```

⚠️ `relevant_chunks` is **mandatory for Recall@K**

---

## 🚀 How to Run

### Option 1: Default Files

```bash
python rag.py
```

Uses:

* `ITI_AgenticAI_Final.pdf`
* `golden_set.json`

---

### Option 2: Custom Files (Recommended)

```bash
python rag.py --file your_doc.pdf --golden your_golden.json
```

---

## 💬 Interactive Mode

After evaluation, the system enters **Q&A mode**:

```
Ask your question: (Type exit to quit)
```

Example:

```
Ask your question: What is Agentic AI?
```

---

## 🧠 System Architecture

### Pipeline Flow

```
User Query
   ↓
Chunk Retrieval (Top-K)
   ↓
Planner → creates reasoning steps
   ↓
Generator → produces answer using context
   ↓
Reflector → validates grounding
   ↓
Final Answer
```

---

## 🔍 Detailed Flow Explanation

### 1. Document Loading

* Reads file based on extension
* Converts into raw text

---

### 2. Chunking

* Splits text into ~800 character chunks
* Preserves sentence boundaries

---

### 3. Embedding Creation

* Uses embedding model
* Stores vectors in `embeddings.pkl` (cache)

---

### 4. Retrieval

* Converts query into embedding
* Computes cosine similarity
* Returns **Top-K relevant chunks**

---

### 5. Agent Loop

#### 🧩 Planner

* Breaks query into reasoning steps

#### ✍️ Generator

* Generates answer using:

  * Plan
  * Retrieved context

#### 🔍 Reflector

* Validates if answer is grounded
* If "NO" → regenerate (max 3 loops)

---

### 6. Evaluation System

Runs on `golden_set.json`

#### Metrics:

| Metric             | Description                                              |
| ------------------ | -------------------------------------------------------- |
| Recall@K           | Measures retrieval completeness                          |
| Precision@K        | Measures relevance of retrieved chunks                   |
| Semantic Score     | Embedding similarity between predicted & expected answer |
| Hallucination Rate | Checks if answer is grounded in context                  |
| Avg Latency        | Time per query                                           |
| Avg Loops          | Number of agent iterations                               |

---

## 📈 Sample Output

```
===== REPORT =====
Recall@K: 0.78
Precision@K: 0.66
Semantic Score: 0.81
Hallucination Rate: 0.12
Avg Latency: 1.45
Avg Loops: 1.7
==================
```

---

## ⚠️ Important Notes

* Ensure `GITHUB_TOKEN` is valid
* First run may take time (embedding creation)
* `embeddings.pkl` avoids recomputation
* Recall@K works only if `relevant_chunks` is present

---

## 🔧 Troubleshooting

### Token Error

```
Missing GITHUB_TOKEN
```

✔ Fix: Check `.env` file

---

### Empty Embeddings Error

```
No valid text to embed
```

✔ Fix: Ensure document has readable text

---

### Low Recall

✔ Check:

* chunk size
* golden dataset quality

<!--

## 🚀 Future Improvements

* Semantic Recall@K (embedding-based)
* MRR (Mean Reciprocal Rank)
* NDCG scoring
* Hybrid retrieval (BM25 + embeddings)
* Cross-encoder re-ranking

---

## 👨‍💻 Author Notes

This project demonstrates:

* Real-world RAG system design
* Agentic AI workflow
* Proper evaluation methodology

-->
