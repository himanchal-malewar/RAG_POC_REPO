import os
import re
import json
import csv
import time
import pickle
import numpy as np
import fitz
from docx import Document
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from azure.ai.inference import EmbeddingsClient, ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# =========================
# CONFIG
# =========================
load_dotenv()

ENDPOINT = "https://models.github.ai/inference"
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise ValueError("Missing GITHUB_TOKEN")

embedding_client = EmbeddingsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

chat_client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# =========================
# RETRY WRAPPERS
# =========================
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def safe_embed(texts):
    return embedding_client.embed(input=texts, model=EMBED_MODEL)

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def safe_chat(messages):
    return chat_client.complete(model=CHAT_MODEL, messages=messages)

# =========================
# DOCUMENT LOADER
# =========================
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".csv":
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)

    else:
        raise ValueError("Unsupported format")

# =========================
# CHUNKING
# =========================
def chunk_text(text, chunk_size=800):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks, current = [], ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks

# =========================
# EMBEDDING (FIXED)
# =========================
def embed_texts(texts, batch_size=20):

    cleaned = [
        str(t).strip()
        for t in texts
        if t and str(t).strip()
    ]

    if not cleaned:
        raise ValueError("No valid text to embed")

    vectors = []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i + batch_size]
        res = safe_embed(batch)
        vectors.extend([np.array(r.embedding) for r in res.data])

    return vectors

def load_or_create_embeddings(chunks, cache_file="embeddings.pkl"):

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    vectors = embed_texts(chunks)

    with open(cache_file, "wb") as f:
        pickle.dump(vectors, f)

    return vectors

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# RETRIEVAL
# =========================
def retrieve(question, chunks, vectors, top_k=3):

    q_vec = embed_texts([question])[0]

    scores = [(i, cosine(q_vec, v)) for i, v in enumerate(vectors)]
    scores.sort(key=lambda x: x[1], reverse=True)

    top = scores[:top_k]
    context = "\n\n".join(chunks[i] for i, _ in top)

    return context, top

# =========================
# METRICS
# =========================
def precision_at_k(question, chunks, top_chunks):
    keywords = question.lower().split()

    relevant = sum(
        any(k in chunks[i].lower() for k in keywords)
        for i, _ in top_chunks
    )

    return relevant / len(top_chunks)

def semantic_recall(expected, context):
    emb = embed_texts([expected, context])
    return cosine(emb[0], emb[1]) > 0.7

def semantic_score(pred, exp):
    emb = embed_texts([pred, exp])
    return cosine(emb[0], emb[1])

def groundedness_score(ans, context):
    emb = embed_texts([ans, context])
    return cosine(emb[0], emb[1])

# =========================
# AGENT MODULES
# =========================
def planner(q):
    res = safe_chat([
        {"role": "system", "content": "Create short reasoning steps."},
        {"role": "user", "content": q}
    ])
    return res.choices[0].message.content

def generator(q, context, plan):
    res = safe_chat([
        {"role": "system", "content": "Use ONLY context."},
        {"role": "user",
         "content": f"Plan:\n{plan}\n\nContext:\n{context}\n\nQ:\n{q}"}
    ])
    return res.choices[0].message.content

def reflector(ans, context):
    res = safe_chat([
        {"role": "system", "content": "Is answer supported? YES/NO"},
        {"role": "user",
         "content": f"Context:\n{context}\n\nAnswer:\n{ans}"}
    ])
    return res.choices[0].message.content.strip()

# =========================
# AGENT LOOP
# =========================
def autonomous_agent(q, chunks, vectors, context=None, top_chunks=None):

    if context is None:
        context, top_chunks = retrieve(q, chunks, vectors)

    plan = planner(q)

    for i in range(3):
        ans = generator(q, context, plan)

        if reflector(ans, context) == "YES":
            return ans, i + 1, context, top_chunks

    return ans, 3, context, top_chunks

# =========================
# EVALUATION
# =========================
def evaluate_system(test_file, chunks, vectors):

    with open(test_file) as f:
        test_set = json.load(f)

    total = len(test_set)

    recall_hits = 0
    precision_scores = []
    semantic_scores = []
    hallucinations = 0
    latency_total = 0
    loops_total = 0

    for test in test_set:

        q = test["question"]
        expected = test["expected_answer"]

        start = time.time()

        context, top_chunks = retrieve(q, chunks, vectors)

        if semantic_recall(expected, context):
            recall_hits += 1

        precision_scores.append(
            precision_at_k(q, chunks, top_chunks)
        )

        ans, loops, context, _ = autonomous_agent(
            q, chunks, vectors, context, top_chunks
        )

        loops_total += loops

        semantic_scores.append(
            semantic_score(ans, expected)
        )

        if groundedness_score(ans, context) < 0.5:
            hallucinations += 1

        latency_total += time.time() - start

    print("\n===== REPORT =====")
    print("Recall:", recall_hits / total)
    print("Precision@K:", sum(precision_scores) / total)
    print("Semantic Score:", sum(semantic_scores) / total)
    print("Hallucination Rate:", hallucinations / total)
    print("Avg Latency:", latency_total / total)
    print("Avg Loops:", loops_total / total)
    print("==================\n")

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":

    file_path = "ITI_AgenticAI_Final.pdf"
    test_file = "golden_set.json"

    text = load_document(file_path)
    chunks = chunk_text(text)
    vectors = load_or_create_embeddings(chunks)

    evaluate_system(test_file, chunks, vectors)

    while True:
        q = input("Ask your question: (Type exit to quit) ")
        if q.lower() == "exit":
            break

        ans, _, _, _ = autonomous_agent(q, chunks, vectors)
        print("\nAnswer:\n", ans)