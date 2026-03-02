import os
import re
import json
import csv
import time
import numpy as np
import fitz
from docx import Document
from dotenv import load_dotenv
from azure.ai.inference import EmbeddingsClient, ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# =========================
# CONFIGURATION
# =========================
load_dotenv()

ENDPOINT = "https://models.github.ai/inference"
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"
TOKEN = os.getenv("GITHUB_TOKEN")

print(TOKEN)

if not TOKEN:
    raise ValueError("GITHUB_TOKEN not found")

embedding_client = EmbeddingsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

chat_client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# =========================
# DOCUMENT LOADER
# =========================
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "".join([page.get_text() for page in doc])

    elif ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # elif ext == ".csv":
    #     df = pd.read_csv(file_path)
    #     rows = []
    #     for _, row in df.iterrows():
    #         rows.append(", ".join([f"{c}: {row[c]}" for c in df.columns]))
    #     return "\n".join(rows)
    
    elif ext == ".csv":
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)

    else:
        raise ValueError("Unsupported file format")

# =========================
# CHUNKING
# =========================
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

# =========================
# EMBEDDING
# =========================
def embed_texts(texts):
    response = embedding_client.embed(
        input=texts,
        model=EMBED_MODEL
    )
    return [np.array(r.embedding) for r in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# RETRIEVAL
# =========================
def retrieve(question, chunks, vectors, top_k=3):
    q_vec = embed_texts([question])[0]
    scores = [(i, cosine_similarity(q_vec, v)) for i, v in enumerate(vectors)]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    context = "\n\n".join([chunks[i] for i, _ in top])
    return context, top

# =========================
# PRECISION & RECALL
# =========================
def precision_at_k(question, chunks, top_chunks):
    keywords = question.lower().split()
    relevant = sum(
        any(k in chunks[i].lower() for k in keywords)
        for i, _ in top_chunks
    )
    return relevant / len(top_chunks)

def recall_at_k(expected_answer, context):
    return expected_answer.lower() in context.lower()

# =========================
# TOOL MISUSE
# =========================
def analytical_question(q):
    keywords = ["average", "sum", "total", "highest", "lowest",
                "count", "max", "min", "mean"]
    return any(k in q.lower() for k in keywords)

def has_numbers(text):
    return bool(re.search(r"\d+", text))

# =========================
# AGENT MODULES
# =========================
def planner(q):
    return chat_client.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Create short reasoning plan."},
            {"role": "user", "content": q}
        ]
    ).choices[0].message.content

def generator(q, context):
    return chat_client.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content":
             "Only answer using provided context. "
             "If not found, say 'I don't know'."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion:\n{q}"}
        ]
    ).choices[0].message.content

def reflector(ans):
    return chat_client.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",
             "content": "Is this answer grounded in context? YES or NO."},
            {"role": "user", "content": ans}
        ]
    ).choices[0].message.content.strip()

# =========================
# SEMANTIC SIMILARITY
# =========================
def semantic_score(predicted, expected):
    emb1 = embed_texts([predicted])[0]
    emb2 = embed_texts([expected])[0]
    return cosine_similarity(emb1, emb2)

# =========================
# AGENT LOOP
# =========================
def autonomous_agent(q, chunks, vectors, max_loops=3):

    loops = 0
    misuse = False

    for _ in range(max_loops):
        loops += 1
        context, top_chunks = retrieve(q, chunks, vectors)

        if analytical_question(q) and not has_numbers(context):
            misuse = True
            return "Insufficient numeric data.", loops, misuse

        ans = generator(q, context)
        verdict = reflector(ans)

        if verdict == "YES":
            return ans, loops, misuse

    return ans, loops, misuse

# =========================
# FULL EVALUATION RUNNER
# =========================
def evaluate_system(test_file, chunks, vectors):

    test_set = json.load(open(test_file))
    total = len(test_set)

    retrieval_hits = 0
    exact_matches = 0
    semantic_scores = []
    hallucinations = 0
    total_latency = 0
    total_loops = 0
    misuse_count = 0

    for test in test_set:

        q = test["question"]
        expected = test["expected_answer"]

        start = time.time()

        context, top_chunks = retrieve(q, chunks, vectors)
        if recall_at_k(expected, context):
            retrieval_hits += 1

        ans, loops, misuse = autonomous_agent(q, chunks, vectors)

        end = time.time()
        latency = end - start

        total_latency += latency
        total_loops += loops
        if misuse:
            misuse_count += 1

        if ans.strip().lower() == expected.strip().lower():
            exact_matches += 1

        sem = semantic_score(ans, expected)
        semantic_scores.append(sem)

        if expected.lower() not in context.lower() and ans.lower() != "i don't know":
            hallucinations += 1

    print("\n========== EVALUATION REPORT ==========")
    print("Retrieval Recall:", retrieval_hits / total)
    print("Exact Match:", exact_matches / total)
    print("Avg Semantic Score:", sum(semantic_scores) / total)
    print("Hallucination Rate:", hallucinations / total)
    print("Tool Misuse Rate:", misuse_count / total)
    print("Avg Latency:", total_latency / total)
    print("Avg Agent Loops:", total_loops / total)
    print("=======================================\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("📂 Loading document...")
    file_path = "ITI_AgenticAI_Final.pdf" # supports .pdf, .txt, .docx, .csv
    golden_test_file = "golden_set.json"

    text = load_document(file_path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)

    # Run evaluation
    evaluate_system(golden_test_file, chunks, vectors)

    # Interactive mode
    while True:
        q = input("Ask (exit to quit): ")
        if q.lower() == "exit":
            break

        ans, _, _ = autonomous_agent(q, chunks, vectors)
        print("\nAnswer:\n", ans)