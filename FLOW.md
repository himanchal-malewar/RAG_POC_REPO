# Agentic RAG Evaluation System

------------------------------------------------------------------------

# 🏗️ High-Level Architecture Diagram

flowchart LR

subgraph Data_Ingestion
    A1[PDF / TXT / DOCX / CSV]
    A2[Document Loader]
    A3[Raw Text]
    A1 --> A2 --> A3
end

subgraph Preprocessing
    B1[Chunking Module]
    B2[Chunked Text]
    A3 --> B1 --> B2
end

subgraph Embedding_Layer
    C1[Embedding Model<br/>text-embedding-3-small]
    C2[Vector Store (In-Memory)]
    B2 --> C1 --> C2
end

subgraph Retrieval_Layer
    D1[User Question]
    D2[Question Embedding]
    D3[Cosine Similarity Search]
    D4[Top-K Context Chunks]
    D1 --> D2 --> D3 --> D4
    C2 --> D3
end

subgraph Agent_Loop
    E1[Planner]
    E2[Generator<br/>(gpt-4o-mini)]
    E3[Reflector]
    D4 --> E1 --> E2 --> E3
    E3 -->|Grounded YES| F1[Final Answer]
    E3 -->|NO Retry| D3
end

subgraph Evaluation_Engine
    G1[Golden Set]
    G2[Exact Match]
    G3[Semantic Similarity]
    G4[Retrieval Recall]
    G5[Hallucination Rate]
    G6[Tool Misuse Detection]
    F1 --> G2
    F1 --> G3
    D4 --> G4
    E2 --> G5
    D4 --> G6
    G1 --> G2
end

------------------------------------------------------------------------
