# RAG Pipeline Architecture

## Data Flow Diagram

DOCUMENT CORPUS
(ai_basics.txt, rag_systems.txt, mlops.txt, llm_agents.txt)
        │
        ▼
CHUNKER
- Fixed-size chunking
- chunk_size = 200
- overlap = 50
- preserves semantic continuity
        │
        ▼
EMBEDDER
- SentenceTransformer: all-MiniLM-L6-v2
- Output: 384-dimensional dense embeddings
- Lightweight + semantic similarity optimized
        │
        ▼
VECTOR STORE (FAISS IndexFlatL2)
- Stores embeddings + raw text
- Fast similarity search (L2 distance)
        │
        ▼
RETRIEVER
- cosine/L2 similarity search
- top-k = 3 chunks retrieved
        │
        ▼
LLM GENERATOR
- Model: mistral:7b-instruct (Ollama)
- Prompt: context-grounded generation
- Output: final answer

## Agent Architecture

USER TASK
    │
    ▼
AGENT CONTROLLER (LLM-based decision system)

Tool Selection Policy:
- Factual QA → Retriever
- Summarization → Summarizer Tool
- Multi-step reasoning → Retriever + LLM
- Simple chat → Direct LLM

    │
    ├──────────────┬──────────────────┬───────────────┐
    ▼              ▼                  ▼
RETRIEVER     SUMMARIZER        DIRECT LLM
(FAISS)       (context reduce)  (no tools)

    │              │                  │
    └──────────────┴──────────────────┘
                   ▼
           ANSWER GENERATOR
           (Mistral 7B Instruct)

                   ▼
            TRACE LOGGER
      (agent_traces/*.json)