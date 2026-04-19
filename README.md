# IDS 568 Milestone 6 – RAG + Agentic AI System

## Overview

This project implements a full Retrieval-Augmented Generation (RAG) system combined with an LLM-based agent controller. The system demonstrates document ingestion, embedding-based retrieval, vector search, and tool-using agents for intelligent reasoning.

---

## Model Information

| Field | Value |
|-------|-------|
| Model | mistral:7b-instruct |
| Size class | 7B |
| Serving stack | Ollama (local) |
| Vector DB | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Hardware | MacBook (CPU only) |
| Avg generation latency | ~2-5 seconds per query |

---

## Setup Instructions

### Step 1: Install Ollama
Go to https://ollama.com and install for your OS, then run:

```bash
ollama pull mistral:7b-instruct
ollama serve
```

### Step 2: Install dependencies

```bash
pip install langchain langchain-community langchain-ollama chromadb sentence-transformers faiss-cpu
```

### Step 3: Build RAG index

```bash
python rag_pipeline.py
```

### Step 4: Run Agent system

```bash
python agent_controller.py
```

---

## System Architecture

### Part 1: RAG Pipeline

documents/ → chunker → embedder → ChromaDB
↓
user query → retriever → Mistral 7B → grounded answer

- Documents loaded from local corpus
- Text split using recursive chunking (chunk size: 200, overlap: 50)
- Embeddings generated using all-MiniLM-L6-v2
- Stored in ChromaDB vector database
- Retrieval performed using MMR similarity search (k=3)
- Context passed into Mistral 7B for grounded generation

### Part 2: Agent System

user task → agent controller → tool selection
→ retriever tool      (factual questions)
→ summarizer tool     (summarization tasks)
→ direct mode         (simple/math tasks)
→ answer generator
→ trace saved to agent_traces/

Tool selection policy:
- Contains "summarize" or "summary" → summarizer
- Contains "2+2", "math", "calculate" → direct
- Everything else → retriever

---

## Dataset

The system uses a small knowledge base containing documents on:
- AI fundamentals
- RAG systems
- Vector databases
- MLOps concepts
- LLM agents

---

## Usage Examples

### RAG Pipeline

```python
result = rag_chain.invoke({"query": "What is retrieval-augmented generation?"})
print(result["result"])
```

### Agent Controller

```python
agent.run("What is data drift?", task_id=1)
```

---

## Evaluation

The system was tested on 10 diverse queries covering:
- Factual QA (RAG concepts, vector DBs, MLOps)
- Summarization tasks
- Direct computation tasks
- Out-of-scope queries (edge cases)

**Results:**
- Average Precision@3: 1.0
- Average retrieval latency: less than 100ms
- Average generation latency: 2-5 seconds
- Hallucination rate: Minimal

---

## Output Traces

All execution traces are stored in agent_traces/

Each trace includes:
- Task ID and task description
- Tool selection decision
- Retrieval latency
- Generation latency
- Total latency
- Final LLM response

---

## File Structure


ids568-milestone6-kalam5/
├── rag_pipeline.py              # Part 1: RAG implementation
├── agent_controller.py          # Part 2: Agent implementation
├── rag_evaluation_report.md     # Part 1: Evaluation metrics
├── rag_pipeline_diagram.md      # Architecture diagram
├── agent_report.md              # Part 2: Analysis
├── agent_traces/                # 10 task traces (JSON)
├── documents/                   # Knowledge base documents
├── chroma_db/                   # Persisted vector index
├── requirements.txt             # Pinned dependencies
└── README.md                    # This file

---

## Known Limitations

- Generation latency is 2-5 seconds on CPU. GPU would reduce this significantly.
- The corpus is small (5 documents). A larger corpus would stress-test retrieval more.
- Tool selection uses rule-based logic which may not generalize to all task types.
- Summarizer tool uses general knowledge rather than retrieved context.
- Out-of-scope queries correctly trigger "Not found in knowledge base" responses.

---

## Key Observations

- RAG improves factual grounding significantly
- Tool-based agents outperform single-prompt LLMs for structured tasks
- Retrieval quality depends heavily on chunking strategy
- Latency is dominated by LLM inference
- Direct mode correctly handles simple tasks without unnecessary retrieval

