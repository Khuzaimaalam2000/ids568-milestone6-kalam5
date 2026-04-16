# Milestone 6: RAG Pipeline & Agentic System
**Course:** MLOps - Module 7  
**NetID:** Kalam5

---

## Model Information

| Field | Value |
|-------|-------|
| Model | mistral:7b-instruct |
| Size class | 7B |
| Serving stack | Ollama (local) |
| Hardware | [YOUR MACHINE e.g. MacBook M2 16GB] |
| Avg generation latency | ~8s on CPU |

---

## Setup Instructions

### 1. Install Ollama
Go to https://ollama.com and install for your OS.

### 2. Pull the model
```bash
ollama pull mistral:7b-instruct
ollama serve  # Keep this running in a separate terminal
```

### 3. Clone and install dependencies
```bash
git clone [your repo URL]
cd ids568-milestone6-kalam5
pip install -r requirements.txt
```

### 4. Verify the model works
```bash
ollama run mistral:7b-instruct "Say hello"
```

---

## Usage

### Part 1: RAG Pipeline
```bash
# Open and run all cells in order
jupyter notebook rag_pipeline.ipynb
```

### Part 2: Agent Controller
```bash
# Make sure rag_pipeline.ipynb has been run first (creates chroma_db/)
python agent_controller.py
```

---

## Architecture Overview

documents/ → chunker → embedder → ChromaDB
↓
user query → retriever → LLM generator → answer
user task → agent controller (LLM) → tool selection
→ retriever tool OR summarizer tool
→ final answer generator
→ trace saved to agent_traces/

---

## Known Limitations

- Generation latency is ~8s on CPU. GPU would reduce this to ~1-2s.
- The corpus is small (5 documents). A larger corpus would stress-test retrieval more.
- Tool selection is LLM-driven and can be inconsistent with ambiguous task phrasing.
- Out-of-scope queries correctly trigger "I don't know" responses but could be improved with better fallback logic.
- The summarizer tool works best on passages under 1000 characters due to context window constraints.

---

## File Structure

ids568-milestone6-kalam5/
├── rag_pipeline.ipynb          # Part 1: RAG implementation
├── agent_controller.py         # Part 2: Agent implementation
├── rag_evaluation_report.md    # Part 1: Evaluation metrics
├── rag_pipeline_diagram.md     # Architecture diagram
├── agent_report.md             # Part 2: Analysis
├── agent_traces/               # 10 task traces (JSON)
├── documents/                  # Knowledge base documents
├── chroma_db/                  # Persisted vector index
├── requirements.txt            # Pinned dependencies
└── README.md                   # This file

