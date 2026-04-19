# Agent Report

## 1. Overview

This project implements a multi-tool AI agent built on top of a Retrieval-Augmented Generation (RAG) pipeline. The agent is designed to intelligently select between different tools to complete tasks, including retrieving factual information, summarizing content, or answering directly using a language model.

The system integrates:
- A vector database (ChromaDB) for document retrieval
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- A local LLM (`mistral:7b-instruct` via Ollama)
- An agent controller for dynamic tool selection

---

## 2. System Architecture

The agent consists of three main tools:

### Retriever Tool
- Uses ChromaDB vector store
- Performs similarity search (top-k = 3)
- Returns relevant document chunks as context

### Summarizer Tool
- Uses the LLM to compress text into bullet points
- Applied when tasks require distillation rather than retrieval

### Direct Mode
- Bypasses tools and directly queries the LLM
- Used for simple or non-knowledge-based queries

---

## 3. Tool Selection Policy

The agent uses an LLM-based controller to select tools dynamically based on task type:

- **Retriever** → for factual queries (e.g., “What is RAG?”)
- **Summarizer** → for summarization tasks (e.g., “Summarize RAG systems”)
- **Direct** → for simple queries (e.g., math questions)
- **Retrieve → Summarize** → not used in final optimized version

In the final run:
- Retriever used: 8 tasks  
- Summarizer used: 1 task  
- Direct used: 1 task  

---

## 4. Evaluation Results

The agent was evaluated on 10 diverse tasks covering factual queries, conceptual explanations, summarization, and simple reasoning.

### Performance Observations

- Retrieval worked consistently and returned relevant context for all factual queries
- The agent correctly selected the retriever tool for most knowledge-based tasks
- Direct mode handled simple queries efficiently (e.g., “2+2”)
- Summarizer was used only once for a summarization task

### Latency

- Average response time: ~50–90 seconds per query  
- Direct queries: ~3–5 seconds  
- Retriever-based queries: ~45–90 seconds  
- Summarization: ~20 seconds  

High latency is primarily due to running a local LLM (`mistral:7b`) via Ollama.

---

## 5. Failure Analysis

### 1. RAG Misinterpretation
In the summarization task, the model incorrectly interpreted RAG as “Red-Amber-Green” instead of Retrieval-Augmented Generation.  
This shows hallucination when retrieval context is not properly used.

### 2. Tool Misuse
The agent sometimes selected summarizer without retrieval for knowledge-based queries, reducing factual grounding.

### 3. Over-Reliance on Retriever
Most queries defaulted to retriever even when not strictly necessary.

### 4. High Latency
Execution is slow due to:
- Local LLM inference
- Multiple sequential LLM calls (tool selection + answer generation)

---

## 6. Trace Logging

All agent executions were logged for analysis.

File:
```
agent_traces.json
```

Each trace includes:
- Task ID
- Query
- Selected tool
- Response time
- Final answer

This enables full transparency and debugging of agent decisions.

---

## 7. Improvements

- Use smaller/faster LLM (e.g., 3B model instead of 7B)
- Add caching for repeated queries
- Improve tool selection prompt logic
- Add reranking for retrieved documents
- Implement hybrid search (dense + keyword)
- Limit context length to reduce latency

---

## 8. Conclusion

The system successfully demonstrates a working agentic RAG pipeline with dynamic tool selection and retrieval-augmented reasoning.

It performs well on factual queries and simple reasoning tasks, but still faces limitations in latency and occasional hallucination when retrieval grounding is weak.

Overall, it provides a strong baseline for scalable agent-based RAG systems.