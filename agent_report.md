# Agent Controller Report
## Milestone 6 - Part 2

---

## Tool Selection Policy

The agent uses Mistral 7B to analyze each incoming task and select from four strategies:

| Trigger | Tool Selected | Reasoning |
|---------|--------------|-----------|
| Factual question about knowledge base topics | `retriever` | Need external knowledge |
| Condensation/summarization of provided text | `summarizer` | Need distillation, no lookup |
| Factual question + need condensed output | `retrieve_then_summarize` | Need both tools in sequence |
| Math, greetings, simple conversational | `direct` | LLM knowledge sufficient |

---

## Retrieval Integration

The retriever tool wraps the ChromaDB vector store from Part 1, making it callable as an agent tool. It returns structured results including retrieved context, source document names, and latency. The agent uses these results as context for the final answer generation step.

---

## Performance Analysis (10 Tasks)

| Task | Tool Used | Success | Notes |
|------|-----------|---------|-------|
| What is RAG and how does it reduce hallucinations? | retriever | ✅ | Correct tool, grounded answer |
| FAISS vs ChromaDB differences | retriever | ✅ | Good source retrieval |
| MLOps monitoring practices | retriever | ✅ | Retrieved correct docs |
| Summarize LLM agent concepts (text provided) | summarizer | ✅ | Correct - text provided directly |
| Retrieve and summarize chunking strategies | retrieve_then_summarize | ✅ | Both tools used correctly |
| Find and summarize vector DB options | retrieve_then_summarize | ✅ | Sequential tool use worked |
| How to combine RAG with agents | retriever | ✅ | Retrieved relevant context |
| Evaluation metrics for RAG | retriever | ✅ | Precision/recall retrieved |
| What is 2 + 2? | direct | ✅ | Correct - no retrieval needed |
| Data drift and canary deployments | retriever | ✅ | Multi-concept retrieval worked |

**Success rate: 10/10 tasks**

---

## Failure Analysis

No hard failures observed in this evaluation set. However, edge cases to watch for:

- **Ambiguous tasks:** If a task could be answered directly OR via retrieval, the LLM tool selector may pick inconsistently across runs due to temperature variation.
- **Long text summarization:** The summarizer works well for short passages but may truncate key information in very long inputs due to context window limits.
- **Tool selection stability:** The selector prompt is sensitive to exact task phrasing. Rephrasing the same question differently can occasionally change the tool selection.

---

## Model Quality and Latency Tradeoffs

- **Mistral 7B** follows tool selection instructions reliably and generates coherent answers grounded in retrieved context.
- Running on CPU adds ~8s generation latency per task. Acceptable for evaluation but not production.
- The model occasionally produces more verbose answers than needed, which could be controlled with a max_tokens limit.
- A 14B model would improve reasoning quality for multi-hop tasks but would double latency on CPU hardware.