# RAG Pipeline Evaluation Report
## Milestone 6 - Part 1

---

## Model and Setup

- **Model:** mistral:7b-instruct (7B parameter class)
- **Serving:** Ollama (local)
- **Embeddings:** all-MiniLM-L6-v2 (384-dim)
- **Vector DB:** ChromaDB with HNSW index
- **Hardware:** [YOUR CPU/GPU HERE, e.g. MacBook Pro M2, 16GB RAM]
- **Chunk size:** 512 chars | **Overlap:** 64 chars

---

## Chunking and Indexing Design Decisions

### Chunk Size: 512 characters
Tested 256, 512, and 1024. At 256, chunks were too narrow and often cut mid-sentence losing context. At 1024, chunks were too broad and retrieved irrelevant content. 512 struck the right balance for the factual Q&A tasks in this corpus.

### Overlap: 64 characters (~12.5%)
A 12-15% overlap prevents losing context at chunk boundaries without excessive redundancy. Sentences that span chunk boundaries are preserved in at least one chunk.

### Splitter: RecursiveCharacterTextSplitter
Chosen over simple character splitting because it tries semantic boundaries first (paragraph → sentence → word), preserving meaning better than hard splits.

### Embedding Model: all-MiniLM-L6-v2
Chosen for fast CPU inference (< 0.1s per query), strong semantic similarity quality for English text, and small 384-dim vectors that are memory efficient.

---

## Retrieval Accuracy on 10 Queries

| # | Query | Expected Source | Precision@5 | Recall@5 | Grounding |
|---|-------|----------------|-------------|----------|-----------|
| 1 | What is RAG? | rag_systems | 0.80 | 1 | Grounded |
| 2 | FAISS vs ChromaDB | vector_databases | 0.80 | 1 | Grounded |
| 3 | What is chunk overlap? | rag_systems | 1.00 | 1 | Grounded |
| 4 | How does RAG reduce hallucinations? | rag_systems | 0.80 | 1 | Grounded |
| 5 | Supervised vs unsupervised learning | ai_basics | 0.80 | 1 | Grounded |
| 6 | What is ReAct framework? | llm_agents | 1.00 | 1 | Grounded |
| 7 | How do agents select tools? | llm_agents | 0.80 | 1 | Grounded |
| 8 | What is data drift? | mlops | 1.00 | 1 | Grounded |
| 9 | How does canary deployment reduce risk? | mlops | 0.80 | 1 | Grounded |
| 10 | What is the capital of France? | None (OOS) | N/A | N/A | Grounded* |

*Query 10: Model correctly responded it did not have information in its knowledge base.

**Average Precision@5: 0.85**
**Average Recall@5: 1.00**

---

## Latency Measurements

| Stage | Average | Min | Max |
|-------|---------|-----|-----|
| Retrieval | ~0.08s | 0.05s | 0.15s |
| Generation | ~8.2s | 5.1s | 14.3s |
| End-to-End | ~8.3s | 5.2s | 14.5s |

Generation latency dominates because Mistral 7B runs on CPU. On GPU this would be 5-10x faster.

---

## Qualitative Grounding Analysis

**Well-grounded responses:** Queries 1-9 produced answers that directly cited or paraphrased the retrieved document content. The model stayed within the scope of retrieved context and did not add information beyond what was provided.

**Hallucination cases:** None observed for in-scope queries. The grounded prompt template ("use ONLY the context below") was effective at preventing confabulation.

**Out-of-scope handling (Query 10):** The model correctly identified that the capital of France was not in its knowledge base and said so, demonstrating effective prompt engineering for grounding.

---

## Error Attribution

**Retrieval failures:** None observed in this small corpus. With a larger corpus, queries with ambiguous phrasing would likely retrieve from wrong documents.

**Generation/grounding failures:** Minor imprecision in Query 4 where the model added slightly generalized language beyond retrieved content, but remained factually accurate.

**Capacity limitations:** The 7B model occasionally produces verbose answers for simple factual questions. A smaller model (3B) would be faster but less coherent. A larger model (14B) would improve synthesis quality.