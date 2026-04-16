# RAG Pipeline Architecture

## Data Flow Diagram

┌─────────────────────────────────────────────────────────────────┐
│                        DOCUMENT CORPUS                          │
│  ai_basics.txt | rag_systems.txt | mlops.txt | llm_agents.txt  │
└────────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                     CHUNKER                                      │
│  RecursiveCharacterTextSplitter                                  │
│  chunk_size=512 | chunk_overlap=64                               │
│  Decision: Split on paragraphs → sentences → words              │
└────────────────────────────┬────────────────────────────────────┘
│  chunks[]
▼
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDER                                     │
│  SentenceTransformer: all-MiniLM-L6-v2                          │
│  Output: 384-dimensional dense vectors                           │
│  Decision: Lightweight, CPU-friendly, strong semantic quality    │
└────────────────────────────┬────────────────────────────────────┘
│  embeddings[]
▼
┌─────────────────────────────────────────────────────────────────┐
│                   VECTOR STORE (ChromaDB)                        │
│  Stores: embeddings + original text + metadata                   │
│  Index type: HNSW (approximate nearest neighbor)                 │
│  Persisted to: ./chroma_db/                                      │
└──────────────┬──────────────────────────────────────────────────┘
│
┌──────────▼──────────┐
│    USER QUERY        │
└──────────┬──────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVER                                    │
│  Similarity search: cosine similarity                            │
│  Returns top-k=5 most relevant chunks                            │
└────────────────────────────┬────────────────────────────────────┘
│  retrieved context
▼
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATOR (LLM)                              │
│  Model: mistral:7b-instruct via Ollama                           │
│  Prompt: grounded prompt enforcing context-only answers          │
│  Output: grounded natural language answer                        │
└────────────────────────────┬────────────────────────────────────┘
│
▼
┌────────────────┐
│  FINAL ANSWER  │
└────────────────┘

## Agent Architecture

┌─────────────────────────────────────────────────────────────────┐
│                      USER TASK                                   │
└────────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│               AGENT CONTROLLER (LLM-driven)                      │
│                                                                  │
│  Tool Selection Policy:                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Factual question? → retriever                        │        │
│  │ Summarization task? → summarizer                     │        │
│  │ Both? → retrieve_then_summarize                      │        │
│  │ Simple/math/conversational? → direct                 │        │
│  └─────────────────────────────────────────────────────┘        │
└────────┬──────────────────────┬──────────────────────┬──────────┘
│                      │                      │
▼                      ▼                      ▼
┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ RETRIEVER TOOL │   │ SUMMARIZER TOOL  │   │  DIRECT (LLM)   │
│                │   │                  │   │                 │
│ ChromaDB       │   │ LLM prompt       │   │ No tool call    │
│ top-4 chunks   │   │ bullet summary   │   │                 │
└────────┬───────┘   └────────┬─────────┘   └────────┬────────┘
│                    │                       │
└────────────────────┴───────────────────────┘
│
▼
┌──────────────────────┐
│   ANSWER GENERATOR   │
│   (Mistral 7B)       │
└──────────┬───────────┘
│
▼
┌──────────────────────┐
│    TRACE LOGGER      │
│  agent_traces/*.json │
└──────────────────────┘