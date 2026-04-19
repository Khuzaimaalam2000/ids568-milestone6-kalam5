import os
import time
import json
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


# -----------------------------
# RETRIEVER TOOL
# -----------------------------
class RetrieverTool:

    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

    def run(self, query: str):
        start = time.time()
        docs = self.retriever.invoke(query)
        latency = time.time() - start

        context = "\n\n".join([d.page_content for d in docs])

        return {
            "context": context,
            "latency": latency,
            "num_docs": len(docs)
        }


# -----------------------------
# AGENT CONTROLLER (FAST VERSION)
# -----------------------------
class AgentController:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.answer_prompt = PromptTemplate(
            template="""
You are a precise AI assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in knowledge base."

Keep answer concise (3-6 lines max).

Question:
{task}

Context:
{context}

Answer:
""",
            input_variables=["task", "context"]
        )

    # -----------------------------
    # FAST RULE-BASED TOOL SELECTION
    # -----------------------------
    def select_tool(self, task: str):

        task_lower = task.lower()

        if any(x in task_lower for x in ["summarize", "summary"]):
            return "summarizer"

        if any(x in task_lower for x in ["2+2", "math", "calculate"]):
            return "direct"

        return "retriever"

    # -----------------------------
    # RUN TASK
    # -----------------------------
    def run(self, task, task_id=0):

        print("\n" + "="*60)
        print(f"TASK {task_id}: {task}")
        print("="*60)

        tool = self.select_tool(task)
        print(f"[AGENT] Tool selected: {tool}")

        context = ""

        # -------------------------
        # RETRIEVAL
        # -------------------------
        if tool == "retriever":

            result = self.retriever.run(task)

            if result["num_docs"] == 0:
                print("[WARNING] No documents retrieved")

            context = result["context"]
            retrieval_latency = result["latency"]
            print(f"[RETRIEVER] docs: {result['num_docs']} ({retrieval_latency:.2f}s)")

        # -------------------------
        # SUMMARIZER (lightweight)
        # -------------------------
        elif tool == "summarizer":
            context = "Summarize based on general knowledge."
            retrieval_latency = 0.0

        # -------------------------
        # DIRECT
        # -------------------------
        else:
            context = "General knowledge."
            retrieval_latency = 0.0

        # -------------------------
        # SINGLE LLM CALL (FAST)
        # -------------------------
        start = time.time()
        final = self.llm.invoke(
            self.answer_prompt.format(task=task, context=context)
        )
        generation_latency = time.time() - start

        print(f"[LLM] Response time: {generation_latency:.2f}s")
        print("\nFINAL ANSWER:\n")
        print(final)

        return {
            "task_id": task_id,
            "task": task,
            "tool_used": tool,
            "answer": final,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "total_latency": retrieval_latency + generation_latency
        }


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    llm = OllamaLLM(
        model="mistral:7b-instruct",
        temperature=0,
    )

    retriever = RetrieverTool(vectorstore)
    agent = AgentController(retriever, llm)

    tasks = [
        "What is RAG?",
        "Explain vector databases",
        "What is chunking?",
        "What is FAISS?",
        "What is MLOps?",
        "What is data drift?",
        "Explain LLM agents",
        "What is tool use?",
        "What is 2+2?",
        "Summarize RAG systems"
    ]

    Path("agent_traces").mkdir(exist_ok=True)

    for i, t in enumerate(tasks, 1):
        result = agent.run(t, i)
        with open(f"agent_traces/task_{i:02d}.json", "w") as f:
            json.dump(result, f, indent=2)

    print("\nAll traces saved to agent_traces/")