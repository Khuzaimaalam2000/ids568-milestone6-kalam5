import os
import json
import time
from datetime import datetime
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────

class RetrieverTool:
    """Tool that searches the vector store for relevant information."""
    
    name = "retriever"
    description = "Use this tool to look up factual information from the knowledge base. Best for questions about AI, RAG, MLOps, agents, or vector databases."
    
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def run(self, query: str) -> dict:
        t_start = time.time()
        docs = self.retriever.invoke(query)
        latency = time.time() - t_start
        
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata["source"].split("/")[-1] for d in docs]))
        
        return {
            "tool": "retriever",
            "query": query,
            "context": context,
            "sources": sources,
            "latency": latency,
            "num_docs": len(docs)
        }


class SummarizerTool:
    """Tool that summarizes a block of text into key bullet points."""
    
    name = "summarizer"
    description = "Use this tool to summarize or condense a long piece of text into key points. Best for distillation and synthesis tasks."
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""Summarize the following text into 3-5 clear bullet points.
            
Text: {text}

Summary (bullet points):""",
            input_variables=["text"]
        )
    
    def run(self, text: str) -> dict:
        t_start = time.time()
        prompt_text = self.prompt.format(text=text)
        result = self.llm.invoke(prompt_text)
        latency = time.time() - t_start
        
        return {
            "tool": "summarizer",
            "input_length": len(text),
            "summary": result,
            "latency": latency
        }


# ─────────────────────────────────────────────
# AGENT CONTROLLER
# ─────────────────────────────────────────────

class AgentController:
    """
    Multi-tool agent that intelligently selects between retriever and summarizer.
    
    Tool selection policy:
    - If the task requires factual lookup → use retriever first
    - If the task requires condensing/summarizing text → use summarizer
    - If the task requires both → retrieve first, then summarize retrieved content
    - If the task is purely generative (no lookup needed) → use LLM directly
    """
    
    def __init__(self, retriever_tool, summarizer_tool, llm):
        self.retriever = retriever_tool
        self.summarizer = summarizer_tool
        self.llm = llm
        self.traces = []
        
        # Tool selection prompt
        self.selector_prompt = PromptTemplate(
            template="""You are an agent controller. Given a task, decide which tool to use.

Available tools:
- retriever: Look up factual information from the knowledge base
- summarizer: Summarize or condense a piece of text
- direct: Answer directly without any tool (for simple/conversational tasks)
- retrieve_then_summarize: First retrieve info, then summarize it

Task: {task}

Respond with ONLY one of these exact words: retriever, summarizer, direct, retrieve_then_summarize

Tool choice:""",
            input_variables=["task"]
        )
        
        # Final answer prompt
        self.answer_prompt = PromptTemplate(
            template="""Using the information below, answer the task clearly and completely.

Task: {task}

Available information:
{context}

Answer:""",
            input_variables=["task", "context"]
        )
    
    def select_tool(self, task: str) -> str:
        """Use LLM to decide which tool to use."""
        prompt = self.selector_prompt.format(task=task)
        response = self.llm.invoke(prompt).strip().lower()
        
        # Parse and validate response
        valid_tools = ["retriever", "summarizer", "direct", "retrieve_then_summarize"]
        for tool in valid_tools:
            if tool in response:
                return tool
        return "retriever"  # Default to retriever if unclear
    
    def run(self, task: str, task_id: int = 0) -> dict:
        """Execute a task using the appropriate tool(s) and log the trace."""
        
        trace = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "steps": [],
            "final_answer": "",
            "total_latency": 0
        }
        
        t_total_start = time.time()
        
        # Step 1: Tool selection
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task}")
        print(f"{'='*60}")
        
        t_select_start = time.time()
        tool_choice = self.select_tool(task)
        select_latency = time.time() - t_select_start
        
        print(f"[AGENT] Tool selected: {tool_choice} ({select_latency:.2f}s)")
        
        trace["steps"].append({
            "step": 1,
            "action": "tool_selection",
            "decision": tool_choice,
            "reasoning": f"LLM selected '{tool_choice}' based on task analysis",
            "latency": select_latency
        })
        
        context = ""
        
        # Step 2: Execute selected tool(s)
        if tool_choice == "retriever":
            result = self.retriever.run(task)
            context = result["context"]
            print(f"[RETRIEVER] Retrieved {result['num_docs']} docs from: {result['sources']} ({result['latency']:.2f}s)")
            trace["steps"].append({
                "step": 2,
                "action": "retriever_called",
                "query": task,
                "sources": result["sources"],
                "num_docs": result["num_docs"],
                "context_preview": context[:200] + "...",
                "latency": result["latency"]
            })
            
        elif tool_choice == "summarizer":
            result = self.summarizer.run(task)
            context = result["summary"]
            print(f"[SUMMARIZER] Summarized {result['input_length']} chars ({result['latency']:.2f}s)")
            trace["steps"].append({
                "step": 2,
                "action": "summarizer_called",
                "input_length": result["input_length"],
                "summary_preview": context[:200],
                "latency": result["latency"]
            })
            
        elif tool_choice == "retrieve_then_summarize":
            # First retrieve
            ret_result = self.retriever.run(task)
            print(f"[RETRIEVER] Retrieved {ret_result['num_docs']} docs ({ret_result['latency']:.2f}s)")
            trace["steps"].append({
                "step": 2,
                "action": "retriever_called",
                "query": task,
                "sources": ret_result["sources"],
                "num_docs": ret_result["num_docs"],
                "latency": ret_result["latency"]
            })
            
            # Then summarize retrieved content
            sum_result = self.summarizer.run(ret_result["context"])
            context = sum_result["summary"]
            print(f"[SUMMARIZER] Summarized retrieved content ({sum_result['latency']:.2f}s)")
            trace["steps"].append({
                "step": 3,
                "action": "summarizer_called_on_retrieved",
                "input_length": sum_result["input_length"],
                "latency": sum_result["latency"]
            })
            
        else:  # direct
            context = "No external tool used - answering from model knowledge."
            print(f"[DIRECT] Answering directly without tool")
            trace["steps"].append({
                "step": 2,
                "action": "direct_answer",
                "reasoning": "Task does not require external tool lookup"
            })
        
        # Step 3: Generate final answer
        t_ans_start = time.time()
        answer_prompt = self.answer_prompt.format(task=task, context=context)
        final_answer = self.llm.invoke(answer_prompt)
        ans_latency = time.time() - t_ans_start
        
        print(f"[ANSWER] Generated in {ans_latency:.2f}s")
        print(f"[ANSWER] {final_answer[:300]}...")
        
        trace["steps"].append({
            "step": len(trace["steps"]) + 1,
            "action": "answer_generation",
            "latency": ans_latency
        })
        
        trace["final_answer"] = final_answer
        trace["total_latency"] = time.time() - t_total_start
        trace["tool_used"] = tool_choice
        
        print(f"[DONE] Total latency: {trace['total_latency']:.2f}s")
        
        self.traces.append(trace)
        return trace
    
    def save_traces(self, output_dir="agent_traces"):
        """Save all traces to individual JSON files."""
        Path(output_dir).mkdir(exist_ok=True)
        for trace in self.traces:
            filename = f"{output_dir}/task_{trace['task_id']:02d}.json"
            with open(filename, "w") as f:
                json.dump(trace, f, indent=2)
        print(f"Saved {len(self.traces)} traces to {output_dir}/")


# ─────────────────────────────────────────────
# MAIN: RUN 10 EVALUATION TASKS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    
    # Load vector store (must have run rag_pipeline.ipynb first)
    print("Loading vector store and models...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    llm = OllamaLLM(model="mistral:7b-instruct", temperature=0.1)
    
    # Initialize tools
    retriever_tool = RetrieverTool(vectorstore)
    summarizer_tool = SummarizerTool(llm)
    
    # Initialize agent
    agent = AgentController(retriever_tool, summarizer_tool, llm)
    
    # 10 diverse multi-step evaluation tasks
    tasks = [
        # Factual retrieval tasks
        "What is RAG and how does it reduce hallucinations?",
        "Explain the difference between FAISS and ChromaDB for vector search.",
        "What MLOps practices help with model monitoring in production?",
        # Summarization tasks
        "Summarize the key concepts of LLM agents in a few bullet points: LLM agents use large language models as reasoning engines. They can select tools, perform multi-step reasoning, maintain memory, and plan tasks. Tool use allows calling external functions. ReAct combines reasoning and acting. Traces log decisions.",
        # Retrieve-then-summarize tasks
        "Retrieve information about chunking strategies and summarize the key design decisions.",
        "Find and summarize what the knowledge base says about vector database options.",
        # Multi-hop reasoning tasks
        "How would you combine RAG with an agent system for better factual accuracy?",
        "What evaluation metrics should I use when building a RAG system?",
        # Edge cases
        "What is 2 + 2?",  # Direct - no retrieval needed
        "Explain data drift and how canary deployments help mitigate its effects in production."
    ]
    
    print(f"\nRunning {len(tasks)} evaluation tasks...\n")
    all_traces = []
    
    for i, task in enumerate(tasks, 1):
        trace = agent.run(task, task_id=i)
        all_traces.append(trace)
    
    # Save all traces
    agent.save_traces()
    
    # Print summary
    print(f"\n{'='*60}")
    print("AGENT EVALUATION SUMMARY")
    print(f"{'='*60}")
    tool_counts = {}
    for t in all_traces:
        tool = t["tool_used"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    print(f"Tool usage breakdown: {tool_counts}")
    avg_latency = sum(t["total_latency"] for t in all_traces) / len(all_traces)
    print(f"Average total latency: {avg_latency:.2f}s")
    print(f"All traces saved to agent_traces/")