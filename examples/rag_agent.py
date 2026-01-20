#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Agent Example.

Demonstrates benchmarking a typical RAG workflow with:
- Document retrieval
- Embedding computation
- LLM generation
- Tool usage
"""

import time
import random
from typing import List, Dict, Any

from agent_cap import Tracer, StepType, TimelineVisualizer


class MockLLM:
    """Mock LLM for demonstration."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Simulate LLM generation with prefill and decode phases."""

        # Prefill phase - process input tokens
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        with self.tracer.step("llm_prefill", StepType.PREFILL) as step:
            # Prefill is compute-bound, ~10ms per 100 tokens
            prefill_time = (input_tokens / 100) * 0.01
            time.sleep(prefill_time + random.uniform(0.01, 0.03))
            step.metadata["input_tokens"] = int(input_tokens)

        # Decode phase - generate output tokens
        with self.tracer.step("llm_decode", StepType.DECODE) as step:
            # Decode is memory-bandwidth bound, ~20ms per token
            decode_time = max_tokens * 0.02
            time.sleep(decode_time * random.uniform(0.8, 1.2))
            step.metadata["output_tokens"] = max_tokens

        return "Generated response based on the context..."


class MockRetriever:
    """Mock document retriever."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def embed_query(self, query: str) -> List[float]:
        """Compute query embedding."""
        with self.tracer.step("query_embedding", StepType.EMBEDDING) as step:
            # Embedding computation
            time.sleep(random.uniform(0.02, 0.05))
            step.metadata["model"] = "text-embedding-ada-002"
            step.metadata["dimensions"] = 1536

        return [random.random() for _ in range(1536)]

    def search(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        with self.tracer.step("vector_search", StepType.RETRIEVAL) as step:
            # Vector DB search
            time.sleep(random.uniform(0.05, 0.15))
            step.metadata["top_k"] = top_k
            step.metadata["index_type"] = "HNSW"

        return [{"id": i, "score": random.random(), "text": f"Document {i}"} for i in range(top_k)]


class MockToolExecutor:
    """Mock tool executor."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool."""
        with self.tracer.step(f"tool_{tool_name}", StepType.TOOL_CALLING) as step:
            # Tool execution varies
            time.sleep(random.uniform(0.05, 0.2))
            step.metadata["tool"] = tool_name
            step.metadata["args"] = kwargs

        return {"status": "success", "result": "Tool output"}


class RAGAgent:
    """A simple RAG agent for demonstration."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.llm = MockLLM(tracer)
        self.retriever = MockRetriever(tracer)
        self.tools = MockToolExecutor(tracer)

    def run(self, query: str) -> str:
        """Run the RAG pipeline."""

        # Step 1: Planning
        with self.tracer.step("analyze_query", StepType.PLANNING) as step:
            time.sleep(random.uniform(0.02, 0.05))
            step.metadata["query_length"] = len(query)
            needs_tools = "calculate" in query.lower() or "search" in query.lower()
            step.metadata["needs_tools"] = needs_tools

        # Step 2: Retrieval
        query_embedding = self.retriever.embed_query(query)
        documents = self.retriever.search(query_embedding, top_k=5)

        # Step 3: Context building
        with self.tracer.step("build_context", StepType.REASONING) as step:
            context = "\n".join([d["text"] for d in documents])
            time.sleep(random.uniform(0.01, 0.02))
            step.metadata["context_length"] = len(context)

        # Step 4: Optional tool usage
        if needs_tools:
            tool_result = self.tools.execute("web_search", query=query)

        # Step 5: Generate response
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        response = self.llm.generate(prompt, max_tokens=128)

        # Step 6: Post-processing
        with self.tracer.step("format_response", StepType.OTHER) as step:
            time.sleep(random.uniform(0.005, 0.01))
            step.metadata["response_length"] = len(response)

        return response


def run_rag_benchmark(num_queries: int = 3):
    """Run RAG agent benchmark."""

    print("=" * 60)
    print("RAG Agent Benchmark")
    print("=" * 60)

    tracer = Tracer("rag-agent-benchmark")

    with tracer:
        agent = RAGAgent(tracer)

        queries = [
            "What is the capital of France?",
            "Calculate the sum of 1 to 100",
            "Explain quantum computing in simple terms",
        ]

        for i, query in enumerate(queries[:num_queries]):
            print(f"\nQuery {i + 1}: {query}")

            with tracer.step(f"query_{i + 1}", StepType.OTHER) as step:
                response = agent.run(query)
                step.metadata["query"] = query

            print(f"Response: {response[:50]}...")

    trace = tracer.get_trace()

    # Visualization
    viz = TimelineVisualizer(trace)

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(viz.to_ascii(width=100))
    print(viz.summary_table())

    # Save outputs
    trace.save("rag_agent_trace.json")
    print("\nTrace saved to: rag_agent_trace.json")

    try:
        viz.save_html("rag_agent_timeline.html")
        print("Timeline saved to: rag_agent_timeline.html")
        print("\nOpen rag_agent_timeline.html in a browser to see the interactive timeline!")
    except ImportError:
        print("\nNote: Install plotly for interactive visualization: pip install plotly")

    return trace


if __name__ == "__main__":
    run_rag_benchmark()
