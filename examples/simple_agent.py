#!/usr/bin/env python3
"""
Simple example demonstrating Agent-CAP benchmarking.

This example simulates an agentic workflow with various step types
and shows how to use the tracer and visualization.
"""

import time
import random
from agent_cap import Tracer, StepType, TimelineVisualizer


def simulate_work(min_ms: float = 10, max_ms: float = 100) -> None:
    """Simulate work by sleeping for a random duration."""
    duration = random.uniform(min_ms, max_ms) / 1000
    time.sleep(duration)


def run_simple_agent():
    """Run a simple simulated agent workflow."""

    # Create a tracer for this workflow
    tracer = Tracer("simple-agent-workflow", use_cuda=False)

    print("Starting Agent-CAP benchmarking demo...")
    print("=" * 50)

    with tracer:
        # Step 1: Planning - Decompose the task
        with tracer.step("task_planning", StepType.PLANNING) as step:
            print("Planning: Decomposing task into subtasks...")
            simulate_work(50, 150)
            step.metadata["subtasks"] = 3

        # Step 2: Retrieval - Fetch relevant context
        with tracer.step("fetch_documents", StepType.RETRIEVAL) as step:
            print("Retrieval: Fetching relevant documents...")
            simulate_work(100, 300)
            step.metadata["documents_fetched"] = 5

        # Step 3: Embedding - Compute embeddings
        with tracer.step("compute_embeddings", StepType.EMBEDDING) as step:
            print("Embedding: Computing document embeddings...")
            simulate_work(50, 100)
            step.metadata["embedding_dim"] = 768

        # Step 4: Reasoning - LLM inference (prefill + decode)
        with tracer.step("llm_prefill", StepType.PREFILL) as step:
            print("LLM Prefill: Processing input tokens...")
            simulate_work(100, 200)
            step.metadata["input_tokens"] = 1024

        with tracer.step("llm_decode", StepType.DECODE) as step:
            print("LLM Decode: Generating output tokens...")
            simulate_work(200, 500)
            step.metadata["output_tokens"] = 256

        # Step 5: Tool calling - Execute external tools
        with tracer.step("api_call", StepType.TOOL_CALLING) as step:
            print("Tool Calling: Making API request...")
            simulate_work(50, 150)
            step.metadata["api"] = "search"

        # Step 6: Code execution - Run generated code
        with tracer.step("run_code", StepType.CODE_EXECUTION) as step:
            print("Code Execution: Running generated code...")
            simulate_work(100, 200)
            step.metadata["language"] = "python"

        # Step 7: Final reasoning - Generate response
        with tracer.step("generate_response", StepType.REASONING) as step:
            print("Reasoning: Generating final response...")
            simulate_work(150, 300)
            step.metadata["response_type"] = "text"

    # Get the trace
    trace = tracer.get_trace()

    print("\n" + "=" * 50)
    print("Workflow completed!")
    print("=" * 50)

    # Create visualizer
    viz = TimelineVisualizer(trace)

    # Print ASCII timeline (works without plotly)
    print(viz.to_ascii())

    # Print summary table
    print(viz.summary_table())

    # Save trace to JSON
    trace.save("simple_agent_trace.json")
    print("\nTrace saved to: simple_agent_trace.json")

    # Save HTML visualization (requires plotly)
    try:
        viz.save_html("simple_agent_timeline.html")
        print("Timeline saved to: simple_agent_timeline.html")
    except ImportError:
        print("Note: Install plotly for interactive HTML visualization")
        print("  pip install plotly")

    return trace


def run_parallel_agent():
    """Demonstrate parallel step execution."""

    tracer = Tracer("parallel-agent-workflow")

    print("\n" + "=" * 50)
    print("Running parallel workflow demo...")
    print("=" * 50)

    with tracer:
        # Sequential planning
        with tracer.step("planning", StepType.PLANNING):
            simulate_work(50, 100)

        # Simulate parallel retrieval (recorded with different thread IDs)
        # In real code, you'd use actual threading
        base_time = tracer._get_relative_time()

        # Simulate 3 parallel retrievals
        for i in range(3):
            tracer.record_step(
                name=f"parallel_fetch_{i}",
                step_type=StepType.RETRIEVAL,
                start_time=base_time,
                end_time=base_time + random.uniform(0.05, 0.15),
                thread_id=f"worker-{i}",
                metadata={"source": f"db_{i}"}
            )

        # Wait for "parallel" work
        time.sleep(0.15)

        # Sequential processing
        with tracer.step("merge_results", StepType.REASONING):
            simulate_work(30, 60)

        with tracer.step("generate_output", StepType.DECODE):
            simulate_work(100, 200)

    trace = tracer.get_trace()
    viz = TimelineVisualizer(trace)

    print(viz.to_ascii(width=100))
    print(viz.summary_table())

    # Save outputs
    trace.save("parallel_agent_trace.json")
    try:
        viz.save_html("parallel_agent_timeline.html")
        print("\nTimeline saved to: parallel_agent_timeline.html")
    except ImportError:
        pass

    return trace


if __name__ == "__main__":
    # Run both demos
    trace1 = run_simple_agent()
    trace2 = run_parallel_agent()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)
