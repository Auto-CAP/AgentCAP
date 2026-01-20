# Agent-CAP

**Benchmarking of Cost, Accuracy, and Performance for Agentic AI Systems**

Agent-CAP is a step-level benchmarking framework that decomposes agentic workloads into atomic operations and measures each step's latency. It provides Nsight Systems-like timeline visualization for analyzing agent performance.

## Features

- **Step-level Profiling**: Trace individual operations in agentic workflows
- **Multiple Timer Backends**: Support for `time.perf_counter()` and CUDA events
- **Timeline Visualization**: Interactive HTML timelines similar to Nsight Systems
- **Zero Dependencies**: Core functionality works without any external packages
- **Easy Integration**: Simple decorators and context managers for instrumentation

## Installation

```bash
# Basic installation
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With CUDA timing support
pip install -e ".[cuda]"

# Full installation
pip install -e ".[all]"
```

## Quick Start

```python
from agent_cap import Tracer, StepType, TimelineVisualizer

# Create a tracer
tracer = Tracer("my-agent-workflow")

# Use context managers to trace steps
with tracer:
    with tracer.step("planning", StepType.PLANNING):
        # Your planning code here
        plan = agent.plan(task)

    with tracer.step("retrieval", StepType.RETRIEVAL):
        # Your retrieval code here
        docs = retriever.search(query)

    with tracer.step("reasoning", StepType.REASONING):
        # Your LLM inference code here
        response = llm.generate(prompt)

# Get the trace and visualize
trace = tracer.get_trace()

# Save to JSON
trace.save("trace.json")

# Create visualization
viz = TimelineVisualizer(trace)
viz.save_html("timeline.html")  # Interactive HTML
print(viz.to_ascii())           # Terminal output
```

## Step Types

Agent-CAP categorizes workflow steps based on their computational characteristics:

| Step Type | Description | Bottleneck |
|-----------|-------------|------------|
| `PLANNING` | Task decomposition, high-level decisions | Compute-bound |
| `REASONING` | Chain-of-thought, inference | Memory-bandwidth bound |
| `RETRIEVAL` | Document fetch, RAG | I/O bound |
| `TOOL_CALLING` | External API calls | Network/CPU bound |
| `CODE_EXECUTION` | Running generated code | CPU/sandbox bound |
| `PREFILL` | LLM prefill phase | Compute-bound |
| `DECODE` | LLM decode phase | Memory-bandwidth bound |
| `EMBEDDING` | Embedding computation | Compute-bound |

## Decorator API

```python
from agent_cap import tracer, StepType

@tracer("fetch_data", StepType.RETRIEVAL)
def fetch_data(query):
    return db.query(query)

@tracer("generate", StepType.DECODE)
def generate(prompt):
    return llm(prompt)
```

## Visualization

### Interactive HTML Timeline

```python
from agent_cap import TimelineVisualizer

viz = TimelineVisualizer(trace)
viz.save_html("timeline.html")
viz.show()  # Opens in browser
```

### ASCII Timeline (Terminal)

```python
print(viz.to_ascii())
```

Output:
```
Timeline: my-agent-workflow
Total Duration: 1234.56 ms
============================================================
                        |    0  250  500  750 1000
                        +--------------------------------
planning               |████                    (150.2ms)
retrieval              |    ██████              (280.5ms)
reasoning              |          ████████████  (450.3ms)
============================================================
```

### Summary Table

```python
print(viz.summary_table())
```

## Examples

Run the example scripts:

```bash
# Simple agent workflow
python examples/simple_agent.py

# RAG agent benchmark
python examples/rag_agent.py
```

## CUDA Timing

For precise GPU timing, use CUDA events:

```python
tracer = Tracer("gpu-workflow", use_cuda=True)

with tracer.step("gpu_compute", StepType.PREFILL):
    # GPU operations are timed with CUDA events
    model(input_tensor)
```

## License

Apache 2.0
