from agent_cap.core.types import Trace
from agent_cap.visualization.timeline import TimelineVisualizer

# trace = Trace.load("baselines/synthetic_e2e_runs/trace_001.json")

trace = Trace.load("baselines/hybrid_e2e_runs/trace_001.json")




viz = TimelineVisualizer(trace)

print(viz.to_ascii(width=100))
print(viz.summary_table())


python - << 'EOF'
from agent_cap.core.types import Trace
from agent_cap.visualization.timeline import TimelineVisualizer

trace = Trace.load("baselines/hybrid_e2e_runs/trace_001.json")
viz = TimelineVisualizer(trace)

print(viz.to_ascii(width=100))
print(viz.summary_table())
EOF