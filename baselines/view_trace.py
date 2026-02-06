import sys
from agent_cap.core.types import Trace
from agent_cap.visualization.timeline import TimelineVisualizer


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python view_trace.py <trace_json_path> [width]")

    trace_path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) >= 3 else 100

    trace = Trace.load(trace_path)
    viz = TimelineVisualizer(trace)

    print(viz.to_ascii(width=width))
    print(viz.summary_table())


if __name__ == "__main__":
    main()