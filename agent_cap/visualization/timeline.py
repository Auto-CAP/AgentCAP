"""
Timeline visualization for Agent-CAP traces.

Creates Nsight Systems-like timeline views showing step execution over time.
"""

import json
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

from agent_cap.core.types import Trace, Step, StepType


# Color scheme for different step types (Nsight-inspired)
STEP_COLORS: Dict[StepType, str] = {
    StepType.PLANNING: "#4CAF50",      # Green
    StepType.REASONING: "#2196F3",     # Blue
    StepType.RETRIEVAL: "#FF9800",     # Orange
    StepType.TOOL_CALLING: "#9C27B0",  # Purple
    StepType.CODE_EXECUTION: "#F44336", # Red
    StepType.PREFILL: "#00BCD4",       # Cyan
    StepType.DECODE: "#3F51B5",        # Indigo
    StepType.EMBEDDING: "#FFEB3B",     # Yellow
    StepType.OTHER: "#9E9E9E",         # Gray
}


class TimelineVisualizer:
    """
    Visualizes Agent-CAP traces as interactive timelines.

    Creates HTML files with interactive Gantt-chart style visualizations
    similar to NVIDIA Nsight Systems.

    Usage:
        viz = TimelineVisualizer(trace)
        viz.save_html("timeline.html")

        # Or with plotly
        fig = viz.to_plotly()
        fig.show()
    """

    def __init__(self, trace: Trace):
        """
        Initialize visualizer with a trace.

        Args:
            trace: The Trace object to visualize
        """
        self.trace = trace
        self._colors = STEP_COLORS.copy()

    def set_color(self, step_type: StepType, color: str) -> None:
        """Set custom color for a step type."""
        self._colors[step_type] = color

    def to_plotly(self, height: Optional[int] = None):
        """
        Create a Plotly figure of the timeline.

        Args:
            height: Optional height in pixels (auto-calculated if not provided)

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError(
                "plotly is required for visualization. Install with: pip install plotly"
            )

        steps = self.trace.steps
        if not steps:
            fig = go.Figure()
            fig.add_annotation(text="No steps recorded", x=0.5, y=0.5)
            return fig

        # Group steps by thread
        threads: Dict[str, List[Step]] = {}
        for step in steps:
            if step.thread_id not in threads:
                threads[step.thread_id] = []
            threads[step.thread_id].append(step)

        # Sort threads and assign y positions
        thread_order = sorted(threads.keys())
        thread_y: Dict[str, int] = {t: i for i, t in enumerate(thread_order)}

        # Calculate figure height
        if height is None:
            height = max(400, len(thread_order) * 60 + 150)

        fig = go.Figure()

        # Add bars for each step
        for step in steps:
            y_pos = thread_y[step.thread_id]
            color = self._colors.get(step.step_type, "#9E9E9E")

            # Create hover text
            hover_text = (
                f"<b>{step.name}</b><br>"
                f"Type: {step.step_type}<br>"
                f"Duration: {step.duration_ms:.2f} ms<br>"
                f"Start: {step.start_time * 1000:.2f} ms<br>"
                f"End: {step.end_time * 1000:.2f} ms"
            )

            if step.metadata:
                for k, v in step.metadata.items():
                    hover_text += f"<br>{k}: {v}"

            # Add bar
            fig.add_trace(go.Bar(
                x=[step.duration * 1000],  # Convert to ms
                y=[f"{step.thread_id}"],
                base=[step.start_time * 1000],
                orientation='h',
                name=step.name,
                marker=dict(color=color),
                text=step.name if step.duration_ms > 50 else "",  # Show text only if wide enough
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            ))

        # Create legend entries for step types
        legend_added = set()
        for step in steps:
            if step.step_type not in legend_added:
                color = self._colors.get(step.step_type, "#9E9E9E")
                fig.add_trace(go.Bar(
                    x=[0],
                    y=[thread_order[0]],
                    orientation='h',
                    name=str(step.step_type),
                    marker=dict(color=color),
                    showlegend=True,
                    visible="legendonly",
                ))
                legend_added.add(step.step_type)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Agent-CAP Timeline: {self.trace.name}",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Time (ms)",
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinecolor="rgba(128, 128, 128, 0.5)",
            ),
            yaxis=dict(
                title="Thread",
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                categoryorder="array",
                categoryarray=thread_order,
            ),
            barmode="overlay",
            height=height,
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            font=dict(color="#ffffff"),
            legend=dict(
                title="Step Types",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=100, r=20, t=80, b=60),
        )

        return fig

    def to_html(self, include_plotlyjs: bool = True) -> str:
        """
        Generate HTML string for the timeline.

        Args:
            include_plotlyjs: If True, include plotly.js in the HTML

        Returns:
            HTML string
        """
        fig = self.to_plotly()
        return fig.to_html(include_plotlyjs=include_plotlyjs)

    def save_html(self, filepath: str, include_plotlyjs: bool = True) -> None:
        """
        Save timeline visualization to an HTML file.

        Args:
            filepath: Path to save the HTML file
            include_plotlyjs: If True, include plotly.js in the HTML
        """
        html = self.to_html(include_plotlyjs=include_plotlyjs)
        Path(filepath).write_text(html)

    def show(self) -> None:
        """Display the timeline in a browser or notebook."""
        fig = self.to_plotly()
        fig.show()

    def to_ascii(self, width: int = 80) -> str:
        """
        Generate ASCII timeline representation.

        Useful for terminal output when plotly is not available.

        Args:
            width: Width of the ASCII output in characters

        Returns:
            ASCII string representation of the timeline
        """
        steps = self.trace.steps
        if not steps:
            return "No steps recorded"

        # Calculate time bounds
        min_time = min(s.start_time for s in steps)
        max_time = max(s.end_time for s in steps)
        time_range = max_time - min_time or 1.0

        # Available width for the timeline (minus labels)
        label_width = 25
        bar_width = width - label_width - 10

        lines = []
        lines.append(f"Timeline: {self.trace.name}")
        lines.append(f"Total Duration: {self.trace.total_duration_ms:.2f} ms")
        lines.append("=" * width)

        # Header
        header = " " * label_width + "|"
        tick_interval = time_range / 4
        for i in range(5):
            tick_time = (min_time + i * tick_interval) * 1000
            header += f"{tick_time:>{bar_width // 4}.0f}"[:bar_width // 4]
        lines.append(header[:width])
        lines.append(" " * label_width + "+" + "-" * bar_width)

        # Steps
        for step in sorted(steps, key=lambda s: s.start_time):
            # Calculate bar position
            start_pos = int((step.start_time - min_time) / time_range * bar_width)
            end_pos = int((step.end_time - min_time) / time_range * bar_width)
            bar_len = max(1, end_pos - start_pos)

            # Create the bar
            label = step.name[:label_width - 1].ljust(label_width - 1)
            bar = " " * start_pos + "█" * bar_len
            bar = bar[:bar_width]

            # Step type indicator
            type_char = str(step.step_type)[0].upper()
            duration_str = f" ({step.duration_ms:.1f}ms)"

            lines.append(f"{label}|{bar}{duration_str}")

        lines.append("=" * width)

        # Legend
        lines.append("\nStep Types:")
        for step_type in StepType:
            type_steps = [s for s in steps if s.step_type == step_type]
            if type_steps:
                total_ms = sum(s.duration_ms for s in type_steps)
                lines.append(f"  {str(step_type):15} : {len(type_steps):3} steps, {total_ms:.2f} ms total")

        return "\n".join(lines)

    def summary_table(self) -> str:
        """
        Generate a summary table of step timings.

        Returns:
            Formatted string table
        """
        summary = self.trace.summary()

        lines = []
        lines.append(f"\n{'=' * 60}")
        lines.append(f"Trace Summary: {summary['name']}")
        lines.append(f"{'=' * 60}")
        lines.append(f"Total Steps: {summary['total_steps']}")
        lines.append(f"Total Duration: {summary['total_duration_ms']:.2f} ms")
        lines.append(f"\n{'Step Type':<20} {'Count':>8} {'Duration (ms)':>15} {'%':>8}")
        lines.append("-" * 60)

        for step_type, duration in sorted(
            summary['duration_by_type_ms'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            count = summary['count_by_type'].get(step_type, 0)
            pct = (duration / summary['total_duration_ms'] * 100) if summary['total_duration_ms'] > 0 else 0
            lines.append(f"{step_type:<20} {count:>8} {duration:>15.2f} {pct:>7.1f}%")

        lines.append("=" * 60)

        return "\n".join(lines)


def visualize_trace(trace: Trace, output_path: Optional[str] = None, show: bool = True) -> TimelineVisualizer:
    """
    Convenience function to visualize a trace.

    Args:
        trace: Trace to visualize
        output_path: Optional path to save HTML file
        show: If True, display in browser/notebook

    Returns:
        TimelineVisualizer instance
    """
    viz = TimelineVisualizer(trace)

    if output_path:
        viz.save_html(output_path)

    if show:
        try:
            viz.show()
        except Exception:
            # Fall back to ASCII if plotly display fails
            print(viz.to_ascii())
            print(viz.summary_table())

    return viz
