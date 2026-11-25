"""Run agent harnesses on Harbor (https://github.com/harbor-framework/harbor).

Harbor owns task environments, trial scheduling and grading. Every harness is a
Harbor agent and takes the same model flags (``--model``, ``--base-url``,
``--api-key-name``, ``--api-format``, ``--effort``):

* ``agent_cap.harbor:CAPAgent`` - an AgentCAP strategy (single, plan-execute,
  supervisor, sequential) on the host; tool calls run in the task container.
* ``agent_cap.harbor:DSHAgent`` - DeepSeek Harness (single agent or Agent
  Team) inside the task container.
* Harbor's own agents (``codex``, ``claude-code``, ``openhands``,
  ``mini-swe-agent``, ...), unchanged, or metered:
  ``agent_cap.harbor.metered:<name>`` records every model request through a
  host-side proxy (``agent_cap.harbor.meter``).

Supporting modules: ``tools`` (tool providers over ``environment.exec``),
``adapters`` (benchmark -> Harbor task directories), ``jobspec`` / ``cli``
(job configs and ``python -m agent_cap.harbor``), ``export_teas`` (Harbor job
dir -> TEAS files, for any harness).

Install with ``pip install 'agent-cap[harbor]'``. Importing this package does not
import Harbor; the agent classes load it on first access.
"""

RESULT_FILENAME = "capagent_result.json"
METER_FILENAME = "meter_result.json"

_LAZY = {
    "CAPAgent": "agent_cap.harbor.agent",
    "DSHAgent": "agent_cap.harbor.dsh",
}


def __getattr__(name):  # `--agent agent_cap.harbor:CAPAgent` without importing Harbor eagerly
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
