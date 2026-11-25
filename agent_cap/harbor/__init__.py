"""Run AgentCAP configurations on Harbor (https://github.com/harbor-framework/harbor).

Harbor owns task environments, trial scheduling and grading; AgentCAP keeps the
team strategies, per-request metering and the TEAS output files.

* ``agent_cap.harbor.agent.CAPAgent`` - Harbor agent that runs an AgentCAP
  strategy on the host and executes tool calls in the task container.
* ``agent_cap.harbor.tools`` - tool providers over ``environment.exec``.
* ``agent_cap.harbor.adapters`` - benchmark -> Harbor task directories
  (imo-answerbench, nl2repo).
* ``agent_cap.harbor.jobspec`` / ``cli`` - job configs and the
  ``python -m agent_cap.harbor`` entry point.
* ``agent_cap.harbor.export_teas`` - Harbor job dir -> TEAS files.

Install with ``pip install 'agent-cap[harbor]'``. Importing this package does not
import Harbor; only ``agent_cap.harbor.agent`` needs it.
"""

RESULT_FILENAME = "capagent_result.json"
