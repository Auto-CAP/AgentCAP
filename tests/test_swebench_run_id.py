"""Regression test: swebench evaluator must use a per-run run_id.

A shared/constant run_id makes swebench.harness report "already run,
skipping" on every run after the first and reuse the first run's results
(identical accuracy across different models/engines).
"""
from pathlib import Path

from agent_cap.agents.evaluators_swebench import unique_run_id


def test_run_id_differs_per_run_dir():
    base = "agentcap_unified"
    a = unique_run_id(base, Path(
        "/x/agentic/amd/sglang/gpt-oss-120b/swe-bench-lite/mi355xx1/batch-size-default/260723-1110"))
    b = unique_run_id(base, Path(
        "/x/agentic/amd/vllm/gpt-oss-120b/swe-bench-lite/mi355xx1/batch-size-default/260723-1211"))
    c = unique_run_id(base, Path(
        "/x/agentic/amd/sglang/deepseek-v3.2/swe-bench-lite/mi355xx8/batch-size-default/260723-0548"))
    assert a != b != c and a != c          # distinct runs -> distinct run_ids
    assert base in a                        # keeps the base prefix
    assert all(ch.isalnum() or ch in "_.-" for ch in a)  # filesystem-safe


def test_run_id_stable_for_same_run_dir():
    p = Path("/x/agentic/amd/vllm/gpt-oss-120b/mcp-atlas/mi355xx1/batch-size-default/260723-1225")
    assert unique_run_id("agentcap_unified", p) == unique_run_id("agentcap_unified", p)
