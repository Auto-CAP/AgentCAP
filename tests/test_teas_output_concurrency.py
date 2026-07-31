import json

from agent_cap.agents.metrics import aggregate_agent_metrics
from agent_cap.agents.teas_output import write_teas_outputs


def test_swebench_metadata_records_requested_and_observed_concurrency(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TEAS_ENGINE", "vllm")
    monkeypatch.setenv("TEAS_CONCURRENCY", "4")
    monkeypatch.setenv("TEAS_OBSERVED_MAX_CONCURRENCY", "4")

    write_teas_outputs(
        tmp_path,
        [{"task_id": "example", "e2e_latency_s": 1.0}],
        dataset="swe-bench-lite",
        wall_time_s=1.0,
        timestamp="20260731_000000",
    )

    metadata = json.loads(
        (tmp_path / "metadata_swe-bench-lite_20260731_000000.json").read_text()
    )
    system_environment = metadata["system_environment"]
    assert system_environment["concurrency"] == 4
    assert system_environment["observed_max_concurrency"] == 4


def test_generic_metrics_does_not_report_machine_visible_gpu_count():
    metrics = aggregate_agent_metrics(
        [],
        wall_time_s=1.0,
        hardware_info={"gpu_type": "NVIDIA B300", "num_gpus": 2},
    )
    assert metrics["hardware"]["gpu_type"] == "NVIDIA B300"
    assert "num_gpus" not in metrics["hardware"]
