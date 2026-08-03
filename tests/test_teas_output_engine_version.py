import json

import pytest

from agent_cap.agents.teas_output import write_teas_outputs


@pytest.mark.parametrize(
    ("dataset", "engine", "engine_version"),
    [
        ("mcp-atlas", "sglang", "0.5.12"),
        ("swe-bench-lite", "vllm", "0.21.0"),
    ],
)
def test_engine_version_is_written_to_metrics_and_metadata(
    tmp_path,
    monkeypatch,
    dataset,
    engine,
    engine_version,
):
    monkeypatch.setenv("TEAS_ENGINE", engine)
    monkeypatch.setenv("TEAS_ENGINE_VERSION", engine_version)
    monkeypatch.setenv("TEAS_PRECISION", "bfloat16")

    write_teas_outputs(
        out_dir=tmp_path,
        rows=[
            {
                "task_id": "example",
                "total_usage": {},
                "eval_passed": True,
            }
        ],
        dataset=dataset,
        wall_time_s=1.0,
        timestamp="20260730_120000",
    )

    suffix = f"{dataset}_20260730_120000"
    metrics = json.loads((tmp_path / f"metrics_{suffix}.json").read_text())
    metadata = json.loads((tmp_path / f"metadata_{suffix}.json").read_text())
    version_key = f"{engine}_version"

    assert metrics["hardware"][version_key] == engine_version
    assert metadata["system_environment"][version_key] == engine_version
