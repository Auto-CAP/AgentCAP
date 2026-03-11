import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunResult:
    id: str
    experiment_name: str
    model_id: str
    model_params_b: float
    model_arch: str
    serving_engine: str
    quantization: str
    tensor_parallel: int
    gpu_type: str

    skill_subset: str
    num_retries: int
    temperature: float
    agent_mode: str
    task_id: str
    task_name: str
    repetition: int
    task_success: Optional[bool] = None
    quality_score: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    latency_e2e_ms: float = 0.0
    avg_gpu_util_pct: float = 0.0
    avg_power_w: float = 0.0
    output_text: str = ""
    trajectory_log: str = ""
    combination_strategy: str = ""
    combination_detail: str = ""
    tool_call_count: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ResultStore:
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                model_id TEXT, model_params_b REAL, model_arch TEXT,
                serving_engine TEXT, quantization TEXT,
                tensor_parallel INTEGER, gpu_type TEXT,
                skill_subset TEXT, num_retries INTEGER,
                temperature REAL, agent_mode TEXT,
                task_id TEXT, task_name TEXT, repetition INTEGER,
                task_success BOOLEAN, quality_score REAL,
                input_tokens INTEGER, output_tokens INTEGER,
                gpu_seconds REAL, peak_vram_mb REAL,
                latency_e2e_ms REAL,
                avg_gpu_util_pct REAL, avg_power_w REAL,
                output_text TEXT, trajectory_log TEXT,
                combination_strategy TEXT, combination_detail TEXT,
                tool_call_count INTEGER,
                started_at TEXT, completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_name);
            CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, quantization);
            """
        )

        for col, col_type in [
            ("combination_strategy", "TEXT"),
            ("combination_detail", "TEXT"),
            ("tool_call_count", "INTEGER"),
        ]:
            try:
                default_value = "0" if col == "tool_call_count" else "''"
                self._conn.execute(
                    f"ALTER TABLE runs ADD COLUMN {col} {col_type} DEFAULT {default_value}"
                )
            except sqlite3.OperationalError:
                pass

        self._conn.commit()

    def save_run(self, run: RunResult) -> None:
        data = asdict(run)
        if not data.get("id"):
            data["id"] = str(uuid.uuid4())

        columns = list(data.keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT OR REPLACE INTO runs ({', '.join(columns)}) VALUES ({placeholders})"
        values = [data[col] for col in columns]
        self._conn.execute(sql, values)
        self._conn.commit()

    def get_runs(
        self,
        experiment_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[RunResult]:
        clauses: List[str] = []
        params: List[Any] = []

        if experiment_name is not None:
            clauses.append("experiment_name = ?")
            params.append(experiment_name)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)

        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY experiment_name, model_id, task_id, repetition"

        rows = self._conn.execute(query, params).fetchall()
        return [RunResult(**dict(row)) for row in rows]

    def get_config_summary(self) -> List[Dict[str, Any]]:
        query = """
            SELECT
                experiment_name,
                model_id,
                quantization,
                skill_subset,
                num_retries,
                temperature,
                agent_mode,
                COUNT(*) AS run_count,
                AVG(CAST(task_success AS REAL)) AS success_rate,
                AVG(quality_score) AS avg_quality_score,
                AVG(input_tokens) AS avg_input_tokens,
                AVG(output_tokens) AS avg_output_tokens,
                AVG(latency_e2e_ms) AS avg_latency_ms,
                AVG(gpu_seconds) AS avg_gpu_seconds,
                AVG(peak_vram_mb) AS avg_peak_vram_mb,
                AVG(avg_gpu_util_pct) AS avg_gpu_util_pct,
                AVG(avg_power_w) AS avg_power_w
            FROM runs
            GROUP BY
                experiment_name,
                model_id,
                quantization,
                skill_subset,
                num_retries,
                temperature,
                agent_mode
            ORDER BY experiment_name, model_id, quantization
        """
        rows = self._conn.execute(query).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
