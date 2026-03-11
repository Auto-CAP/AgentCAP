import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import time
import urllib.error
import urllib.request


@dataclass
class ServerConfig:
    engine: str
    model_id: str
    quantization: str
    tp: int = 1
    port: int = 30000
    python_path: str = "python"
    env_vars: Dict[str, str] = field(default_factory=dict)
    extra_flags: List[str] = field(default_factory=list)
    extra_args: Dict[str, str] = field(default_factory=dict)


class ModelServerManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.command: List[str] = []
        self.stdout_log_path: Optional[Path] = None
        self.stderr_log_path: Optional[Path] = None
        self._stdout_handle = None
        self._stderr_handle = None

    def _build_command(self) -> List[str]:
        if self.config.engine == "sglang":
            cmd = [
                self.config.python_path,
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.config.model_id,
                "--tp",
                str(self.config.tp),
                "--port",
                str(self.config.port),
            ]
        elif self.config.engine == "vllm":
            cmd = [
                self.config.python_path,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.config.model_id,
                "--tensor-parallel-size",
                str(self.config.tp),
                "--port",
                str(self.config.port),
            ]
        else:
            raise ValueError(f"Unsupported engine: {self.config.engine}")

        if self.config.quantization != "fp16":
            cmd.extend(["--quantization", self.config.quantization])

        cmd.append("--trust-remote-code")
        cmd.extend(self.config.extra_flags)

        for key, value in self.config.extra_args.items():
            cmd.extend([f"--{key}", value])
        return cmd

    def launch(self) -> None:
        if self.process and self.is_alive():
            raise RuntimeError("Server is already running")

        self.command = self._build_command()
        stamp = int(time.time())
        self.stdout_log_path = Path.cwd() / f"server_{self.config.engine}_{self.config.port}_{stamp}.out.log"
        self.stderr_log_path = Path.cwd() / f"server_{self.config.engine}_{self.config.port}_{stamp}.err.log"
        self._stdout_handle = self.stdout_log_path.open("w", encoding="utf-8")
        self._stderr_handle = self.stderr_log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(self.config.env_vars)

        self.process = subprocess.Popen(
            self.command,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            start_new_session=True,
            env=env,
        )

    def wait_until_ready(self, timeout: float = 600) -> bool:
        if not self.process:
            raise RuntimeError("Server has not been launched")

        url = f"http://localhost:{self.config.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1.0)
        return False

    def shutdown(self) -> None:
        if not self.process:
            return

        if self.is_alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

        if self._stdout_handle:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle:
            self._stderr_handle.close()
            self._stderr_handle = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self) -> "ModelServerManager":
        self.launch()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
