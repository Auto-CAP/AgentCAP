"""Test doubles for agent_cap.harbor: a local-process environment and a Docker one.

Both implement the subset of Harbor's ``BaseEnvironment`` that CAPAgent uses:
``exec(command, cwd=, timeout_sec=)`` returning an object with ``stdout``,
``stderr`` and ``return_code``, and ``upload_file(source, target)``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecResult:
    stdout: Optional[str]
    stderr: Optional[str]
    return_code: int


class LocalEnvironment:
    """Runs commands on this machine with paths rebased under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.commands = []

    def _map(self, path: str) -> str:
        return str(self.root / path.lstrip("/")) if path.startswith("/") else path

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        workdir = Path(self._map(cwd or "/"))
        workdir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return ExecResult(out.decode(), err.decode(), proc.returncode)

    async def upload_file(self, source_path, target_path):
        target = Path(self._map(str(target_path)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target)


class DockerEnvironment:
    """A running container driven with ``docker exec`` / ``docker cp``."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.container: Optional[str] = None

    def start(self) -> "DockerEnvironment":
        self.container = subprocess.run(
            ["docker", "run", "-d", "--rm", self.image, "sleep", "infinity"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return self

    def stop(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        args = ["docker", "exec"]
        if cwd:
            args += ["-w", cwd]
        args += [self.container, "bash", "-c", command]
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return ExecResult(out.decode(), err.decode(), proc.returncode)

    async def upload_file(self, source_path, target_path):
        proc = await asyncio.create_subprocess_exec(
            "docker", "cp", str(source_path), f"{self.container}:{target_path}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker cp failed for {target_path}")


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
