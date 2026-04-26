"""Sandboxed Python code executor for the Agentic RL environment."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from typing import Any

from .base import BaseTool

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):
    """Execute Python code in a sandboxed environment.

    Supports two backends:
    - "subprocess": local subprocess with resource limits (for dev/testing)
    - "docker": Docker container isolation (for production training)
    """

    name = "code_executor"
    description = (
        "Execute Python code and return the output. "
        "The code should print its result to stdout."
    )
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
        },
    }

    def __init__(
        self,
        backend: str = "subprocess",
        timeout: float = 30.0,
        max_memory_mb: int = 512,
        docker_image: str = "verl-sandbox:latest",
    ):
        self.backend = backend
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.docker_image = docker_image

    def execute(self, code: str, **kwargs: Any) -> dict:
        if self.backend == "subprocess":
            return self._execute_subprocess(code)
        elif self.backend == "docker":
            return self._execute_docker(code)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _execute_subprocess(self, code: str) -> dict:
        """Execute code in a local subprocess with resource limits."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin"),
                    "HOME": "/tmp",
                },
            )
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s",
                "returncode": -1,
            }
        finally:
            os.unlink(tmp_path)

    def _execute_docker(self, code: str) -> dict:
        """Execute code inside a Docker container for production isolation."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = [
                "docker", "run", "--rm",
                "--network=none",
                f"--memory={self.max_memory_mb}m",
                "--cpus=1",
                "--pids-limit=64",
                "--read-only",
                "--tmpfs=/tmp:size=64m",
                "-v", f"{tmp_path}:/code/script.py:ro",
                self.docker_image,
                "python3", "/code/script.py",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # extra buffer for container startup
            )
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Docker execution timed out after {self.timeout}s",
                "returncode": -1,
            }
        finally:
            os.unlink(tmp_path)
