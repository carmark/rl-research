"""Tool implementations conforming to verl's BaseTool interface.

verl's tool interface (verl/tools/base_tool.py):

    class BaseTool:
        async def create(instance_id, **kwargs) -> (instance_id, ToolResponse)
        async def execute(instance_id, parameters, **kwargs) -> (ToolResponse, step_reward, metrics)
        async def calc_reward(instance_id, **kwargs) -> float
        async def release(instance_id, **kwargs) -> None

    class ToolResponse(BaseModel):
        text: str | None = None
        image: list[Any] | None = None
        video: list[Any] | None = None

Tools are registered via YAML config (tool_config.yaml), not Python decorators.

This module provides verl-compatible tool implementations that can be referenced
in tool_config.yaml via class_name.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight ToolResponse (mirrors verl's pydantic model for standalone use)
# ---------------------------------------------------------------------------
try:
    from verl.tools.base_tool import BaseTool as VerlBaseTool, ToolResponse
except ImportError:
    # Fallback for development without verl installed
    from pydantic import BaseModel

    class ToolResponse(BaseModel):  # type: ignore[no-redef]
        text: str | None = None
        image: list[Any] | None = None
        video: list[Any] | None = None

    class VerlBaseTool:  # type: ignore[no-redef]
        """Stub when verl is not installed."""
        def __init__(self, config: dict, tool_schema=None):
            self.config = config
            self.tool_schema = tool_schema
            self.name = ""


# =========================================================================
# 1. Calculator Tool
# =========================================================================
_SAFE_MATH = {
    k: getattr(math, k)
    for k in [
        "sqrt", "pow", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "ceil", "floor", "fabs", "factorial", "gcd",
        "pi", "e", "inf",
    ]
}
_SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "int": int, "float": float})


class CalculatorTool(VerlBaseTool):
    """Safe mathematical expression evaluator.

    Registered in tool_config.yaml as:
        class_name: src.tools.verl_tools.CalculatorTool
    """

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema)
        self.name = "calculator"
        # Track per-instance state for reward computation
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id=None, **kwargs):
        iid = instance_id or f"calc_{id(self)}"
        self._instances[iid] = {"calls": 0, "successes": 0}
        return iid, ToolResponse(text="Calculator ready.")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        expression = parameters.get("expression", "")
        state = self._instances.get(instance_id, {"calls": 0, "successes": 0})
        state["calls"] += 1

        # Safety check
        forbidden = ["import", "__", "exec", "eval", "open", "os.", "sys."]
        if any(p in expression.lower() for p in forbidden):
            return (
                ToolResponse(text=f"Error: expression contains forbidden pattern"),
                -0.05,  # step penalty for bad input
                {"error": "forbidden_pattern"},
            )

        try:
            result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
            if not isinstance(result, (int, float)):
                raise ValueError(f"Non-numeric result: {type(result)}")
            state["successes"] += 1
            return (
                ToolResponse(text=str(result)),
                0.05,   # small step reward for successful tool use
                {"result": result},
            )
        except Exception as e:
            return (
                ToolResponse(text=f"Error: {e}"),
                0.0,
                {"error": str(e)},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instances.get(instance_id, {})
        if state.get("calls", 0) == 0:
            return 0.0
        return state.get("successes", 0) / state["calls"] * 0.1

    async def release(self, instance_id: str, **kwargs):
        self._instances.pop(instance_id, None)


# =========================================================================
# 2. Code Executor Tool
# =========================================================================
class CodeExecutorTool(VerlBaseTool):
    """Sandboxed Python code executor.

    Supports two backends:
    - subprocess: local process with resource limits (dev/testing)
    - docker: container isolation (production)
    """

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema)
        self.name = "code_executor"
        self.backend = config.get("backend", "subprocess")
        self.timeout = config.get("timeout", 30.0)
        self.docker_image = config.get("docker_image", "verl-sandbox:latest")
        self.max_memory_mb = config.get("max_memory_mb", 512)
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id=None, **kwargs):
        iid = instance_id or f"code_{id(self)}"
        self._instances[iid] = {"calls": 0, "successes": 0}
        return iid, ToolResponse(text="Code executor ready.")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        code = parameters.get("code", "")
        state = self._instances.get(instance_id, {"calls": 0, "successes": 0})
        state["calls"] += 1

        if self.backend == "docker":
            result = self._run_docker(code)
        else:
            result = self._run_subprocess(code)

        if result["returncode"] == 0:
            state["successes"] += 1
            return (
                ToolResponse(text=result["stdout"] or "(no output)"),
                0.05,
                {"returncode": 0},
            )
        else:
            error_text = result["stderr"] or f"Exit code: {result['returncode']}"
            return (
                ToolResponse(text=f"Error: {error_text}"),
                0.0,
                {"returncode": result["returncode"], "stderr": result["stderr"][:200]},
            )

    def _run_subprocess(self, code: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            r = subprocess.run(
                ["python3", tmp], capture_output=True, text=True,
                timeout=self.timeout,
                env={"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": "/tmp"},
            )
            return {"stdout": r.stdout.strip(), "stderr": r.stderr.strip(),
                    "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timeout after {self.timeout}s",
                    "returncode": -1}
        finally:
            os.unlink(tmp)

    def _run_docker(self, code: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            cmd = [
                "docker", "run", "--rm", "--network=none",
                f"--memory={self.max_memory_mb}m", "--cpus=1",
                "--pids-limit=64", "--read-only", "--tmpfs=/tmp:size=64m",
                "-v", f"{tmp}:/code/script.py:ro",
                self.docker_image, "python3", "/code/script.py",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout + 10)
            return {"stdout": r.stdout.strip(), "stderr": r.stderr.strip(),
                    "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Docker timeout", "returncode": -1}
        finally:
            os.unlink(tmp)

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instances.get(instance_id, {})
        if state.get("calls", 0) == 0:
            return 0.0
        return state.get("successes", 0) / state["calls"] * 0.1

    async def release(self, instance_id: str, **kwargs):
        self._instances.pop(instance_id, None)


# =========================================================================
# 3. Web Search Tool
# =========================================================================
class WebSearchTool(VerlBaseTool):
    """Web search tool with mock and real API backends."""

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema)
        self.name = "web_search"
        self.backend = config.get("backend", "mock")
        self.api_key = config.get("api_key")
        self.corpus = config.get("corpus", {})
        self.max_results = config.get("max_results", 5)
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id=None, **kwargs):
        iid = instance_id or f"search_{id(self)}"
        self._instances[iid] = {"calls": 0}
        return iid, ToolResponse(text="Web search ready.")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        query = parameters.get("query", "")
        state = self._instances.get(instance_id, {"calls": 0})
        state["calls"] += 1

        if self.backend == "mock":
            results = self._mock_search(query)
        else:
            results = self._api_search(query)

        text = json.dumps(results, ensure_ascii=False, indent=2)
        return (
            ToolResponse(text=text),
            0.05,   # step reward for using search
            {"num_results": len(results)},
        )

    def _mock_search(self, query: str) -> list[dict]:
        results = []
        q_lower = query.lower()
        for key, entries in self.corpus.items():
            if any(w in key.lower() for w in q_lower.split()):
                results.extend(entries)
        if not results:
            results = [{
                "title": f"Result for: {query}",
                "snippet": f"Mock result for '{query}'.",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            }]
        return results[:self.max_results]

    def _api_search(self, query: str) -> list[dict]:
        try:
            import requests
            resp = requests.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": self.api_key, "num": self.max_results},
                timeout=10,
            )
            resp.raise_for_status()
            return [
                {"title": r.get("title", ""), "snippet": r.get("snippet", ""),
                 "url": r.get("link", "")}
                for r in resp.json().get("organic_results", [])
            ][:self.max_results]
        except Exception as e:
            return [{"error": str(e)}]

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0  # search itself is neutral; reward comes from using results

    async def release(self, instance_id: str, **kwargs):
        self._instances.pop(instance_id, None)


# =========================================================================
# 4. Database Query Tool
# =========================================================================
class DatabaseQueryTool(VerlBaseTool):
    """Read-only SQL query tool using SQLite."""

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema)
        self.name = "database_query"
        self.db_path = config.get("db_path", ":memory:")
        self.schema_sql = config.get("schema_sql")
        self.max_rows = config.get("max_rows", 100)
        self._instances: dict[str, sqlite3.Connection] = {}

    async def create(self, instance_id=None, **kwargs):
        iid = instance_id or f"db_{id(self)}"
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self.schema_sql:
            conn.executescript(self.schema_sql)
            conn.commit()
        self._instances[iid] = conn
        return iid, ToolResponse(text="Database ready.")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        sql = parameters.get("sql", "")
        conn = self._instances.get(instance_id)
        if conn is None:
            return ToolResponse(text="Error: no database connection"), -0.1, {}

        # Safety: SELECT only
        sql_upper = sql.strip().upper()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return (
                ToolResponse(text="Error: only SELECT queries are allowed"),
                -0.05,
                {"error": "non_select"},
            )
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for kw in dangerous:
            if re.search(rf"\b{kw}\b", sql_upper):
                return (
                    ToolResponse(text=f"Error: forbidden SQL keyword: {kw}"),
                    -0.05,
                    {"error": f"forbidden_{kw}"},
                )

        try:
            cursor = conn.execute(sql)
            cols = [d[0] for d in cursor.description]
            rows = [dict(zip(cols, r)) for r in cursor.fetchmany(self.max_rows)]
            text = json.dumps(rows, ensure_ascii=False, indent=2)
            return ToolResponse(text=text), 0.05, {"num_rows": len(rows)}
        except Exception as e:
            return ToolResponse(text=f"SQL Error: {e}"), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        conn = self._instances.pop(instance_id, None)
        if conn:
            conn.close()
