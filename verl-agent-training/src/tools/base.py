"""Base tool interface and registry for the Agentic RL tool-calling environment."""

from __future__ import annotations

import abc
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        d = {"name": self.name, "success": self.success}
        if self.success:
            d["result"] = self.result
        else:
            d["error"] = self.error
        d["execution_time"] = round(self.execution_time, 3)
        return d

    def to_context_string(self) -> str:
        """Format result for injection into the LLM context."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseTool(abc.ABC):
    """Abstract base class for all tools.

    Subclasses must define ``name``, ``description``, ``parameters`` and
    implement the ``execute`` method.
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def get_schema(self) -> dict:
        """Return the JSON-Schema tool description for the system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments and return a result."""
        ...

    def safe_execute(self, timeout: float = 30.0, **kwargs: Any) -> ToolResult:
        """Execute with error handling and timing."""
        start = time.time()
        try:
            result = self.execute(**kwargs)
            elapsed = time.time() - start
            if elapsed > timeout:
                return ToolResult(
                    name=self.name,
                    success=False,
                    error=f"Execution exceeded timeout ({timeout}s)",
                    execution_time=elapsed,
                )
            return ToolResult(
                name=self.name, success=True, result=result, execution_time=elapsed
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.warning("Tool %s failed: %s", self.name, e)
            return ToolResult(
                name=self.name,
                success=False,
                error=str(e),
                execution_time=elapsed,
            )


@dataclass
class ToolRegistry:
    """Registry of available tools."""

    _tools: dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def execute(self, name: str, timeout: float = 30.0, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )
        return tool.safe_execute(timeout=timeout, **kwargs)

    def build_system_prompt_section(self) -> str:
        """Generate the tool description section for the LLM system prompt."""
        schemas = self.get_all_schemas()
        lines = ["You have access to the following tools:\n"]
        for s in schemas:
            lines.append(f"### {s['name']}")
            lines.append(s["description"])
            lines.append(f"Parameters: {json.dumps(s['parameters'], indent=2)}")
            lines.append("")
        lines.append(
            "To use a tool, output a <tool_call> block with a JSON object "
            'containing "name" and "arguments".'
        )
        return "\n".join(lines)
