"""Tool sandbox environment for Agentic RL training.

Manages tool registration, execution, environment state, and cleanup.
Implements the ToolEnvironment class that wraps the ToolRegistry with
per-episode state management and safety controls.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .tools.base import BaseTool, ToolRegistry, ToolResult
from .tools.web_search import WebSearchTool
from .tools.calculator import CalculatorTool
from .tools.code_executor import CodeExecutorTool
from .tools.database import DatabaseQueryTool

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for the tool environment."""

    tool_timeout: float = 30.0
    max_tool_calls_per_turn: int = 5
    max_total_tool_calls: int = 50
    code_executor_backend: str = "subprocess"  # "subprocess" or "docker"
    docker_image: str = "verl-sandbox:latest"
    search_backend: str = "mock"  # "mock" or "serpapi"
    search_api_key: str | None = None
    search_corpus: dict | None = None
    db_path: str = ":memory:"
    db_schema: str | None = None
    cleanup_on_reset: bool = True


@dataclass
class EnvironmentState:
    """Tracks per-episode environment state."""

    episode_id: str = ""
    total_tool_calls: int = 0
    tool_call_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0


class ToolEnvironment:
    """Tool execution environment for Agentic RL.

    Manages tool registration, execution, and per-episode state.
    Ensures environment isolation between episodes (cleanup).
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.registry = ToolRegistry()
        self.state = EnvironmentState()
        self._setup_default_tools()

    def _setup_default_tools(self) -> None:
        """Register built-in tools."""
        self.registry.register(
            WebSearchTool(
                backend=self.config.search_backend,
                api_key=self.config.search_api_key,
                corpus=self.config.search_corpus or {},
            )
        )
        self.registry.register(CalculatorTool())
        self.registry.register(
            CodeExecutorTool(
                backend=self.config.code_executor_backend,
                timeout=self.config.tool_timeout,
                docker_image=self.config.docker_image,
            )
        )
        db_tool = DatabaseQueryTool(db_path=self.config.db_path)
        if self.config.db_schema:
            db_tool.load_schema(self.config.db_schema)
        self.registry.register(db_tool)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a custom tool."""
        self.registry.register(tool)

    def reset(self, episode_id: str = "") -> None:
        """Reset environment state for a new episode.

        Clears tool call history and resets counters.
        In production, this would also clean up sandbox artifacts
        (temp files, DB state, etc.) to prevent environment leakage.
        """
        self.state = EnvironmentState(
            episode_id=episode_id,
            start_time=time.time(),
        )
        if self.config.cleanup_on_reset:
            self._cleanup_sandbox()
        logger.debug("Environment reset for episode %s", episode_id)

    def _cleanup_sandbox(self) -> None:
        """Clean up sandbox artifacts from previous episode.

        Prevents environment leakage (ref: ROLL anti-leakage strategy).
        """
        # Reset database state
        db_tool = self.registry.get("database_query")
        if isinstance(db_tool, DatabaseQueryTool) and self.config.db_schema:
            db_tool.close()
            db_tool.db_path = self.config.db_path
            db_tool.load_schema(self.config.db_schema)

    def execute_tool_calls(
        self, tool_calls: list[dict]
    ) -> list[ToolResult]:
        """Execute a batch of tool calls for one turn.

        Args:
            tool_calls: List of dicts with 'name' and 'arguments' keys.

        Returns:
            List of ToolResult objects.
        """
        # Enforce per-turn limit
        calls_to_execute = tool_calls[: self.config.max_tool_calls_per_turn]
        if len(tool_calls) > self.config.max_tool_calls_per_turn:
            logger.warning(
                "Truncated tool calls from %d to %d (per-turn limit)",
                len(tool_calls),
                self.config.max_tool_calls_per_turn,
            )

        results = []
        for call in calls_to_execute:
            # Check total limit
            if self.state.total_tool_calls >= self.config.max_total_tool_calls:
                results.append(
                    ToolResult(
                        name=call.get("name", "unknown"),
                        success=False,
                        error="Maximum total tool calls reached",
                    )
                )
                continue

            name = call.get("name", "")
            arguments = call.get("arguments", {})

            result = self.registry.execute(
                name=name,
                timeout=self.config.tool_timeout,
                **arguments,
            )
            results.append(result)

            self.state.total_tool_calls += 1
            self.state.tool_call_history.append(
                {
                    "call": call,
                    "result": result.to_dict(),
                    "timestamp": time.time(),
                }
            )

        return results

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the LLM system prompt."""
        return self.registry.build_system_prompt_section()

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas as a list of dicts (for JSON injection)."""
        return self.registry.get_all_schemas()

    def get_episode_stats(self) -> dict:
        """Get statistics for the current episode."""
        return {
            "episode_id": self.state.episode_id,
            "total_tool_calls": self.state.total_tool_calls,
            "elapsed_time": time.time() - self.state.start_time,
            "tools_used": list(
                set(h["call"]["name"] for h in self.state.tool_call_history)
            ),
        }

    def format_tool_results(self, results: list[ToolResult]) -> str:
        """Format tool results for injection into the LLM context."""
        parts = []
        for r in results:
            parts.append(f"<tool_response>\n{r.to_context_string()}\n</tool_response>")
        return "\n".join(parts)
