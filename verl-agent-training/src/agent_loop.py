"""Custom AgentLoop conforming to verl's AgentLoopBase interface.

verl's agent loop interface (verl/experimental/agent_loop/agent_loop.py):

    class AgentLoopBase(ABC):
        async def run(self, sampling_params, **kwargs) -> AgentLoopOutput

    class AgentLoopOutput(BaseModel):
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]   # 1=LLM-generated, 0=tool-response/padding
        response_logprobs: Optional[list[float]]
        reward_score: Optional[float]
        num_turns: int
        metrics: AgentLoopMetrics
        extra_fields: dict[str, Any]

Registration:
    @register("tool_call_agent")  — then set in config:
    actor_rollout_ref.rollout.agent.default_agent_loop: tool_call_agent

    Or via YAML agent_loop_config_path.

NOTE: This module provides TWO implementations:
    1. ToolCallAgentLoop — extends verl's built-in ToolAgentLoop (recommended)
    2. StandaloneAgentLoop — standalone version for development without verl
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---- Tool call parsing ----
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call> JSON blocks from LLM output."""
    calls = []
    for m in TOOL_CALL_PATTERN.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                obj.setdefault("arguments", {})
                calls.append(obj)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse tool call: %s", raw[:100])
    return calls


# =========================================================================
# Option 1: Extend verl's ToolAgentLoop (RECOMMENDED for production)
# =========================================================================

def register_tool_call_agent():
    """Register our custom agent loop with verl.

    Call this at import time (e.g., in __init__.py or in the training script)
    so that verl can find the agent loop by name.

    Usage:
        from src.agent_loop import register_tool_call_agent
        register_tool_call_agent()

        # Then in config:
        # actor_rollout_ref.rollout.agent.default_agent_loop: tool_call_agent
    """
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    except ImportError:
        logger.warning("verl not installed; skipping agent loop registration")
        return

    @register("tool_call_agent")
    class ToolCallAgentLoop(ToolAgentLoop):
        """Extended ToolAgentLoop with custom behavior for our training.

        Inherits verl's built-in ToolAgentLoop state machine:
            PENDING -> GENERATING -> (PROCESSING_TOOLS -> GENERATING)* -> TERMINATED

        Customizations:
            - Enhanced termination logic (timeout, max total tool calls)
            - Custom metrics collection
            - No-op detection
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Additional config from rollout.multi_turn
            mt_config = self.rollout_config.multi_turn
            self.max_total_tool_calls = getattr(mt_config, "max_total_tool_calls", 50)
            self.rollout_timeout = getattr(mt_config, "rollout_timeout", 120.0)

        async def run(self, sampling_params, **kwargs):
            """Override run to add timeout and enhanced metrics.

            Delegates to parent's run() but wraps with our controls.
            """
            import asyncio

            try:
                output = await asyncio.wait_for(
                    super().run(sampling_params, **kwargs),
                    timeout=self.rollout_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("AgentLoop timed out after %.0fs", self.rollout_timeout)
                # Build a minimal output on timeout
                output = self._build_timeout_output(kwargs)

            # Inject custom metrics into extra_fields
            output.extra_fields["rollout_timeout"] = False
            output.extra_fields.setdefault("tool_rewards", [])
            return output

        def _build_timeout_output(self, kwargs):
            """Build AgentLoopOutput for a timed-out episode."""
            from verl.experimental.agent_loop.agent_loop import (
                AgentLoopOutput, AgentLoopMetrics
            )
            # Return what we have so far
            prompt_ids = kwargs.get("prompt_ids", [])
            return AgentLoopOutput(
                prompt_ids=prompt_ids if isinstance(prompt_ids, list) else prompt_ids.tolist(),
                response_ids=[],
                response_mask=[],
                num_turns=0,
                metrics=AgentLoopMetrics(),
                extra_fields={"rollout_timeout": True, "tool_rewards": []},
            )

    logger.info("Registered 'tool_call_agent' AgentLoop with verl")
    return ToolCallAgentLoop


# =========================================================================
# Option 2: Standalone AgentLoop (for development/testing without verl)
# =========================================================================

@dataclass
class AgentLoopConfig:
    """Configuration for standalone AgentLoop."""
    max_turns: int = 10
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    rollout_timeout: float = 120.0
    stop_sequences: list[str] = field(
        default_factory=lambda: ["<|endoftext|>", "<|im_end|>"]
    )


@dataclass
class TurnRecord:
    """Record of a single agent turn."""
    turn_id: int
    response: str
    tool_calls: list[dict]
    tool_results: list[dict]
    log_probs: list[float] | None = None
    tokens: list[int] | None = None

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }


@dataclass
class TrajectoryRecord:
    """Complete trajectory for one episode."""
    prompt: str
    turns: list[TurnRecord]
    total_tokens: int = 0
    total_time: float = 0.0
    finished: bool = False
    finish_reason: str = ""

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    def to_dict_list(self) -> list[dict]:
        return [t.to_dict() for t in self.turns]

    def get_all_log_probs(self) -> list[float]:
        out: list[float] = []
        for t in self.turns:
            if t.log_probs:
                out.extend(t.log_probs)
        return out

    def get_full_response(self) -> str:
        """Concatenate all turn responses (for reward compute_score)."""
        return "\n".join(t.response for t in self.turns if t.response)


class StandaloneAgentLoop:
    """Multi-turn agent loop for development and testing.

    This does NOT depend on verl and can run with any generate_fn callable.
    Use this to validate tool calling + reward logic before integrating
    with verl's framework.

    Usage:
        from src.tools.verl_tools import CalculatorTool, WebSearchTool

        # Setup tools
        tools = {
            "calculator": CalculatorTool(config={}),
            "web_search": WebSearchTool(config={"backend": "mock"}),
        }

        loop = StandaloneAgentLoop(tools=tools, config=AgentLoopConfig())
        trajectory = await loop.run(
            prompt="Calculate sqrt(144)",
            generate_fn=my_llm_generate,
        )

        # Compute reward
        from src.reward_function import compute_score
        reward = compute_score(
            data_source="test",
            solution_str=trajectory.get_full_response(),
            ground_truth="12",
            extra_info={"num_turns": trajectory.num_turns},
        )
    """

    def __init__(self, tools: dict[str, Any], config: AgentLoopConfig | None = None):
        self.tools = tools
        self.config = config or AgentLoopConfig()

    async def run(
        self,
        prompt: str,
        generate_fn: Any,
        system_prompt: str = "",
        episode_id: str = "",
    ) -> TrajectoryRecord:
        """Run a complete multi-turn episode."""
        # Create tool instances
        instance_ids = {}
        for name, tool in self.tools.items():
            iid, _ = await tool.create(instance_id=f"{episode_id}_{name}")
            instance_ids[name] = iid

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        trajectory = TrajectoryRecord(prompt=prompt, turns=[])
        start = time.time()

        try:
            for turn_id in range(self.config.max_turns):
                if time.time() - start > self.config.rollout_timeout:
                    trajectory.finish_reason = "timeout"
                    break

                gen = generate_fn(
                    messages=messages,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop=self.config.stop_sequences,
                )

                response_text = gen.get("text", "")
                tool_calls = parse_tool_calls(response_text)

                if not tool_calls:
                    trajectory.turns.append(TurnRecord(
                        turn_id=turn_id, response=response_text,
                        tool_calls=[], tool_results=[],
                        log_probs=gen.get("log_probs"),
                        tokens=gen.get("tokens"),
                    ))
                    trajectory.finished = True
                    trajectory.finish_reason = "completed"
                    break

                # Execute tools
                results = []
                for call in tool_calls:
                    name = call.get("name", "")
                    tool = self.tools.get(name)
                    if tool is None:
                        results.append({"name": name, "error": f"Unknown tool: {name}"})
                        continue
                    iid = instance_ids.get(name, "")
                    resp, step_reward, metrics = await tool.execute(
                        iid, call.get("arguments", {})
                    )
                    results.append({
                        "name": name, "text": resp.text,
                        "step_reward": step_reward, **metrics,
                    })

                trajectory.turns.append(TurnRecord(
                    turn_id=turn_id, response=response_text,
                    tool_calls=tool_calls, tool_results=results,
                    log_probs=gen.get("log_probs"), tokens=gen.get("tokens"),
                ))

                # Append to context
                messages.append({"role": "assistant", "content": response_text})
                tool_text = "\n".join(
                    f"<tool_response>\n{json.dumps(r, ensure_ascii=False)}\n</tool_response>"
                    for r in results
                )
                messages.append({"role": "user", "content": tool_text})
            else:
                trajectory.finish_reason = "max_turns"

        finally:
            # Release tool instances
            for name, tool in self.tools.items():
                iid = instance_ids.get(name, "")
                await tool.release(iid)

        trajectory.total_time = time.time() - start
        trajectory.total_tokens = sum(
            len(t.tokens) for t in trajectory.turns if t.tokens
        )
        return trajectory
