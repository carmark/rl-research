"""AgentLoop for multi-turn tool-calling RL rollout.

Encapsulates the interaction loop between the LLM (via vLLM server)
and the tool environment.  Each episode consists of multiple turns
where the LLM generates a response, tool calls are parsed and executed,
and the results are appended to the context for the next turn.
"""

from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .tool_env import ToolEnvironment

logger = logging.getLogger(__name__)

# Regex to extract <tool_call>...</tool_call> blocks
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


@dataclass
class AgentLoopConfig:
    """Configuration for the AgentLoop."""

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
    """Record of a single turn in the agent loop."""

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
        """Convert to the trajectory format expected by the reward function."""
        return [t.to_dict() for t in self.turns]

    def get_all_log_probs(self) -> list[float]:
        """Concatenate log-probs from all turns."""
        probs = []
        for t in self.turns:
            if t.log_probs:
                probs.extend(t.log_probs)
        return probs


def parse_tool_calls(response: str) -> list[dict]:
    """Parse tool calls from LLM output.

    Expected format:
        <tool_call>
        {"name": "tool_name", "arguments": {"arg1": "val1"}}
        </tool_call>

    Returns a list of parsed tool call dicts.
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(response):
        raw = match.group(1).strip()
        try:
            call = json.loads(raw)
            if isinstance(call, dict) and "name" in call:
                if "arguments" not in call:
                    call["arguments"] = {}
                calls.append(call)
            else:
                logger.warning("Malformed tool call (missing 'name'): %s", raw[:100])
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool call JSON: %s — %s", raw[:100], e)
    return calls


def has_tool_calls(response: str) -> bool:
    """Check if the response contains any tool call blocks."""
    return bool(TOOL_CALL_PATTERN.search(response))


def strip_tool_calls(response: str) -> str:
    """Remove tool call blocks from the response text."""
    return TOOL_CALL_PATTERN.sub("", response).strip()


class AgentLoop:
    """Multi-turn agent interaction loop for RL rollout.

    This class orchestrates the interaction between the LLM inference
    engine and the tool environment.  It is designed to be used with
    verl's AgentLoop / Server mode, but can also run standalone for
    testing.

    Usage:
        env = ToolEnvironment(config)
        loop = AgentLoop(env=env, config=AgentLoopConfig())

        # With a callable generate function (e.g. vLLM client)
        trajectory = loop.run(
            prompt="What is the population of France?",
            system_prompt="You are a helpful assistant with tool access.",
            generate_fn=my_vllm_generate,
        )
    """

    def __init__(
        self,
        env: ToolEnvironment,
        config: AgentLoopConfig | None = None,
    ):
        self.env = env
        self.config = config or AgentLoopConfig()

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        generate_fn: Any = None,
        episode_id: str = "",
    ) -> TrajectoryRecord:
        """Run a complete multi-turn agent episode.

        Args:
            prompt: The user task prompt.
            system_prompt: Optional system prompt (tool descriptions
                will be appended automatically).
            generate_fn: Callable that takes messages (list[dict]) and
                generation params, returns a dict with 'text', 'log_probs',
                and 'tokens'.
            episode_id: Unique episode identifier.

        Returns:
            TrajectoryRecord with all turns and metadata.
        """
        if generate_fn is None:
            raise ValueError("generate_fn is required")

        # Reset environment
        self.env.reset(episode_id=episode_id)

        # Build initial messages
        tool_desc = self.env.get_tool_descriptions()
        if system_prompt:
            full_system = f"{system_prompt}\n\n{tool_desc}"
        else:
            full_system = tool_desc

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        trajectory = TrajectoryRecord(prompt=prompt, turns=[])
        start_time = time.time()

        for turn_id in range(self.config.max_turns):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.rollout_timeout:
                trajectory.finish_reason = "timeout"
                break

            # Generate LLM response
            gen_result = generate_fn(
                messages=messages,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=self.config.stop_sequences,
            )

            response_text = gen_result.get("text", "")
            log_probs = gen_result.get("log_probs")
            tokens = gen_result.get("tokens")

            # Parse tool calls
            tool_calls = parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls — agent is done
                turn = TurnRecord(
                    turn_id=turn_id,
                    response=response_text,
                    tool_calls=[],
                    tool_results=[],
                    log_probs=log_probs,
                    tokens=tokens,
                )
                trajectory.turns.append(turn)
                trajectory.finished = True
                trajectory.finish_reason = "completed"
                break

            # Execute tool calls
            results = self.env.execute_tool_calls(tool_calls)
            result_dicts = [r.to_dict() for r in results]

            # Record turn
            turn = TurnRecord(
                turn_id=turn_id,
                response=response_text,
                tool_calls=tool_calls,
                tool_results=result_dicts,
                log_probs=log_probs,
                tokens=tokens,
            )
            trajectory.turns.append(turn)

            # Append to context for next turn
            messages.append({"role": "assistant", "content": response_text})
            tool_result_text = self.env.format_tool_results(results)
            messages.append({"role": "user", "content": tool_result_text})

        else:
            # Reached max turns
            trajectory.finish_reason = "max_turns"

        trajectory.total_time = time.time() - start_time
        trajectory.total_tokens = sum(
            len(t.tokens) for t in trajectory.turns if t.tokens
        )

        logger.info(
            "Episode %s: %d turns, %d tool calls, %.1fs, reason=%s",
            episode_id,
            trajectory.num_turns,
            trajectory.num_tool_calls,
            trajectory.total_time,
            trajectory.finish_reason,
        )

        return trajectory


def build_verl_agent_rollout_fn(
    env: ToolEnvironment,
    config: AgentLoopConfig | None = None,
):
    """Build a rollout function compatible with verl's AgentLoop interface.

    This returns a callable that can be registered as the rollout function
    in verl's training pipeline.  It handles multi-turn tool interaction
    and returns the trajectory in DataProto-compatible format.

    Usage in verl config:
        rollout:
          name: agent
          agent_rollout_fn: src.agent_loop.build_verl_agent_rollout_fn
    """
    loop = AgentLoop(env=env, config=config)

    def rollout_fn(
        prompt: str,
        generate_fn: Any,
        system_prompt: str = "",
        episode_id: str = "",
        **kwargs,
    ) -> dict:
        trajectory = loop.run(
            prompt=prompt,
            system_prompt=system_prompt,
            generate_fn=generate_fn,
            episode_id=episode_id,
        )
        return {
            "trajectory": trajectory.to_dict_list(),
            "log_probs": trajectory.get_all_log_probs(),
            "num_turns": trajectory.num_turns,
            "num_tool_calls": trajectory.num_tool_calls,
            "finished": trajectory.finished,
            "finish_reason": trajectory.finish_reason,
            "total_time": trajectory.total_time,
        }

    return rollout_fn
