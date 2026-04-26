"""Reward function for Agentic RL tool-calling training.

Implements a hybrid reward system combining:
  - Rule-based rewards (format, task completion, process)
  - LLM-as-Judge rewards (reasoning quality)
  - No-op detection and penalty
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the reward function."""

    # Weights
    rule_weight: float = 0.7
    judge_weight: float = 0.3

    # Component scores
    format_correct: float = 0.1
    format_incorrect: float = -0.1
    task_correct: float = 1.0
    task_incorrect: float = 0.0
    process_per_call: float = 0.05
    process_max_calls: int = 10

    # No-op penalty: multiply total reward by this factor when no tools called
    noop_factor: float = 0.3

    # LLM-as-Judge
    judge_model: str = "gpt-4"
    judge_endpoint: str | None = None
    judge_api_key: str | None = None
    judge_enabled: bool = False  # disabled by default for cost


def check_tool_call_format(trajectory: list[dict]) -> tuple[float, int, int]:
    """Check whether tool calls in the trajectory have valid JSON format.

    Returns:
        (format_score, num_valid_calls, num_total_calls)
    """
    num_valid = 0
    num_total = 0

    for turn in trajectory:
        tool_calls = turn.get("tool_calls", [])
        for call in tool_calls:
            num_total += 1
            if _is_valid_tool_call(call):
                num_valid += 1

    if num_total == 0:
        return 0.0, 0, 0

    return num_valid / num_total, num_valid, num_total


def _is_valid_tool_call(call: dict) -> bool:
    """Validate that a tool call has the required structure."""
    if not isinstance(call, dict):
        return False
    if "name" not in call:
        return False
    if "arguments" not in call:
        return False
    if not isinstance(call["name"], str) or not call["name"]:
        return False
    if not isinstance(call["arguments"], dict):
        return False
    return True


def verify_answer(
    trajectory: list[dict],
    ground_truth: Any,
    match_mode: str = "exact",
) -> float:
    """Verify whether the final answer matches the ground truth.

    Args:
        trajectory: List of turn dicts, last turn should contain 'response'.
        ground_truth: Expected answer (string, number, or list).
        match_mode: 'exact', 'contains', or 'numeric'.

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    if not trajectory:
        return 0.0

    final_response = trajectory[-1].get("response", "")
    if not final_response:
        return 0.0

    gt_str = str(ground_truth).strip().lower()
    resp_str = final_response.strip().lower()

    if match_mode == "exact":
        return 1.0 if gt_str == resp_str else 0.0
    elif match_mode == "contains":
        return 1.0 if gt_str in resp_str else 0.0
    elif match_mode == "numeric":
        return _numeric_match(resp_str, gt_str)
    else:
        return 1.0 if gt_str == resp_str else 0.0


def _numeric_match(response: str, ground_truth: str) -> float:
    """Extract numbers from both strings and compare."""
    resp_nums = re.findall(r"-?\d+\.?\d*", response)
    gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth)
    if not gt_nums:
        return 0.0
    gt_val = float(gt_nums[-1])
    for num_str in reversed(resp_nums):
        try:
            if abs(float(num_str) - gt_val) < 1e-6:
                return 1.0
        except ValueError:
            continue
    return 0.0


def count_valid_tool_calls(trajectory: list[dict]) -> int:
    """Count the number of tool calls that executed successfully."""
    count = 0
    for turn in trajectory:
        for result in turn.get("tool_results", []):
            if isinstance(result, dict) and result.get("success", False):
                count += 1
    return count


def has_tool_calls(trajectory: list[dict]) -> bool:
    """Check if the trajectory contains any tool calls."""
    for turn in trajectory:
        if turn.get("tool_calls"):
            return True
    return False


def llm_judge_score(
    prompt: str,
    trajectory: list[dict],
    config: RewardConfig,
) -> float:
    """Call an external LLM to evaluate the quality of the trajectory.

    Returns a score between 0.0 and 1.0.
    """
    if not config.judge_enabled or not config.judge_endpoint:
        return 0.5  # neutral score when judge is disabled

    try:
        import requests
    except ImportError:
        logger.warning("requests not available, returning neutral judge score")
        return 0.5

    # Build evaluation prompt
    traj_text = _format_trajectory_for_judge(trajectory)
    eval_prompt = (
        "You are an expert evaluator. Rate the quality of the following "
        "agent trajectory on a scale of 0 to 1.\n\n"
        f"Task: {prompt}\n\n"
        f"Trajectory:\n{traj_text}\n\n"
        "Consider:\n"
        "1. Were the right tools selected?\n"
        "2. Were the tool arguments reasonable?\n"
        "3. Was the reasoning logical?\n"
        "4. Was the final answer well-supported?\n\n"
        "Respond with ONLY a number between 0 and 1."
    )

    try:
        headers = {"Content-Type": "application/json"}
        if config.judge_api_key:
            headers["Authorization"] = f"Bearer {config.judge_api_key}"

        resp = requests.post(
            config.judge_endpoint,
            headers=headers,
            json={
                "model": config.judge_model,
                "messages": [{"role": "user", "content": eval_prompt}],
                "temperature": 0.0,
                "max_tokens": 16,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        score = float(content)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning("LLM judge failed: %s", e)
        return 0.5


def _format_trajectory_for_judge(trajectory: list[dict]) -> str:
    """Format the trajectory for the LLM judge."""
    lines = []
    for i, turn in enumerate(trajectory):
        lines.append(f"--- Turn {i + 1} ---")
        if turn.get("response"):
            lines.append(f"Agent: {turn['response'][:500]}")
        for call in turn.get("tool_calls", []):
            lines.append(f"Tool call: {json.dumps(call, ensure_ascii=False)[:200]}")
        for result in turn.get("tool_results", []):
            lines.append(f"Tool result: {json.dumps(result, ensure_ascii=False)[:200]}")
    return "\n".join(lines)


def compute_reward(
    prompt: str,
    trajectory: list[dict],
    ground_truth: Any,
    config: RewardConfig | None = None,
    match_mode: str = "contains",
) -> dict[str, float]:
    """Compute the total reward for a trajectory.

    Args:
        prompt: The original task prompt.
        trajectory: List of turn dicts with keys:
            - response: str (LLM output)
            - tool_calls: list[dict] (parsed tool calls)
            - tool_results: list[dict] (execution results)
        ground_truth: Expected answer for verification.
        config: Reward configuration. Uses defaults if None.
        match_mode: Answer matching mode ('exact', 'contains', 'numeric').

    Returns:
        Dict with individual scores and total reward.
    """
    if config is None:
        config = RewardConfig()

    # 1. Format reward
    format_ratio, num_valid, num_total = check_tool_call_format(trajectory)
    if num_total > 0:
        format_score = (
            config.format_correct * format_ratio
            + config.format_incorrect * (1 - format_ratio)
        )
    else:
        format_score = 0.0

    # 2. Task completion reward
    task_score = verify_answer(trajectory, ground_truth, match_mode)
    task_reward = config.task_correct if task_score > 0.5 else config.task_incorrect

    # 3. Process reward
    valid_calls = min(count_valid_tool_calls(trajectory), config.process_max_calls)
    process_score = valid_calls * config.process_per_call

    # 4. LLM-as-Judge score
    judge_score = llm_judge_score(prompt, trajectory, config)

    # 5. Combine
    rule_score = format_score + task_reward + process_score
    total = config.rule_weight * rule_score + config.judge_weight * judge_score

    # 6. No-op penalty
    is_noop = not has_tool_calls(trajectory)
    if is_noop:
        total *= config.noop_factor

    return {
        "total": total,
        "format_score": format_score,
        "task_score": task_reward,
        "process_score": process_score,
        "judge_score": judge_score,
        "num_valid_calls": num_valid,
        "num_total_calls": num_total,
        "is_noop": is_noop,
    }
