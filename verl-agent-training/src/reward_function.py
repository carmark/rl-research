"""Custom reward function conforming to verl's compute_score interface.

verl calls this function for each completed trajectory to compute a scalar
reward.  The function is registered in the training config:

    reward:
      custom_reward_function:
        path: src/reward_function.py
        name: compute_score

verl's expected signature (sync or async):

    def compute_score(
        data_source: str,       # dataset name
        solution_str: str,      # model's full response text
        ground_truth: str,      # expected answer from dataset
        extra_info: dict = None,# metadata including num_turns, tool_rewards, etc.
        **kwargs
    ) -> float | dict           # float or {"score": float, ...}

When using ToolAgentLoop, extra_info contains:
    - num_turns: int
    - tool_rewards: list[float]  (step rewards from BaseTool.execute)
    - turn_scores: list[float]
    - rollout_reward_scores: list[float]
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---- Configuration (can be overridden via kwargs or environment) ----
DEFAULT_CONFIG = {
    "rule_weight": 0.7,
    "tool_reward_weight": 0.3,
    "task_correct": 1.0,
    "task_incorrect": 0.0,
    "format_bonus": 0.1,
    "format_penalty": -0.1,
    "noop_factor": 0.3,
    "match_mode": "contains",   # "exact", "contains", "numeric"
    # DeepSeek V4 GRM mode
    "reward_mode": "rule",      # "rule" (default), "grm" (Generative Reward Model)
    "reasoning_effort": "auto", # "auto", "non_think", "think_high", "think_max"
    "grm_rubric_weight": 0.4,   # weight for GRM rubric score when in grm mode
    "effort_penalty_factor": 0.1,  # penalty multiplier for reasoning effort mismatch
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute reward score for a tool-calling agent trajectory.

    This is the entry point called by verl's reward system.

    Returns:
        dict with "score" key (required by verl) plus diagnostic fields.
    """
    extra_info = extra_info or {}
    config = {**DEFAULT_CONFIG, **kwargs}

    reward_mode = config["reward_mode"]

    # --- GRM mode (DeepSeek V4-style Generative Reward Model) ---
    if reward_mode == "grm":
        return _compute_grm_score(
            data_source, solution_str, ground_truth, extra_info, config
        )

    # --- Standard rule-based mode ---
    # --- 1. Task completion (rule-based) ---
    match_mode = config["match_mode"]
    task_correct = _verify_answer(solution_str, ground_truth, match_mode)
    task_score = config["task_correct"] if task_correct else config["task_incorrect"]

    # --- 2. Format quality ---
    format_score = _check_format(solution_str, config)

    # --- 3. Tool step rewards (from ToolAgentLoop) ---
    tool_rewards = extra_info.get("tool_rewards", [])
    tool_reward_sum = sum(tool_rewards) if tool_rewards else 0.0

    # --- 4. No-op detection ---
    num_turns = extra_info.get("num_turns", 1)
    has_tools = bool(tool_rewards) or _has_tool_calls(solution_str)
    is_noop = not has_tools

    # --- 5. Reasoning Effort awareness (V4-style) ---
    effort_adjustment = _compute_effort_adjustment(
        solution_str, extra_info, config
    )

    # --- 6. Combine ---
    rule_score = task_score + format_score
    total = (config["rule_weight"] * rule_score
             + config["tool_reward_weight"] * tool_reward_sum)

    if is_noop:
        total *= config["noop_factor"]

    total += effort_adjustment

    return {
        "score": total,           # required by verl
        "task_correct": task_correct,
        "task_score": task_score,
        "format_score": format_score,
        "tool_reward_sum": tool_reward_sum,
        "num_tool_calls": len(tool_rewards),
        "num_turns": num_turns,
        "is_noop": is_noop,
        "effort_adjustment": effort_adjustment,
        "reward_mode": reward_mode,
    }


# ---- Internal helpers ----

def _verify_answer(solution: str, ground_truth: str, mode: str) -> bool:
    """Check if the model's answer matches the ground truth."""
    if not solution or not ground_truth:
        return False

    gt = str(ground_truth).strip().lower()
    sol = solution.strip().lower()

    if mode == "exact":
        return gt == sol
    elif mode == "contains":
        return gt in sol
    elif mode == "numeric":
        return _numeric_match(sol, gt)
    return gt in sol


def _numeric_match(solution: str, ground_truth: str) -> bool:
    """Extract numbers and compare."""
    gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth)
    if not gt_nums:
        return False
    gt_val = float(gt_nums[-1])

    sol_nums = re.findall(r"-?\d+\.?\d*", solution)
    for n in reversed(sol_nums):
        try:
            if abs(float(n) - gt_val) / (abs(gt_val) + 1e-8) < 0.01:
                return True
        except ValueError:
            continue
    return False


def _check_format(solution: str, config: dict) -> float:
    """Score the format quality of tool calls in the response."""
    # Look for <tool_call> blocks
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = pattern.findall(solution)
    if not matches:
        return 0.0  # no tool calls to evaluate

    valid = 0
    for raw in matches:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                valid += 1
        except (json.JSONDecodeError, TypeError):
            pass

    total = len(matches)
    if total == 0:
        return 0.0

    ratio = valid / total
    return config["format_bonus"] * ratio + config["format_penalty"] * (1 - ratio)


def _has_tool_calls(solution: str) -> bool:
    """Check if the response text contains any tool call blocks."""
    return "<tool_call>" in solution or "<|DSML|tool_call>" in solution


# ---- DeepSeek V4 GRM Mode ----

def _compute_grm_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    config: dict,
) -> dict:
    """Compute reward using Generative Reward Model (GRM) mode.

    In GRM mode (inspired by DeepSeek V4), the actor model also serves as
    a reward model. The rubric score is expected in extra_info["grm_rubric_score"],
    provided by the GRM evaluation pass.

    GRM generates Rubric-Guided evaluation text, from which a structured
    score is extracted. This combines with rule-based signals.
    """
    # Rule-based components (always computed)
    task_correct = _verify_answer(solution_str, ground_truth, config["match_mode"])
    task_score = config["task_correct"] if task_correct else config["task_incorrect"]
    format_score = _check_format(solution_str, config)

    # GRM rubric score (from external GRM evaluation pass)
    grm_rubric_score = extra_info.get("grm_rubric_score", 0.0)

    # Tool rewards
    tool_rewards = extra_info.get("tool_rewards", [])
    tool_reward_sum = sum(tool_rewards) if tool_rewards else 0.0

    # No-op detection
    has_tools = bool(tool_rewards) or _has_tool_calls(solution_str)
    is_noop = not has_tools

    # Reasoning Effort adjustment
    effort_adjustment = _compute_effort_adjustment(
        solution_str, extra_info, config
    )

    # GRM combines rule score with rubric score
    rule_weight = 1.0 - config["grm_rubric_weight"]
    rule_score = task_score + format_score
    total = (rule_weight * rule_score
             + config["grm_rubric_weight"] * grm_rubric_score
             + config["tool_reward_weight"] * tool_reward_sum)

    if is_noop:
        total *= config["noop_factor"]

    total += effort_adjustment

    return {
        "score": total,
        "task_correct": task_correct,
        "task_score": task_score,
        "format_score": format_score,
        "grm_rubric_score": grm_rubric_score,
        "tool_reward_sum": tool_reward_sum,
        "num_tool_calls": len(tool_rewards),
        "num_turns": extra_info.get("num_turns", 1),
        "is_noop": is_noop,
        "effort_adjustment": effort_adjustment,
        "reward_mode": "grm",
    }


def _compute_effort_adjustment(
    solution_str: str,
    extra_info: dict,
    config: dict,
) -> float:
    """Compute Reasoning Effort-aware reward adjustment.

    DeepSeek V4 supports multiple reasoning effort levels. This function
    adjusts the reward based on whether the model's reasoning depth matches
    the expected effort level.

    - Non-think: penalize if model includes lengthy reasoning
    - Think Max: penalize if model skips reasoning on complex tasks
    """
    effort = config.get("reasoning_effort", "auto")
    if effort == "auto":
        # Determine effort from extra_info or task metadata
        effort = extra_info.get("reasoning_effort", "think_high")

    factor = config.get("effort_penalty_factor", 0.1)

    # Count thinking blocks in response
    import re
    thinking_blocks = re.findall(r"<think>.*?</think>", solution_str, re.DOTALL)
    has_thinking = len(thinking_blocks) > 0
    thinking_length = sum(len(b) for b in thinking_blocks)

    if effort == "non_think":
        # Penalize excessive reasoning in non-think mode
        if thinking_length > 200:
            return -factor * min(thinking_length / 1000, 1.0)
    elif effort == "think_max":
        # Penalize lack of reasoning in think_max mode
        if not has_thinking or thinking_length < 100:
            return -factor
    # think_high: no adjustment

    return 0.0
