"""Data preprocessing for Agentic RL tool-calling training.

Handles:
  - Loading and formatting training prompts
  - Injecting tool descriptions into system prompts
  - Building the conversation template for different model families
  - Preparing data for verl DataProto format
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    train_file: str = "data/train_prompts.jsonl"
    eval_file: str = "data/eval_prompts.jsonl"
    max_prompt_length: int = 2048
    model_family: str = "deepseek"  # "deepseek", "deepseek_v4", "qwen", "llama"
    tool_call_format: str = "xml"  # "xml" (<tool_call>), "dsml" (V4 |DSML|), "json"
    reasoning_effort: str = "think_high"  # "non_think", "think_high", "think_max"
    enable_million_token: bool = False  # V4-style metadata/per-token separation
    shuffle: bool = True
    seed: int = 42


# System prompt templates per model family
SYSTEM_PROMPTS = {
    "deepseek": (
        "You are a helpful AI assistant with access to external tools. "
        "When you need to use a tool, output a <tool_call> block with a JSON "
        'object containing "name" and "arguments". After receiving the tool '
        "result in a <tool_response> block, continue your reasoning. "
        "When you have enough information, provide your final answer directly.\n\n"
        "{tool_descriptions}"
    ),
    "deepseek_v4": (
        "You are a helpful AI assistant with access to external tools.\n\n"
        "## Tool Call Format\n"
        "When you need to use a tool, output tool calls using the following "
        "|DSML| XML schema:\n\n"
        "<|DSML|tool_calls>\n"
        "  <|DSML|tool_call>\n"
        '    {{"name": "tool_name", "arguments": {{"arg": "value"}}}}\n'
        "  </|DSML|tool_call>\n"
        "</|DSML|tool_calls>\n\n"
        "Tool results will appear in <|DSML|tool_response> blocks.\n\n"
        "## Reasoning\n"
        "You may use <think>...</think> blocks to show your reasoning process. "
        "Your thinking will be preserved across tool calls for continuity.\n\n"
        "{reasoning_effort_instruction}"
        "## Available Tools\n"
        "{tool_descriptions}"
    ),
    "qwen": (
        "You are Qwen, a helpful AI assistant. You have access to the following tools:\n\n"
        "{tool_descriptions}\n\n"
        "To call a tool, use the <tool_call> XML tag with a JSON body.\n"
        "Example:\n"
        "<tool_call>\n"
        '{{"name": "calculator", "arguments": {{"expression": "2+2"}}}}\n'
        "</tool_call>\n\n"
        "Tool results will appear in <tool_response> tags."
    ),
    "llama": (
        "You are a helpful assistant with tool-calling capabilities. "
        "When you receive a tool call response, use the output to form your answer.\n\n"
        "Available tools:\n{tool_descriptions}"
    ),
}

# Reasoning Effort instructions for DeepSeek V4
REASONING_EFFORT_INSTRUCTIONS = {
    "non_think": "",  # No special instruction
    "think_high": (
        "Think through your approach before acting. "
        "Use <think> blocks to organize your reasoning.\n\n"
    ),
    "think_max": (
        "Please think very carefully and deeply about this task. "
        "Break down the problem step by step in <think> blocks. "
        "Consider edge cases and verify your reasoning before providing "
        "your final answer.\n\n"
    ),
}


@dataclass
class TrainingExample:
    """A single training example."""

    prompt_id: str
    prompt: str
    system_prompt: str
    ground_truth: Any
    metadata: dict


class DataProcessor:
    """Prepare training data for Agentic RL.

    Loads JSONL datasets and formats them with tool descriptions
    and model-specific conversation templates.
    """

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def load_prompts(self, file_path: str | None = None) -> list[dict]:
        """Load prompts from a JSONL file.

        Expected format per line:
        {
            "id": "unique_id",
            "prompt": "The user's question or task",
            "ground_truth": "expected answer",
            "metadata": {}  // optional
        }
        """
        path = Path(file_path or self.config.train_file)
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            return []

        prompts = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    prompts.append(data)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)

        logger.info("Loaded %d prompts from %s", len(prompts), path)
        return prompts

    def build_system_prompt(self, tool_descriptions: str) -> str:
        """Build the system prompt with tool descriptions injected."""
        template = SYSTEM_PROMPTS.get(
            self.config.model_family,
            SYSTEM_PROMPTS["deepseek"],
        )
        # V4 templates need reasoning effort instruction
        if self.config.model_family == "deepseek_v4":
            effort_instruction = REASONING_EFFORT_INSTRUCTIONS.get(
                self.config.reasoning_effort, ""
            )
            return template.format(
                tool_descriptions=tool_descriptions,
                reasoning_effort_instruction=effort_instruction,
            )
        return template.format(tool_descriptions=tool_descriptions)

    def prepare_examples(
        self,
        prompts: list[dict],
        tool_descriptions: str,
    ) -> list[TrainingExample]:
        """Prepare training examples with system prompts."""
        system_prompt = self.build_system_prompt(tool_descriptions)
        examples = []

        for item in prompts:
            prompt_text = item.get("prompt", "")
            if len(prompt_text) > self.config.max_prompt_length:
                logger.warning(
                    "Prompt %s exceeds max length (%d > %d), truncating",
                    item.get("id", "?"),
                    len(prompt_text),
                    self.config.max_prompt_length,
                )
                prompt_text = prompt_text[: self.config.max_prompt_length]

            examples.append(
                TrainingExample(
                    prompt_id=item.get("id", f"prompt_{len(examples)}"),
                    prompt=prompt_text,
                    system_prompt=system_prompt,
                    ground_truth=item.get("ground_truth", ""),
                    metadata=item.get("metadata", {}),
                )
            )

        if self.config.shuffle:
            import random
            rng = random.Random(self.config.seed)
            rng.shuffle(examples)

        return examples

    def build_conversation(self, example: TrainingExample) -> list[dict]:
        """Build a conversation in the standard messages format."""
        return [
            {"role": "system", "content": example.system_prompt},
            {"role": "user", "content": example.prompt},
        ]

    def prepare_for_verl(
        self,
        examples: list[TrainingExample],
    ) -> list[dict]:
        """Convert examples to the format expected by verl DataProto.

        Returns a list of dicts that can be passed to verl's data loading.
        """
        verl_data = []
        for ex in examples:
            verl_data.append(
                {
                    "prompt_id": ex.prompt_id,
                    "messages": self.build_conversation(ex),
                    "ground_truth": ex.ground_truth,
                    "metadata": ex.metadata,
                }
            )
        return verl_data

    def iterate_batches(
        self,
        examples: list[TrainingExample],
        batch_size: int,
    ) -> Iterator[list[TrainingExample]]:
        """Yield batches of examples."""
        for i in range(0, len(examples), batch_size):
            yield examples[i : i + batch_size]

    def prepare_million_token_format(
        self,
        examples: list[TrainingExample],
        output_dir: str,
    ) -> dict:
        """Prepare data in DeepSeek V4-style metadata/per-token separated format.

        For million-token contexts, traditional DataProto with unified tensors
        causes OOM. V4 separates:
        - metadata.jsonl: sample-level fields (prompt_id, reward, etc.)
        - per_token/<id>.bin: per-token fields (input_ids, masks, log_probs)

        This enables:
        - Loading all metadata into memory for scheduling/sampling
        - Loading per-token data on demand from shared memory or filesystem
        """
        if not self.config.enable_million_token:
            logger.info("Million-token format disabled, using standard format")
            return {"format": "standard", "examples": len(examples)}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        per_token_dir = output_path / "per_token"
        per_token_dir.mkdir(exist_ok=True)

        metadata_records = []
        for ex in examples:
            metadata_records.append({
                "prompt_id": ex.prompt_id,
                "ground_truth": ex.ground_truth,
                "metadata": ex.metadata,
                "prompt_length": len(ex.prompt),
                "reasoning_effort": self.config.reasoning_effort,
            })

        metadata_path = output_path / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for record in metadata_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Prepared million-token format: %d examples, metadata at %s",
            len(examples), metadata_path,
        )
        return {
            "format": "million_token",
            "metadata_path": str(metadata_path),
            "per_token_dir": str(per_token_dir),
            "num_examples": len(examples),
        }
