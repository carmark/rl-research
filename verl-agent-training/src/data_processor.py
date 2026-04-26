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
    model_family: str = "deepseek"  # "deepseek", "qwen", "llama"
    tool_call_format: str = "xml"  # "xml" (<tool_call>), "json" (function_call)
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
