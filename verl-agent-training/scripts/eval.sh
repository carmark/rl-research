#!/bin/bash
# =============================================================================
# Evaluation Script for Tool-Calling Agent
# =============================================================================
# Evaluates a trained agent checkpoint on eval datasets.
#
# Usage:
#   bash scripts/eval.sh --checkpoint checkpoints/step_1000 --data data/eval_prompts.jsonl
# =============================================================================

set -euo pipefail

CHECKPOINT=""
DATA_FILE="data/eval_prompts.jsonl"
OUTPUT_DIR="eval_results/"
NUM_GPUS=8
TP_SIZE=8
MAX_TURNS=10
TEMPERATURE=0.0    # greedy for eval
NUM_SAMPLES=1      # single sample for eval

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2";  shift 2 ;;
        --data)       DATA_FILE="$2";   shift 2 ;;
        --output)     OUTPUT_DIR="$2";  shift 2 ;;
        --gpus)       NUM_GPUS="$2";    shift 2 ;;
        --tp)         TP_SIZE="$2";     shift 2 ;;
        --max-turns)  MAX_TURNS="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 --checkpoint PATH [options]"
            echo "  --checkpoint PATH   Model checkpoint to evaluate"
            echo "  --data FILE         Evaluation data (default: data/eval_prompts.jsonl)"
            echo "  --output DIR        Output directory (default: eval_results/)"
            echo "  --gpus N            Number of GPUs (default: 8)"
            echo "  --tp N              Tensor parallel size (default: 8)"
            echo "  --max-turns N       Max agent turns (default: 10)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo " Evaluating Tool-Calling Agent"
echo "============================================="
echo " Checkpoint: ${CHECKPOINT}"
echo " Data:       ${DATA_FILE}"
echo " Output:     ${OUTPUT_DIR}"
echo " GPUs:       ${NUM_GPUS} (TP=${TP_SIZE})"
echo " Max turns:  ${MAX_TURNS}"
echo "============================================="

python -c "
import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('$0'))))

from src.data_processor import DataProcessor, DataConfig
from src.tool_env import ToolEnvironment, EnvironmentConfig
from src.agent_loop import AgentLoop, AgentLoopConfig
from src.reward_function import compute_reward, RewardConfig

# Load eval data
processor = DataProcessor(DataConfig(eval_file='${DATA_FILE}'))
prompts = processor.load_prompts('${DATA_FILE}')

if not prompts:
    print('ERROR: No evaluation prompts found')
    sys.exit(1)

print(f'Loaded {len(prompts)} evaluation prompts')

# Setup environment
env = ToolEnvironment(EnvironmentConfig())
loop_config = AgentLoopConfig(
    max_turns=${MAX_TURNS},
    temperature=${TEMPERATURE},
)

# Setup reward
reward_config = RewardConfig()

# NOTE: In a real evaluation, you would load the checkpoint into vLLM:
#   from vllm import LLM
#   llm = LLM(model='${CHECKPOINT}', tensor_parallel_size=${TP_SIZE})
#
# For now, we demonstrate the evaluation framework structure.
print()
print('Evaluation framework initialized.')
print(f'  Tools available: {env.registry.list_tools()}')
print(f'  Max turns: {loop_config.max_turns}')
print(f'  Temperature: {loop_config.temperature}')
print()
print('To run actual evaluation, load the checkpoint into vLLM and provide')
print('the generate_fn to AgentLoop.run().')
print()

# Compute baseline reward statistics on eval data format
results = []
for i, prompt_data in enumerate(prompts[:5]):  # Preview first 5
    print(f'  [{i+1}] {prompt_data.get(\"prompt\", \"\")[:80]}...')
    print(f'      Ground truth: {prompt_data.get(\"ground_truth\", \"N/A\")}')

print()
print(f'Total prompts to evaluate: {len(prompts)}')
print(f'Results will be saved to: ${OUTPUT_DIR}/results.jsonl')
" 2>&1

echo ""
echo "Evaluation script complete."
echo "Note: Full evaluation requires a running vLLM server with the checkpoint loaded."
