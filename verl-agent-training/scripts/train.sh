#!/bin/bash
# =============================================================================
# Agentic RL Tool-Calling Agent Training Script
# =============================================================================
# Starts verl training on a Ray cluster with the specified configuration.
#
# Usage:
#   # Single-node validation (8 GPU)
#   bash scripts/train.sh --config configs/grpo_deepseek_tool.yaml --gpus 8
#
#   # Multi-node cluster (64 GPU)
#   bash scripts/train.sh --config configs/grpo_deepseek_tool.yaml --gpus 64
#
#   # With custom model
#   bash scripts/train.sh --config configs/grpo_deepseek_tool.yaml \
#       --model Qwen/Qwen2.5-7B-Instruct --backend fsdp --gpus 8
# =============================================================================

set -euo pipefail

# ---- Default Arguments ----
CONFIG="configs/grpo_deepseek_tool.yaml"
NUM_GPUS=8
MODEL=""
BACKEND=""
RAY_ADDRESS="auto"
WANDB_PROJECT="verl-agent"
DRY_RUN=false

# ---- Parse Arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG="$2";       shift 2 ;;
        --gpus)      NUM_GPUS="$2";     shift 2 ;;
        --model)     MODEL="$2";        shift 2 ;;
        --backend)   BACKEND="$2";      shift 2 ;;
        --ray)       RAY_ADDRESS="$2";  shift 2 ;;
        --wandb)     WANDB_PROJECT="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true;      shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --config CONFIG    Training config YAML (default: $CONFIG)"
            echo "  --gpus N           Number of GPUs (default: $NUM_GPUS)"
            echo "  --model NAME       Override model name"
            echo "  --backend NAME     Override training backend (fsdp/megatron)"
            echo "  --ray ADDRESS      Ray cluster address (default: auto)"
            echo "  --wandb PROJECT    W&B project name"
            echo "  --dry-run          Print config without running"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Environment Setup ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Tokenizers parallelism (avoid deadlocks with Ray)
export TOKENIZERS_PARALLELISM=false

# Ray
export RAY_ADDRESS="${RAY_ADDRESS}"

# WandB
export WANDB_PROJECT="${WANDB_PROJECT}"

echo "============================================="
echo " Agentic RL Tool-Calling Agent Training"
echo "============================================="
echo " Config:    ${CONFIG}"
echo " GPUs:      ${NUM_GPUS}"
echo " Ray:       ${RAY_ADDRESS}"
echo " Backend:   ${BACKEND:-from config}"
echo " Model:     ${MODEL:-from config}"
echo "============================================="

# ---- Build Override Arguments ----
OVERRIDES=""
if [[ -n "$MODEL" ]]; then
    OVERRIDES="${OVERRIDES} model.name=${MODEL}"
fi
if [[ -n "$BACKEND" ]]; then
    OVERRIDES="${OVERRIDES} training.backend=${BACKEND}"
fi
OVERRIDES="${OVERRIDES} resources.num_gpus=${NUM_GPUS}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry run — would execute:"
    echo "  python -m verl.trainer.main \\"
    echo "    --config ${CONFIG} \\"
    echo "    ${OVERRIDES}"
    exit 0
fi

# ---- Check Dependencies ----
echo ""
echo "Checking dependencies..."
python -c "import verl; print(f'verl version: {verl.__version__}')" 2>/dev/null || {
    echo "ERROR: verl not installed. Run: pip install verl"
    exit 1
}
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null || {
    echo "WARNING: vLLM not installed. Install for server mode rollout."
}

# ---- Check Ray Cluster ----
echo "Checking Ray cluster..."
if [[ "$RAY_ADDRESS" == "auto" ]]; then
    # Start Ray if not already running
    ray status 2>/dev/null || {
        echo "Starting local Ray cluster..."
        ray start --head --num-gpus "${NUM_GPUS}"
    }
fi

RAY_NODES=$(python -c "import ray; ray.init(address='${RAY_ADDRESS}'); print(len(ray.nodes())); ray.shutdown()" 2>/dev/null || echo "?")
echo "Ray cluster nodes: ${RAY_NODES}"

# ---- Launch Training ----
echo ""
echo "Launching training..."
echo "Start time: $(date)"
echo ""

python -m verl.trainer.main \
    --config "${CONFIG}" \
    ${OVERRIDES}

echo ""
echo "Training complete at $(date)"
