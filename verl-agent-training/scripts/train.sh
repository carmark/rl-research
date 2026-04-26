#!/bin/bash
# =============================================================================
# Agentic RL Tool-Calling Agent Training Script
# =============================================================================
# Starts verl training using verl's actual CLI entry point.
#
# Usage:
#   # Single-node validation (8 GPU, Qwen2.5-7B, FSDP)
#   bash scripts/train.sh --gpus 8 \
#       --override actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
#       --override actor_rollout_ref.actor.strategy=fsdp
#
#   # Multi-node cluster (64 GPU, DeepSeek MoE, Megatron)
#   bash scripts/train.sh --gpus 64
#
#   # Custom config
#   bash scripts/train.sh --config ppo_deepseek_tool --gpus 64
# =============================================================================

set -euo pipefail

# ---- Defaults ----
CONFIG_NAME="grpo_deepseek_tool"
NUM_GPUS=8
NNODES=1
GPUS_PER_NODE=8
DRY_RUN=false
OVERRIDES=()

# ---- Parse Args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)     CONFIG_NAME="$2";   shift 2 ;;
        --gpus)       NUM_GPUS="$2";      shift 2 ;;
        --nnodes)     NNODES="$2";        shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --override)   OVERRIDES+=("$2");  shift 2 ;;
        --dry-run)    DRY_RUN=true;       shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --config NAME      Config name in configs/ (default: grpo_deepseek_tool)"
            echo "  --gpus N           Total GPUs (default: 8)"
            echo "  --nnodes N         Number of nodes (default: 1)"
            echo "  --gpus-per-node N  GPUs per node (default: 8)"
            echo "  --override K=V     Hydra config override (can repeat)"
            echo "  --dry-run          Print command without running"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Compute nnodes from total GPUs if not explicitly set
if [[ "$NNODES" == "1" && "$NUM_GPUS" -gt "$GPUS_PER_NODE" ]]; then
    NNODES=$(( NUM_GPUS / GPUS_PER_NODE ))
fi

# ---- Environment ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================="
echo " verl Agentic RL Training"
echo "============================================="
echo " Config:         ${CONFIG_NAME}"
echo " Nodes:          ${NNODES}"
echo " GPUs/node:      ${GPUS_PER_NODE}"
echo " Total GPUs:     $((NNODES * GPUS_PER_NODE))"
echo " Project dir:    ${PROJECT_DIR}"
echo "============================================="

# ---- Build command ----
CMD=(
    python3 -m verl.trainer.main_ppo
    "--config-path=${PROJECT_DIR}/configs"
    "--config-name=${CONFIG_NAME}"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
)

# Append user overrides
for ov in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do
    CMD+=("$ov")
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry run — would execute:"
    echo "  ${CMD[*]}"
    exit 0
fi

# ---- Preflight checks ----
echo ""
echo "Checking dependencies..."
python3 -c "import verl; print(f'  verl: {verl.__version__}')" 2>/dev/null || {
    echo "ERROR: verl not installed. Run: pip install 'verl>=0.7.1'"
    exit 1
}
python3 -c "import vllm; print(f'  vLLM: {vllm.__version__}')" 2>/dev/null || {
    echo "WARNING: vLLM not installed."
}
python3 -c "import ray; print(f'  Ray:  {ray.__version__}')" 2>/dev/null || {
    echo "WARNING: Ray not installed."
}

# ---- Check Ray ----
echo ""
ray status 2>/dev/null || {
    echo "Starting local Ray..."
    ray start --head --num-gpus "${GPUS_PER_NODE}"
}

# ---- Launch ----
echo ""
echo "Launching: ${CMD[*]}"
echo "Start: $(date)"
echo ""

"${CMD[@]}"

echo ""
echo "Done: $(date)"
