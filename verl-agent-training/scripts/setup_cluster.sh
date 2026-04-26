#!/bin/bash
# =============================================================================
# Cluster Environment Setup Script
# =============================================================================
# Initializes the environment for Agentic RL training:
#   1. Installs Python dependencies (verl, vLLM, etc.)
#   2. Sets up Ray cluster
#   3. Builds sandbox Docker image
#   4. Downloads model weights (optional)
#
# Usage:
#   # On head node:
#   bash scripts/setup_cluster.sh --head --gpus-per-node 8
#
#   # On worker nodes:
#   bash scripts/setup_cluster.sh --worker --head-address <head_ip>:6379
#
#   # Download model:
#   bash scripts/setup_cluster.sh --download-model deepseek-ai/DeepSeek-V2.5
# =============================================================================

set -euo pipefail

# ---- Default Arguments ----
NODE_TYPE=""
HEAD_ADDRESS=""
GPUS_PER_NODE=8
DOWNLOAD_MODEL=""
SKIP_DOCKER=false
PYTHON_ENV="verl-agent"

while [[ $# -gt 0 ]]; do
    case $1 in
        --head)           NODE_TYPE="head";           shift ;;
        --worker)         NODE_TYPE="worker";         shift ;;
        --head-address)   HEAD_ADDRESS="$2";          shift 2 ;;
        --gpus-per-node)  GPUS_PER_NODE="$2";        shift 2 ;;
        --download-model) DOWNLOAD_MODEL="$2";        shift 2 ;;
        --skip-docker)    SKIP_DOCKER=true;           shift ;;
        -h|--help)
            echo "Usage: $0 [--head|--worker] [options]"
            echo "  --head                 Setup as head node"
            echo "  --worker               Setup as worker node"
            echo "  --head-address ADDR    Head node address (worker only)"
            echo "  --gpus-per-node N      GPUs per node (default: 8)"
            echo "  --download-model NAME  Download model from HuggingFace"
            echo "  --skip-docker          Skip Docker image build"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo " Cluster Environment Setup"
echo "============================================="

# ---- Step 1: Install Python Dependencies ----
echo ""
echo "[Step 1] Installing Python dependencies..."

pip install --upgrade pip

# Core framework
pip install "verl>=0.7.1"

# Inference engine
pip install vllm

# Training backend (Megatron)
pip install megatron-core

# Utilities
pip install ray[default] wandb transformers accelerate

# Project dependencies
pip install requests pyyaml

echo "Dependencies installed."

# ---- Step 2: Ray Cluster Setup ----
if [[ -n "$NODE_TYPE" ]]; then
    echo ""
    echo "[Step 2] Setting up Ray cluster ($NODE_TYPE node)..."

    # Stop existing Ray
    ray stop 2>/dev/null || true

    if [[ "$NODE_TYPE" == "head" ]]; then
        ray start --head \
            --num-gpus "${GPUS_PER_NODE}" \
            --dashboard-host 0.0.0.0 \
            --dashboard-port 8265

        echo ""
        echo "Head node started. Worker nodes should connect with:"
        HEAD_IP=$(hostname -I | awk '{print $1}')
        echo "  bash scripts/setup_cluster.sh --worker --head-address ${HEAD_IP}:6379"

    elif [[ "$NODE_TYPE" == "worker" ]]; then
        if [[ -z "$HEAD_ADDRESS" ]]; then
            echo "ERROR: --head-address required for worker nodes"
            exit 1
        fi
        ray start --address "${HEAD_ADDRESS}" \
            --num-gpus "${GPUS_PER_NODE}"
        echo "Worker node joined cluster at ${HEAD_ADDRESS}"
    fi

    # Verify
    echo ""
    echo "Ray cluster status:"
    ray status
fi

# ---- Step 3: Docker Sandbox Image ----
if [[ "$SKIP_DOCKER" != "true" ]]; then
    echo ""
    echo "[Step 3] Building sandbox Docker image..."

    if command -v docker &>/dev/null; then
        docker build -t verl-sandbox:latest -f docker/Dockerfile.sandbox .
        echo "Sandbox image built: verl-sandbox:latest"
    else
        echo "WARNING: Docker not installed. Skipping sandbox image build."
        echo "  Code executor will use subprocess backend (less isolated)."
    fi
else
    echo ""
    echo "[Step 3] Skipping Docker image build (--skip-docker)"
fi

# ---- Step 4: Download Model ----
if [[ -n "$DOWNLOAD_MODEL" ]]; then
    echo ""
    echo "[Step 4] Downloading model: ${DOWNLOAD_MODEL}..."

    python -c "
from huggingface_hub import snapshot_download
import os

model_name = '${DOWNLOAD_MODEL}'
cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f'Downloading {model_name} to {cache_dir}...')
snapshot_download(model_name, cache_dir=cache_dir)
print('Download complete.')
"
fi

# ---- Step 5: Verify Setup ----
echo ""
echo "[Verification]"
echo "============================================="

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'GPU 0: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: NOT INSTALLED')

try:
    import verl
    print(f'verl: {verl.__version__}')
except Exception:
    print('verl: NOT INSTALLED')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except Exception:
    print('vLLM: NOT INSTALLED')

try:
    import ray
    print(f'Ray: {ray.__version__}')
except Exception:
    print('Ray: NOT INSTALLED')
"

echo "============================================="
echo " Setup complete!"
echo "============================================="
