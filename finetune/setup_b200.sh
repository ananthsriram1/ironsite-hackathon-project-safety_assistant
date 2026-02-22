#!/usr/bin/env bash
# =============================================================================
# B200 instance setup for Qwen3-VL fine-tuning on ConstructionSite 10k
# Run on a fresh Ubuntu instance with NVIDIA drivers + CUDA already installed.
# =============================================================================
set -e

echo "==> Updating package list and installing utilities..."
sudo apt-get update
sudo apt-get install -y git git-lfs

echo "==> Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing Hugging Face ecosystem and training libraries..."
pip install transformers datasets accelerate peft trl evaluate bitsandbytes pillow qwen-vl-utils tensorboard

echo "==> Installing flash-attn (faster attention on B200)..."
pip install flash-attn --no-build-isolation

echo "==> Log in to Hugging Face (required for dataset + model access):"
echo "    Get your token from https://huggingface.co/settings/tokens"
echo "    Then run: huggingface-cli login"
huggingface-cli login || true

echo "==> Setup complete. Next: cd into project and run python train.py"
