#!/usr/bin/env bash
# Run from /workspace/ironsite on the Vast.ai instance
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv inside the project folder
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Ensure data, jobs, and models directories exist
mkdir -p data jobs models

if [ ! -f models/last.pt ]; then
  echo "⚠️  models/last.pt not found — copy your fine-tuned YOLO weights here"
fi

# SAM3 and Qwen3-VL are preloaded at /workspace/models/ on this instance

echo ""
echo "Setup complete. To start the server (GPU2 excluded):"
echo "source venv/bin/activate && \\"
echo "CUDA_VISIBLE_DEVICES=0,1,3 \\"
echo "VLM_GPU_ID=1 \\"
echo "VLM_CHUNK_SECONDS=60 \\"
echo "VLM_MAX_HAZARD_FRAMES=40 \\"
echo "VLM_VIDEO_FPS=0.5 \\"
echo "uvicorn main:app --host 0.0.0.0 --port 8000"
