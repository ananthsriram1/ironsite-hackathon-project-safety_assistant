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

if [ ! -f models/sam3.pt ]; then
  echo "Downloading SAM3 weights..."
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir="models",
    local_dir_use_symlinks=False,
    token=os.getenv("HF_TOKEN"),
)
print("SAM3 downloaded to models/sam3.pt")
EOF
fi

echo ""
echo "Setup complete. To start the server:"
echo "  source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
