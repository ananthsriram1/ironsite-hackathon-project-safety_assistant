FROM python:3.11

WORKDIR /app

COPY requirements.txt .
# ultralytics pulls in opencv-python (needs libGL); swap for headless in Docker
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir "opencv-python-headless>=4.6.0"

# Download SAM3 weights at build time
ARG HF_TOKEN
RUN mkdir -p /app/models && \
    python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir='/app/models', local_dir_use_symlinks=False, token=os.getenv('HF_TOKEN'))" \
    || echo "WARNING: SAM3 download failed — mount model at /app/models/sam3.pt"

# Qwen3-VL is ~16GB — too large to bake into the image.
# Mount a pre-downloaded copy at /app/models/qwen3-vl or set VLM_MODEL_ID
# to the HuggingFace repo ID and it will download on first inference.

# Source is bind-mounted at runtime for live reload (see docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
