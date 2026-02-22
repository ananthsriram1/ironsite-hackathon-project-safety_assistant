FROM python:3.11

WORKDIR /app

COPY requirements.txt .
# ultralytics pulls in opencv-python (needs libGL); swap it for headless after install
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir "opencv-python-headless>=4.6.0"

# Download SAM3 weights into the models directory at build time
ARG HF_TOKEN
RUN mkdir -p /app/models && \
    HF_TOKEN=${HF_TOKEN} python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir='/app/models', token='${HF_TOKEN}')"

# Source is bind-mounted at runtime for live reload (see docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
