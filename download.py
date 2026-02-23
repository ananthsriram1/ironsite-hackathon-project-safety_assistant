import os
from huggingface_hub import hf_hub_download, snapshot_download


HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set. Please add it to your .env file or shell env.")


hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir="models",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)


snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir="models/qwen3-vl",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)
                                                                               