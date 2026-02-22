"""
Pre-download Qwen3-VL-8B-Instruct and its processor to the Hugging Face cache.
Run this after installing dependencies and logging in (huggingface-cli login).
"""
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

if __name__ == "__main__":
    print(f"Downloading processor for {MODEL_ID}...")
    AutoProcessor.from_pretrained(MODEL_ID)
    print("Downloading model (this may take a while)...")
    Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model and processor cached. You can run training next.")
