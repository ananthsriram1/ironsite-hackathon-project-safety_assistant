"""
Run the fine-tuned Qwen3-VL safety model on a single image.
Usage:
  python inference.py --image path/to/image.jpg
  python inference.py --image path/to/image.jpg --adapter ./qwen-safety-finetuned/final_adapter
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


def main():
    p = argparse.ArgumentParser(description="Run safety inspection on a construction site image.")
    p.add_argument("--image", type=str, required=True, help="Path to a construction site image.")
    p.add_argument("--adapter", type=str, default="./qwen-safety-finetuned/final_adapter", help="Path to LoRA adapter.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model ID.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(args.adapter)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Same prompt as in training; image passed separately to processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Analyze this construction site image. Are there any safety rule violations regarding PPE, edge protection, or heavy machinery? If yes, state which rule and briefly explain.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        images=[image],
        text=[text],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response_ids = out_ids[:, input_len:]
    response = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("Response:", response.strip())


if __name__ == "__main__":
    main()
