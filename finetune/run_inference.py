"""
Run the fine-tuned safety model on a single image.
Usage: python run_inference.py --image /path/to/image.jpg
"""
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

ADAPTER_PATH = "/workspace/adapter1"
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to image (e.g. /workspace/Benchmarking Vids/frame.jpg)")
    p.add_argument("--adapter", default=ADAPTER_PATH, help="Path to adapter folder")
    args = p.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.adapter)

    print("Loading base model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    print("Loading image...")
    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this construction site image. Are there any safety rule violations?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    print("Running model...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    response = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("\n--- Response ---")
    print(response.strip())


if __name__ == "__main__":
    main()
