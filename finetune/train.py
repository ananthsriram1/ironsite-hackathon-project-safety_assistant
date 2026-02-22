"""
Fine-tune Qwen3-VL-8B-Instruct on ConstructionSite 10k with LoRA.

PREREQUISITE: run  python prepare_data.py  first to create ./dataset_cache/

Launch options:
  python train.py                                    (1 GPU, guaranteed to work)
  accelerate launch --num_processes=8 train.py       (8 GPUs via accelerate)
"""
import os
import sys
import traceback
import torch
from datasets import DatasetDict
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "./qwen-safety-finetuned"
DATASET_CACHE_DIR = "./dataset_cache"


def get_collate_fn(processor, image_token_id, vision_start_id, vision_end_id):
    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            for ex in examples
        ]
        images = [[ex["image"].convert("RGB")] for ex in examples]
        batch = processor(images=images, text=texts, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == vision_start_id] = -100
        labels[labels == vision_end_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    # Detect if we're running under accelerate/torchrun or plain python
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    # For plain `python train.py`, pin to GPU 0
    if world_size == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if is_main:
        print(f"GPUs: {world_size}, local_rank: {local_rank}")

    # ---- Load pre-processed dataset from disk ----
    if not os.path.exists(DATASET_CACHE_DIR):
        print(f"ERROR: {DATASET_CACHE_DIR} not found. Run `python prepare_data.py` first.")
        sys.exit(1)

    dataset = DatasetDict.load_from_disk(DATASET_CACHE_DIR)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    if is_main:
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ---- Load processor ----
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    # ---- Load model ----
    if is_main:
        print("Loading model...")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    if is_main:
        print(f"Attention: {attn_impl}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" if world_size == 1 else None,
        attn_implementation=attn_impl,
    )

    image_token_id = getattr(model.config, "image_token_id", 151655)
    vision_start_id = getattr(model.config, "vision_start_token_id", 151652)
    vision_end_id = getattr(model.config, "vision_end_token_id", 151653)

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # ---- Training config ----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,

        bf16=True,
        tf32=True,

        learning_rate=2e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        num_train_epochs=3,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        eval_strategy="steps",
        eval_steps=100,

        logging_steps=5,
        logging_first_step=True,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/logs",

        optim="adamw_torch_fused",

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,

        remove_unused_columns=False,
    )

    collate_fn = get_collate_fn(processor, image_token_id, vision_start_id, vision_end_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    if is_main:
        eff = world_size * 1 * 16
        print(f"Starting training (effective batch size: {eff})")
        print(f"TensorBoard: tensorboard --logdir {OUTPUT_DIR}/logs")

    trainer.train()

    if is_main:
        trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
        processor.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
        print(f"Saved to {OUTPUT_DIR}/final_adapter")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
