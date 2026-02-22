"""
Step 1: Prepare the dataset ONCE before training.
Run this with plain python (NOT torchrun):
    python prepare_data.py

Saves formatted dataset to ./dataset_cache/ so train.py can load it instantly.
"""
import random
from datasets import load_dataset

DATASET_ID = "LouisChen15/ConstructionSite"
CACHE_DIR = "./dataset_cache"
EVAL_RATIO = 0.05

SYSTEM_PROMPT = (
    "You are an expert construction site safety inspector. "
    "You analyze images of construction sites to identify safety rule violations, "
    "describe site conditions, and locate workers, equipment, and hazards. "
    "When violations are found, cite the specific rule, explain the violation, "
    "and provide the bounding box [x_min, y_min, x_max, y_max] in normalized coordinates."
)

RULE_DESCRIPTIONS = {
    1: "Rule 1: Use of basic PPE (hard hats, safety glasses, vests, protective clothing).",
    2: "Rule 2: Use of safety harness when working ≥3m high without edge protection.",
    3: "Rule 3: Edge protection (guardrails, fences) for underground projects ≥3m depth.",
    4: "Rule 4: No workers within excavator blind spots or operating radius.",
}

SAFETY_PROMPTS = [
    "Analyze this construction site image. Are there any safety rule violations? If yes, state which rule is violated, explain the issue, and provide the bounding box of the violator.",
    "Inspect this construction site for safety compliance. Check for PPE violations, fall protection issues, edge protection, and excavator proximity hazards. Report any findings with locations.",
    "Review this image for construction safety violations. The rules to check are: (1) PPE usage, (2) safety harness at heights, (3) edge protection for excavations, (4) excavator proximity. What do you find?",
    "As a safety inspector, examine this construction site image. Identify any rule violations, describe the violation, and provide bounding box coordinates for the violator.",
    "Check this construction site image for OSHA-style safety violations. Report each violation with the rule number, a brief explanation, and the location in the image.",
]

CAPTION_PROMPTS = [
    "Describe this construction site image in detail, including workers, equipment, and site conditions.",
    "Provide a detailed description of what you see in this construction site image.",
    "What is happening in this construction site image? Describe all visible workers, machinery, and conditions.",
]

GROUNDING_PROMPTS = [
    "Locate all excavators, rebar, and workers with white hard hats in this image. Provide bounding boxes in [x_min, y_min, x_max, y_max] format.",
    "Identify and locate the following objects in this construction site: excavators, rebar, and workers wearing white hard hats. Give bounding box coordinates.",
]


def format_bbox(bbox_list):
    if not bbox_list:
        return "none detected"
    parts = []
    for bbox in bbox_list:
        coords = [str(int(c * 1000)) for c in bbox]
        parts.append(f"[{', '.join(coords)}]")
    return ", ".join(parts)


def build_safety_answer(example):
    parts = []
    for i in range(1, 5):
        val = example.get(f"rule_{i}_violation")
        if val is None:
            continue
        if isinstance(val, dict):
            reason = val.get("reason", "")
            bbox = val.get("bounding_box", [])
            line = f"{RULE_DESCRIPTIONS[i]} VIOLATION DETECTED."
            if reason:
                line += f" {reason}"
            if bbox:
                line += f" Bounding box: {format_bbox(bbox)}."
            parts.append(line)
    if not parts:
        return "No safety rule violations detected in this image."
    return " ".join(parts)


def build_caption_answer(example):
    return example.get("image_caption", "A construction site image.")


def build_grounding_answer(example):
    parts = [
        f"Excavators: {format_bbox(example.get('excavator', []))}.",
        f"Rebar: {format_bbox(example.get('rebar', []))}.",
        f"Workers with white hard hats: {format_bbox(example.get('worker_with_white_hard_hat', []))}.",
    ]
    return " ".join(parts)


def format_example(example, idx):
    rng = random.Random(42 + idx)
    roll = rng.random()
    if roll < 0.60:
        user_text = rng.choice(SAFETY_PROMPTS)
        answer = build_safety_answer(example)
    elif roll < 0.85:
        user_text = rng.choice(CAPTION_PROMPTS)
        answer = build_caption_answer(example)
    else:
        user_text = rng.choice(GROUNDING_PROMPTS)
        answer = build_grounding_answer(example)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return {"messages": messages}


def main():
    print(f"Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Loaded {len(ds)} samples.")

    cols_to_remove = [c for c in ds.column_names if c != "image"]
    print("Formatting (multi-task: safety VQA + captioning + grounding)...")
    formatted = ds.map(format_example, with_indices=True, remove_columns=cols_to_remove, num_proc=4, desc="Format")

    split = formatted.train_test_split(test_size=EVAL_RATIO, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    print(f"Saving to {CACHE_DIR}/...")
    split.save_to_disk(CACHE_DIR)
    print("Done. Now run: python train.py  (or torchrun --nproc_per_node=N train.py)")


if __name__ == "__main__":
    main()
