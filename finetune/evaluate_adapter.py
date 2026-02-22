import torch
import re
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_ID = "LouisChen15/ConstructionSite"
SYSTEM_PROMPT = (
    "You are an expert construction site safety inspector. "
    "You analyze images of construction sites to identify safety rule violations, "
    "describe site conditions, and locate workers, equipment, and hazards. "
    "When violations are found, cite the specific rule, explain the violation, "
    "and provide the bounding box [x_min, y_min, x_max, y_max] in normalized coordinates."
)

# CHANGE THIS TO TEST YOUR DIFFERENT MODELS
# Example: "./qwen-safety-finetuned/final_adapter" or "./qwen-safety-8x-finetuned/final_adapter"
ADAPTER_PATH = "./qwen-safety-finetuned/final_adapter"
NUM_TEST_SAMPLES = 100 # How many images to evaluate (increase for more robust score)
TEST_SPLIT_RATIO = 0.05
RANDOM_SEED = 42
DEBUG_SAMPLES = 5

# ==========================================
# 2. SCORING MATH: Intersection over Union (IoU)
# ==========================================
def calculate_iou(boxA, boxB):
    """Calculate IoU for two bounding boxes formatted as [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ==========================================
# 3. PARSING LLM OUTPUT
# ==========================================
def extract_bboxes(text):
    """Finds all bounding boxes and normalizes to [0..1000] int coordinates."""
    pattern = r"[\[\(]\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\]\)]"
    matches = re.findall(pattern, text)
    boxes = []
    for m in matches:
        coords = [float(v) for v in m]
        # If model emits normalized coords [0..1], rescale to [0..1000].
        if all(0.0 <= c <= 1.0 for c in coords):
            coords = [c * 1000.0 for c in coords]
        x1, y1, x2, y2 = [int(round(c)) for c in coords]
        # Canonicalize + clamp.
        x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
        y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes

def extract_rules(text):
    """Finds which rules (1-4) the model flagged as violations."""
    rules_found = set()
    lowered = text.lower()
    for match in re.findall(r"rule\s*#?\s*:?\s*([1-4])", text, flags=re.IGNORECASE):
        rules_found.add(int(match))
    # Fallback semantic mapping when the model doesn't explicitly write "Rule X".
    if any(k in lowered for k in ["ppe", "hard hat", "safety vest", "safety glasses", "protective clothing"]):
        rules_found.add(1)
    if any(k in lowered for k in ["harness", "fall protection", "working at height", "3m"]):
        rules_found.add(2)
    if any(k in lowered for k in ["edge protection", "guardrail", "fence", "excavation", "underground"]):
        rules_found.add(3)
    if any(k in lowered for k in ["excavator", "blind spot", "operating radius", "swing radius"]):
        rules_found.add(4)
    return rules_found

def get_ground_truth(example):
    """Extracts ground truth bounding boxes and rules from the raw dataset."""
    gt_rules = set()
    gt_bboxes = []
    for i in range(1, 5):
        key = f"rule_{i}_violation"
        val = example.get(key)
        if val and isinstance(val, dict):
            gt_rules.add(i)
            # Scale GT [0.0 - 1.0] to [0 - 1000] to match model output
            for bbox in val.get("bounding_box", []):
                gt_bboxes.append([int(c * 1000) for c in bbox])
    return gt_rules, gt_bboxes

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def main():
    print(f"Loading base model and adapter from {ADAPTER_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    except Exception:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Attention: {attn_impl}")

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    # Load the fine-tuned weights on top of the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Loading test dataset...")
    raw_train = load_dataset(DATASET_ID, split="train")
    split = raw_train.train_test_split(test_size=TEST_SPLIT_RATIO, seed=RANDOM_SEED)
    test_ds = split["test"]
    n = min(NUM_TEST_SAMPLES, len(test_ds))
    dataset = test_ds.select(range(n))
    print(f"Using deterministic holdout: {n} samples from test split (seed={RANDOM_SEED}).")

    total_iou = 0.0
    iou_count = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    parsed_rule_samples = 0
    parsed_bbox_samples = 0

    print("Starting evaluation...")
    for idx, example in enumerate(tqdm(dataset)):
        image = example["image"].convert("RGB")

        # Format the strict evaluation prompt
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this construction site image. Are there any safety rule violations? If yes, state which rule is violated, explain briefly, and provide the bounding box [x_min, y_min, x_max, y_max]."}
            ]}
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=[text_prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        # Trim prompt from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        pred_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # --- PARSE & SCORE ---
        pred_rules = extract_rules(pred_text)
        pred_bboxes = extract_bboxes(pred_text)
        gt_rules, gt_bboxes = get_ground_truth(example)
        if pred_rules:
            parsed_rule_samples += 1
        if pred_bboxes:
            parsed_bbox_samples += 1

        if idx < DEBUG_SAMPLES:
            print(f"\n[DEBUG sample {idx}]")
            print(f"pred_rules={sorted(pred_rules)} gt_rules={sorted(gt_rules)}")
            print(f"pred_bboxes={pred_bboxes[:2]} gt_bboxes={gt_bboxes[:2]}")
            print(f"pred_text={pred_text[:350].replace(chr(10), ' ')}")

        # 1. Rule Classification Scoring
        for rule in pred_rules:
            if rule in gt_rules:
                true_positives += 1
            else:
                false_positives += 1
        for rule in gt_rules:
            if rule not in pred_rules:
                false_negatives += 1

        # 2. Bounding Box IoU Scoring (Greedy Match)
        for p_box in pred_bboxes:
            best_iou = 0.0
            for g_box in gt_bboxes:
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou
            iou_count += 1

    # ==========================================
    # 5. FINAL METRICS
    # ==========================================
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / iou_count if iou_count > 0 else 0.0

    print("\n" + "="*40)
    print(f"📊 EVALUATION RESULTS FOR: {ADAPTER_PATH}")
    print("="*40)
    print(f"Rule Detection Precision : {precision:.4f} (When it says there's a violation, is it right?)")
    print(f"Rule Detection Recall    : {recall:.4f} (Did it catch all the actual violations?)")
    print(f"Rule Detection F1 Score  : {f1_score:.4f} (Overall rule detection health)")
    print(f"Mean Bounding Box IoU    : {mean_iou:.4f} (How perfectly do the boxes align?)")
    print(f"Samples with parsed rules: {parsed_rule_samples}/{len(dataset)}")
    print(f"Samples with parsed boxes: {parsed_bbox_samples}/{len(dataset)}")
    print("="*40)

if __name__ == "__main__":
    main()
