"""
Inspect LouisChen15/ConstructionSite columns and a few samples.
Run before training to confirm column names and answer formatting.
"""
from datasets import load_dataset

def main():
    ds = load_dataset("LouisChen15/ConstructionSite", split="train")
    print("Column names:", ds.column_names)
    print("\nFirst example keys and types:")
    ex = ds[0]
    for k, v in ex.items():
        t = type(v).__name__
        if hasattr(v, "shape"):
            t += f" shape={getattr(v, 'shape', None)}"
        elif isinstance(v, (list, dict)) and len(str(v)) > 80:
            t += f" len={len(v)}"
        print(f"  {k}: {t}")
    print("\nRule violations (reason text):")
    for i in range(1, 5):
        key = f"rule_{i}_violation"
        val = ex.get(key)
        print(f"  {key}: {val}")
    print("\nimage_caption (first 200 chars):", (ex.get("image_caption") or "")[:200])

if __name__ == "__main__":
    main()
