"""
Pre-download the ConstructionSite 10k dataset to the Hugging Face cache.
Run this after installing dependencies and logging in (huggingface-cli login).
Accept the dataset terms at https://huggingface.co/datasets/LouisChen15/ConstructionSite if required.
"""
from datasets import load_dataset

DATASET_ID = "LouisChen15/ConstructionSite"

if __name__ == "__main__":
    print(f"Downloading dataset {DATASET_ID} (train split)...")
    load_dataset(DATASET_ID, split="train")
    print("Dataset cached. You can run training next.")
