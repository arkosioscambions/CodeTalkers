import json
import random
from datasets import load_dataset

def sample_hf_dataset(dataset_name, split, output_prefix, ratios, seed=42):
    random.seed(seed)
    ds = load_dataset(dataset_name, split=split)

    for ratio in ratios:
        # Shuffle once per ratio (with fixed seed for reproducibility)
        sampled = ds.shuffle(seed=seed).select(range(int(len(ds) * ratio)))
        output_path = f"{output_prefix}_{int(ratio*100)}pct.jsonl"
        with open(output_path, "w") as f:
            for entry in sampled:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {output_path} ({len(sampled)} samples)")

if __name__ == "__main__":
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    # OSS-Instruct-75K dataset
    sample_hf_dataset(
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "train",
        "oss_instruct",
        ratios,
        seed=42
    )

    # Evol-Instruct-110K dataset
    sample_hf_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "train",
        "evol_instruct",
        ratios,
        seed=42
    )
