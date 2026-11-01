"""
01_download_data.py
Politomorphism — Defensive Political Anthropology layer
Step 1: Download and inspect a propaganda-detection corpus.

We use PropXplain (QCRI): a multilingual, *explainable* propaganda-detection
dataset. Each example carries a propaganda label AND a natural-language
explanation of why the text is (or is not) propagandistic. We load the
English configuration, inspect its structure, and save a local copy so
later steps do not depend on the network.

Nothing is trained here. We only acquire and look at the data.
"""

from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Keep terminal output readable.
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 140)

# Folder where we store a local copy of the data.
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Downloading the PropXplain corpus (English) from Hugging Face...")
    print("(First run may take a minute while it fetches the files.)\n")

    # PropXplain has two configurations: "english" and "arabic".
    # We work in English. trust_remote_code=True is required because this
    # dataset ships a small loading script on the Hub.
    dataset = load_dataset("QCRI/PropXplain", "english", trust_remote_code=True)

    # Show which splits (train/dev/test) came with the dataset.
    print("Splits available in this dataset:")
    for split_name in dataset:
        print(f"  - {split_name}: {len(dataset[split_name])} rows")
    print()

    # Pick the first available split for inspection (usually 'train').
    first_split_name = list(dataset.keys())[0]
    split = dataset[first_split_name]

    # Show the schema: what fields each example has.
    print(f"Columns (features) in split '{first_split_name}':")
    for feature_name in split.features:
        print(f"  - {feature_name}: {split.features[feature_name]}")
    print()

    # Convert to a pandas DataFrame so we can look at it comfortably.
    df = split.to_pandas()

    print(f"Total examples in '{first_split_name}': {len(df)}\n")
    print("First 3 examples (full rows):")
    for i in range(min(3, len(df))):
        print(f"\n--- Example {i} ---")
        for col in df.columns:
            value = str(df.iloc[i][col])
            if len(value) > 300:
                value = value[:300] + " [...]"
            print(f"{col}: {value}")
    print()

    # Save a local copy of every split as CSV for later steps.
    for split_name in dataset:
        out_path = DATA_DIR / f"propxplain_english_{split_name}.csv"
        dataset[split_name].to_pandas().to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved: {out_path.resolve()}")

    print("\nStep 1 complete. Data acquired and inspected.")


if __name__ == "__main__":
    main()