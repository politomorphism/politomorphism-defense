"""
02_inspect_labels.py
Politomorphism — Defensive Political Anthropology layer
Step 2: Inspect label values and class balance across splits.

We read the local CSV copies saved in step 1 and report, for each split:
 - how many rows it has
 - the exact label values present and how many of each
This tells us the shape of the classification problem before we model it.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

SPLITS = ["train", "validation", "test"]


def main() -> None:
    for split_name in SPLITS:
        path = DATA_DIR / f"propxplain_english_{split_name}.csv"
        if not path.exists():
            print(f"[skip] {path} not found.")
            continue

        df = pd.read_csv(path)

        print(f"=== Split: {split_name} ===")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Find the label column regardless of capitalisation.
        label_col = None
        for candidate in ["Label", "label"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            print("No 'Label' column found.\n")
            continue

        print(f"Label column: '{label_col}'")
        print("Label value counts:")
        counts = df[label_col].value_counts(dropna=False)
        for value, count in counts.items():
            pct = 100.0 * count / len(df)
            print(f"  {value!r}: {count} ({pct:.1f}%)")
        print()


if __name__ == "__main__":
    main()