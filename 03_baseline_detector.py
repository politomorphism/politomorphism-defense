"""
03_baseline_detector.py
Politomorphism — Defensive Political Anthropology layer
Step 3: A transparent baseline propaganda detector.

Pipeline:
  text -> TF-IDF features -> Logistic Regression (class-balanced)

We train on the training split and evaluate on the held-out test split.
Because the classes are imbalanced (far more non-propagandistic than
propagandistic), we report precision / recall / F1 for the POSITIVE
class ('propagandistic'), a confusion matrix, and sample predictions.
Accuracy alone would be misleading here, so we do not rely on it.
"""

from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path("data")

TEXT_COL = "input"
LABEL_COL = "Label"
POSITIVE_LABEL = "propagandistic"


def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"propxplain_english_{split_name}.csv"
    df = pd.read_csv(path)
    # Keep only the two columns we need, drop rows with missing text/label.
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    return df


def main() -> None:
    print("Loading data...")
    train_df = load_split("train")
    test_df = load_split("test")
    print(f"  train rows: {len(train_df)}")
    print(f"  test rows:  {len(test_df)}\n")

    X_train = train_df[TEXT_COL].tolist()
    y_train = train_df[LABEL_COL].tolist()
    X_test = test_df[TEXT_COL].tolist()
    y_test = test_df[LABEL_COL].tolist()

    # Build a transparent pipeline: TF-IDF + Logistic Regression.
    # class_weight="balanced" counteracts the class imbalance so the model
    # is not rewarded for simply always predicting the majority class.
    model = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,                # ignore terms appearing in < 2 docs
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    print("Training the baseline detector...")
    model.fit(X_train, y_train)
    print("Training done.\n")

    print("Evaluating on the held-out test split...\n")
    y_pred = model.predict(X_test)

    # Full per-class report.
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Focused metrics on the positive (propaganda) class.
    p = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    r = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    print(f"Positive class = '{POSITIVE_LABEL}'")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1:        {f:.3f}\n")

    # Confusion matrix with explicit label order.
    labels_order = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    print(f"Confusion matrix (rows = true, cols = predicted), order = {labels_order}:")
    print(cm)
    print()

    # A few concrete predictions so we can read what the model does.
    print("Sample predictions from the test set:")
    shown = 0
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        if shown >= 6:
            break
        snippet = text[:120] + (" [...]" if len(text) > 120 else "")
        mark = "OK " if true_label == pred_label else "XX "
        print(f"  [{mark}] true={true_label:18s} pred={pred_label:18s} | {snippet}")
        shown += 1

    print("\nStep 3 complete. Baseline detector trained and evaluated.")


if __name__ == "__main__":
    main()