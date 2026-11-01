"""
04_transformer_detector.py
Politomorphism - Defensive Political Anthropology layer
Step 4: A contextual transformer detector (DistilBERT), fine-tuned on the
same PropXplain English data as the baseline and evaluated identically.

This is the successor to 03_baseline_detector.py. It replaces the lexical
TF-IDF features with a context-aware transformer, so it can represent the
structural manipulation (presupposition, insinuation, rhetorical
equivalence) that the lexical baseline misses. Metrics are computed exactly
as in the baseline for a fair comparison, with precision / recall / F1 on
the 'propagandistic' class as the defensively relevant figures.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ---- Configuration ----
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("data")
TEXT_COL = "input"
LABEL_COL = "Label"
LABELS = ["non-propagandistic", "propagandistic"]  # index 0, 1
POSITIVE_LABEL = "propagandistic"
MAX_LENGTH = 256
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
LEARNING_RATE = 2e-5
SEED = 42

label2id = {name: i for i, name in enumerate(LABELS)}
id2label = {i: name for i, name in enumerate(LABELS)}


def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"propxplain_english_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 01_download_data.py first to create the data/ CSVs."
        )
    df = pd.read_csv(path)[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL].isin(LABELS)]
    return df


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to counter class imbalance,
    mirroring the 'balanced' weighting used in the baseline."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    set_seed(SEED)

    print("Loading data...")
    train_df = load_split("train")
    test_df = load_split("test")
    print(f"  train rows: {len(train_df)}")
    print(f"  test rows:  {len(test_df)}")

    # Balanced class weights: n_samples / (n_classes * count_c)
    counts = train_df[LABEL_COL].value_counts()
    n = len(train_df)
    k = len(LABELS)
    weights = [n / (k * int(counts.get(name, 1))) for name in LABELS]
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  class weights {LABELS} = {[round(w, 3) for w in weights]}")

    def to_dataset(df: pd.DataFrame) -> Dataset:
        d = df.copy()
        d["labels"] = d[LABEL_COL].map(label2id).astype(int)
        return Dataset.from_pandas(d[[TEXT_COL, "labels"]], preserve_index=False)

    train_ds = to_dataset(train_df)
    test_ds = to_dataset(test_df)

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    print("(First run downloads the model, ~250 MB.)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=[TEXT_COL])
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=[TEXT_COL])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="transformer_out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=collator,
        class_weights=class_weights,
    )

    print("\nTraining the transformer detector (this is the slow part on CPU)...")
    trainer.train()
    print("Training done.\n")

    print("Evaluating on the held-out test split...\n")
    pred_output = trainer.predict(test_tok)
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = np.argmax(logits, axis=-1)
    y_pred = [id2label[int(i)] for i in pred_ids]
    y_true = [id2label[int(i)] for i in pred_output.label_ids]

    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=3))

    p = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    print(f"Positive class = '{POSITIVE_LABEL}'")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1:        {f:.3f}\n")

    labels_order = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    print(f"Confusion matrix (rows = true, cols = predicted), order = {labels_order}:")
    print(cm)

    print("\nStep 4 complete. Transformer detector trained and evaluated.")
    print("Compare the propagandistic-class F1 against the baseline's 0.506.")


if __name__ == "__main__":
    main()
