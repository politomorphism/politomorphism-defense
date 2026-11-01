"""
04_transformer_clean.py
Corrected transformer run on an ARTICLE-DISJOINT split (fixes the leakage the
audit found in the official PropXplain English split). Build the clean split
from the three official CSVs in data/, fine-tune DistilBERT, evaluate on the
clean test set, and SAVE per-paragraph scores so a proper comparison
(PR/AP, matched precision, paired transitions) can be computed later.

Run this on your machine (it can reach Hugging Face). It prints the corrected
metrics and writes clean/transformer_test_scores.csv — send that file back.
"""
import os, re, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, average_precision_score)
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

DATA = Path("data"); OUT = Path("clean"); OUT.mkdir(exist_ok=True)
TEXT, LAB, SID = "input", "Label", "sample_id"
POS = "propagandistic"; LABELS = ["non-propagandistic", "propagandistic"]
label2id = {n: i for i, n in enumerate(LABELS)}; id2label = {i: n for i, n in enumerate(LABELS)}
MODEL_NAME = "distilbert-base-uncased"; MAX_LENGTH = 256; EPOCHS = 3
TRAIN_BATCH = 8; EVAL_BATCH = 16; LR = 2e-5; SEED = 42

def load(split):
    df = pd.read_csv(DATA / f"propxplain_english_{split}.csv")
    df[LAB] = df[LAB].astype(str).str.strip()
    df = df[df[LAB].isin(LABELS)].copy(); df[TEXT] = df[TEXT].astype(str)
    return df[[SID, TEXT, LAB]]

def clean_split():
    pool = pd.concat([load("train"), load("validation"), load("test")], ignore_index=True)
    pool["_n"] = pool[TEXT].map(lambda s: re.sub(r"\s+", " ", s.strip().lower()))
    pool = pool.drop_duplicates("_n").reset_index(drop=True)
    pool["article"] = pool[SID].map(lambda s: s.rsplit("_", 1)[0] if "_" in str(s) else str(s))
    y = (pool[LAB] == POS).astype(int).values; g = pool["article"].values
    i_trf, i_te = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(pool, y, g))
    trf = pool.iloc[i_trf].reset_index(drop=True)
    i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.12, random_state=SEED).split(trf, None, trf["article"].values))
    return trf.iloc[i_tr].reset_index(drop=True), trf.iloc[i_va].reset_index(drop=True), pool.iloc[i_te].reset_index(drop=True)

class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k): super().__init__(*a, **k); self.cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs); logits = out.logits
        w = self.cw.to(logits.device) if self.cw is not None else None
        loss = nn.CrossEntropyLoss(weight=w)(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, out) if return_outputs else loss

def main():
    set_seed(SEED)
    tr, va, te = clean_split()
    print(f"clean split  train={len(tr)} val={len(va)} test={len(te)} "
          f"(test prop={int((te[LAB]==POS).sum())})")
    counts = tr[LAB].value_counts(); n = len(tr); k = len(LABELS)
    cw = torch.tensor([n/(k*int(counts.get(x,1))) for x in LABELS], dtype=torch.float)

    def ds(d):
        d = d.copy(); d["labels"] = d[LAB].map(label2id).astype(int)
        return Dataset.from_pandas(d[[TEXT, "labels"]], preserve_index=False)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id)
    def tk(b): return tok(b[TEXT], truncation=True, max_length=MAX_LENGTH)
    tr_t = ds(tr).map(tk, batched=True, remove_columns=[TEXT])
    te_t = ds(te).map(tk, batched=True, remove_columns=[TEXT])
    args = TrainingArguments(output_dir="transformer_clean_out", num_train_epochs=EPOCHS,
                             per_device_train_batch_size=TRAIN_BATCH, per_device_eval_batch_size=EVAL_BATCH,
                             learning_rate=LR, logging_steps=50, save_strategy="no", report_to="none", seed=SEED)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr_t, eval_dataset=te_t,
                              data_collator=DataCollatorWithPadding(tokenizer=tok), class_weights=cw)
    print("Training on the CLEAN split (slow on CPU)..."); trainer.train()
    out = trainer.predict(te_t); logits = out.predictions
    if isinstance(logits, tuple): logits = logits[0]
    # positive-class probability via softmax
    z = torch.tensor(logits); prob = torch.softmax(z, dim=-1)[:, 1].numpy()
    pred_ids = logits.argmax(-1)
    yte = (te[LAB].values == POS).astype(int)
    yhat = np.array([id2label[int(i)] for i in pred_ids])
    P = precision_score(te[LAB], yhat, pos_label=POS, zero_division=0)
    R = recall_score(te[LAB], yhat, pos_label=POS, zero_division=0)
    F = f1_score(te[LAB], yhat, pos_label=POS, zero_division=0)
    cm = confusion_matrix(te[LAB], yhat, labels=LABELS)
    ap = average_precision_score(yte, prob)
    print("\n=== CLEAN transformer (article-disjoint) ===")
    print(f"precision={P:.3f} recall={R:.3f} F1={F:.3f} AP={ap:.3f}")
    print(f"confusion [[TN,FP],[FN,TP]] = {cm.tolist()}  flagged={int((yhat==POS).sum())}/{len(te)}")
    pd.DataFrame({"sample_id": te[SID], "article": te["article"], "y_true": yte,
                  "transformer_score": prob, "transformer_pred": (yhat==POS).astype(int)}
                 ).to_csv(OUT / "transformer_test_scores.csv", index=False)
    json.dump({"precision": float(P), "recall": float(R), "f1": float(F),
               "average_precision": float(ap),
               "confusion_tn_fp_fn_tp": [int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])],
               "n_test": int(len(te))}, open(OUT/"transformer_clean_metrics.json","w"), indent=2)
    print("\nSaved clean/transformer_test_scores.csv — send this file back.")

if __name__ == "__main__":
    main()
