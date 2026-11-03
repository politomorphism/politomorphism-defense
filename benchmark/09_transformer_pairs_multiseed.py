"""
09_transformer_pairs_multiseed.py
Multi-seed run of the contrastive-pairs benchmark (v0.3, 60 pairs).

Why: the v0.3 result (transformer pair accuracy 0.917 vs lexical baseline
0.633, disjoint CIs) came from a single training seed. One honest limit in the
protocol is "one model at one seed." This script removes that limit. It trains
DistilBERT on the SAME clean article-disjoint PropXplain split five times, each
time with a different training seed (model initialisation and data order), and
scores the 60 contrastive pairs after every run. That gives five pair
accuracies, from which we report the mean, the standard deviation, and a 95%
confidence interval on the transformer's pair accuracy across seeds -- and the
margin over the fixed lexical baseline (0.633).

The data split is held fixed (seed 42) so every run scores the pairs under the
same conditions as the published 0.917; only the training randomness varies.

Run on your machine (needs Hugging Face). About 35 min per seed on CPU, so
roughly 3 hours for all five. It saves progress after each seed, so a seed that
finishes is never lost if you stop and restart. Put this file and
contrastive_pairs_v3.csv in the project folder, next to the data/ folder.

Outputs:
  pair_scores_v3_multiseed.csv   per-member scores for every seed
                                 (columns: seed, pair_id, member, score)
  pairs_multiseed_summary.csv    one row per seed with its pair accuracy
It prints the across-seed mean, sd, and 95% CI, and the margin over 0.633.
"""
import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

DATA = Path("data"); PAIRS = Path("contrastive_pairs_v3.csv")
TEXT, LAB, SID = "input", "Label", "sample_id"; POS = "propagandistic"
LABELS = ["non-propagandistic", "propagandistic"]
label2id = {n: i for i, n in enumerate(LABELS)}; id2label = {i: n for i, n in enumerate(LABELS)}
MODEL = "distilbert-base-uncased"; MAXLEN = 256; EPOCHS = 3; TB = 8; EB = 16; LR = 2e-5
SPLIT_SEED = 42                      # data split held fixed across runs
SEEDS = [13, 21, 42, 101, 202]       # training seeds that vary
BASELINE_PAIR_ACC = 0.633            # fixed lexical baseline on the 60-pair set

SCORES_OUT = Path("pair_scores_v3_multiseed.csv")
SUMMARY_OUT = Path("pairs_multiseed_summary.csv")


def load(s):
    d = pd.read_csv(DATA / f"propxplain_english_{s}.csv"); d[LAB] = d[LAB].astype(str).str.strip()
    d = d[d[LAB].isin(LABELS)].copy(); d[TEXT] = d[TEXT].astype(str); return d[[SID, TEXT, LAB]]


def clean_train():
    pool = pd.concat([load("train"), load("validation"), load("test")], ignore_index=True)
    pool["_n"] = pool[TEXT].map(lambda s: re.sub(r"\s+", " ", s.strip().lower()))
    pool = pool.drop_duplicates("_n").reset_index(drop=True)
    pool["article"] = pool[SID].map(lambda s: s.rsplit("_", 1)[0] if "_" in str(s) else str(s))
    y = (pool[LAB] == POS).astype(int).values; g = pool["article"].values
    itrf, _ = next(GroupShuffleSplit(1, test_size=0.15, random_state=SPLIT_SEED).split(pool, y, g))
    trf = pool.iloc[itrf].reset_index(drop=True)
    itr, _ = next(GroupShuffleSplit(1, test_size=0.12, random_state=SPLIT_SEED).split(trf, None, trf["article"].values))
    return trf.iloc[itr].reset_index(drop=True)


class WT(Trainer):
    def __init__(self, *a, cw=None, **k): super().__init__(*a, **k); self.cw = cw
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("labels"); o = model(**inputs); lg = o.logits
        w = self.cw.to(lg.device) if self.cw is not None else None
        loss = nn.CrossEntropyLoss(weight=w)(lg.view(-1, model.config.num_labels), y.view(-1))
        return (loss, o) if return_outputs else loss


def ds(df, textcol, tok):
    d = df.copy()
    if LAB in d: d["labels"] = d[LAB].map(label2id).astype(int)
    cols = [textcol] + (["labels"] if "labels" in d else [])
    return Dataset.from_pandas(d[cols].rename(columns={textcol: TEXT}), preserve_index=False).map(
        lambda b: tok(b[TEXT], truncation=True, max_length=MAXLEN), batched=True, remove_columns=[TEXT])


def run_seed(seed, tr, cw, pairs):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label=id2label, label2id=label2id)
    args = TrainingArguments(output_dir=f"pairs_out_seed{seed}", num_train_epochs=EPOCHS,
        per_device_train_batch_size=TB, per_device_eval_batch_size=EB, learning_rate=LR,
        logging_steps=200, save_strategy="no", report_to="none", seed=seed)
    trainer = WT(model=model, args=args, train_dataset=ds(tr, TEXT, tok),
                 data_collator=DataCollatorWithPadding(tokenizer=tok), cw=cw)
    print(f"[seed {seed}] training on the clean split...")
    trainer.train()
    scored = pairs[["pair_id", "member", "text"]].copy()
    out = trainer.predict(ds(scored, "text", tok)); lg = out.predictions
    lg = lg[0] if isinstance(lg, tuple) else lg
    scored["score"] = torch.softmax(torch.tensor(lg), dim=-1)[:, 1].numpy()
    scored["seed"] = seed
    return scored[["seed", "pair_id", "member", "score"]]


def pair_accuracy(scored_one_seed):
    w = scored_one_seed.pivot_table(index="pair_id", columns="member", values="score", aggfunc="first")
    return float((w["manipulative"] > w["control"]).mean())


def main():
    tr = clean_train()
    counts = tr[LAB].value_counts(); n = len(tr); k = len(LABELS)
    cw = torch.tensor([n / (k * int(counts.get(x, 1))) for x in LABELS], dtype=torch.float)
    pairs = pd.read_csv(PAIRS)

    done = {}
    if SCORES_OUT.exists():  # resume: keep seeds already finished
        prev = pd.read_csv(SCORES_OUT)
        for s, g in prev.groupby("seed"):
            done[int(s)] = g[["seed", "pair_id", "member", "score"]]
        if done: print(f"Resuming; already have seeds {sorted(done)}")

    all_scores = list(done.values())
    for seed in SEEDS:
        if seed in done:
            continue
        sc = run_seed(seed, tr, cw, pairs)
        all_scores.append(sc)
        pd.concat(all_scores, ignore_index=True).to_csv(SCORES_OUT, index=False)
        print(f"[seed {seed}] pair accuracy = {pair_accuracy(sc):.3f}  (saved)")

    merged = pd.concat(all_scores, ignore_index=True)
    rows = [{"seed": s, "pair_accuracy": pair_accuracy(g)} for s, g in merged.groupby("seed")]
    summary = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary.to_csv(SUMMARY_OUT, index=False)

    accs = summary["pair_accuracy"].values
    m = accs.mean(); sd = accs.std(ddof=1); ns = len(accs)
    se = sd / ns ** 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print("\n=== Multi-seed transformer, 60-pair v0.3 benchmark ===")
    print(summary.to_string(index=False))
    print(f"\nSeeds: {ns}")
    print(f"Transformer pair accuracy: mean {m:.3f}, sd {sd:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Lexical baseline (fixed):  {BASELINE_PAIR_ACC:.3f}")
    print(f"Margin over baseline:      {m - BASELINE_PAIR_ACC:+.3f}")
    print(f"CI excludes baseline?      {'YES' if lo > BASELINE_PAIR_ACC else 'NO'}")
    print(f"\nSaved {SCORES_OUT.name} and {SUMMARY_OUT.name}.")


if __name__ == "__main__":
    main()
