"""
Clean, article-disjoint re-evaluation of the TF-IDF + LogReg baseline.
Fixes the leakage found by the audit: the official PropXplain English split
scatters paragraphs from the same article across train/val/test. We pool all
rows, split by ARTICLE (no article spans two sets), then refit and evaluate.
Everything is decided before looking at test performance; seed is fixed.
"""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, average_precision_score, precision_recall_curve)

DATA = Path("data"); OUT = Path("clean"); OUT.mkdir(exist_ok=True)
TEXT, LAB, SID = "input", "Label", "sample_id"
POS = "propagandistic"
SEED = 42

def load(split):
    df = pd.read_csv(DATA / f"propxplain_english_{split}.csv")
    df[LAB] = df[LAB].astype(str).str.strip()
    df = df[df[LAB].isin(["propagandistic", "non-propagandistic"])].copy()
    df[TEXT] = df[TEXT].astype(str)
    return df[[SID, TEXT, LAB]]

pool = pd.concat([load("train"), load("validation"), load("test")], ignore_index=True)
# drop exact-text duplicates (keep first) so identical paragraphs can't straddle sets
pool["_n"] = pool[TEXT].map(lambda s: re.sub(r"\s+", " ", s.strip().lower()))
pool = pool.drop_duplicates("_n").reset_index(drop=True)
pool["article"] = pool[SID].map(lambda s: s.rsplit("_", 1)[0] if "_" in str(s) else str(s))
y = (pool[LAB] == POS).astype(int).values
groups = pool["article"].values
print(f"pooled unique-text rows: {len(pool)} | articles: {pool['article'].nunique()} | positives: {y.sum()}")

# article-disjoint split: ~15% test, ~10% val, ~75% train (by articles)
idx_tr_full, idx_test = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(pool, y, groups))
pool_trf = pool.iloc[idx_tr_full].reset_index(drop=True)
g2 = pool_trf["article"].values
idx_tr, idx_val = next(GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=SEED).split(pool_trf, None, g2))
tr = pool_trf.iloc[idx_tr].reset_index(drop=True)
va = pool_trf.iloc[idx_val].reset_index(drop=True)
te = pool.iloc[idx_test].reset_index(drop=True)

# sanity: no article overlap
assert not (set(tr.article) & set(te.article)), "article leak tr/te"
assert not (set(va.article) & set(te.article)), "article leak va/te"
assert not (set(tr.article) & set(va.article)), "article leak tr/va"

for nm, d in [("train", tr), ("val", va), ("test", te)]:
    vc = d[LAB].value_counts()
    d.to_csv(OUT / f"clean_{nm}.csv", index=False)
    print(f"{nm}: rows={len(d)} prop={vc.get(POS,0)} nonprop={vc.get('non-propagandistic',0)} articles={d['article'].nunique()}")

# ---- fit baseline (identical settings to the paper) ----
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
model.fit(tr[TEXT], tr[LAB])
pos_col = list(model.classes_).index(POS)

def scores(d):
    return model.predict_proba(d[TEXT])[:, pos_col]
te_s, va_s = scores(te), scores(va)
yte = (te[LAB] == POS).astype(int).values
yva = (va[LAB] == POS).astype(int).values

# default rule (predict)
yhat = model.predict(te[TEXT])
P = precision_score(te[LAB], yhat, pos_label=POS, zero_division=0)
R = recall_score(te[LAB], yhat, pos_label=POS, zero_division=0)
F = f1_score(te[LAB], yhat, pos_label=POS, zero_division=0)
cm = confusion_matrix(te[LAB], yhat, labels=["non-propagandistic", POS])
acc = (yhat == te[LAB].values).mean()
ap = average_precision_score(yte, te_s)
print("\n=== CLEAN baseline (article-disjoint), default predict ===")
print(f"precision={P:.3f} recall={R:.3f} F1={F:.3f} accuracy={acc:.3f}")
print(f"confusion [[TN,FP],[FN,TP]] = {cm.tolist()}")
print(f"average precision (AP, test) = {ap:.3f}")
print(f"flagged = {int((yhat==POS).sum())}/{len(te)}")

# matched-precision thresholds selected on val, applied to test
def thr_for_precision(yv, sv, target):
    p, r, t = precision_recall_curve(yv, sv)
    best = None
    for i in range(len(t)):
        if p[i] >= target:
            if best is None or r[i] > best[1]:
                best = (t[i], r[i])
    return best[0] if best else None

print("\nmatched-precision (threshold chosen on val, applied to test):")
for tgt in (0.5, 0.6, 0.7):
    th = thr_for_precision(yva, va_s, tgt)
    if th is None:
        print(f"  target P>={tgt}: infeasible on val"); continue
    pred = (te_s >= th).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    print(f"  target P>={tgt}: test P={pr:.3f} R={rc:.3f} flagged={int(pred.sum())} (thr={th:.3f})")

# save scores for later paired analysis with the transformer
pd.DataFrame({"sample_id": te[SID], "article": te.article, "y_true": yte,
              "baseline_score": te_s, "baseline_pred": (yhat == POS).astype(int)}
             ).to_csv(OUT / "baseline_test_scores.csv", index=False)
pd.DataFrame({"sample_id": va[SID], "article": va.article, "y_true": yva,
              "baseline_score": va_s}).to_csv(OUT / "baseline_val_scores.csv", index=False)
json.dump({"precision": P, "recall": R, "f1": F, "accuracy": float(acc),
           "confusion_tn_fp_fn_tp": [int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])],
           "average_precision": float(ap), "n_test": len(te),
           "n_train": len(tr), "n_val": len(va)},
          open(OUT / "baseline_clean_metrics.json", "w"), indent=2)
print("\nsaved clean split + baseline scores to clean/")
