"""
evaluate_pairs.py
Contrastive-pairs benchmark for STRUCTURAL manipulation sensitivity.

Each pair has a manipulative member and a control member matched on topic and
vocabulary; only the rhetorical STRUCTURE differs. A model that detects
structure should score the manipulative member higher. Because vocabulary is
matched, a purely lexical model should be near chance (pair accuracy ~0.5).

Primary metric: pair accuracy = fraction of pairs where
    score(manipulative) > score(control).
We also report a lexical-overlap control (how similar the two members' words
are) to show the pairs are genuinely matched, so a high pair accuracy cannot be
dismissed as the model keying on different vocabulary.

This script scores the pairs with the TF-IDF + logistic-regression baseline
trained on PropXplain (same config as the papers). A transformer variant that
loads per-member scores from a CSV is included for the model run done on a
machine that can reach Hugging Face.
"""
from pathlib import Path
import sys, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA = Path("data"); PAIRS = Path("contrastive_pairs_v3.csv")
TEXT, LAB, POS = "input", "Label", "propagandistic"

def load_propx(split):
    d = pd.read_csv(DATA / f"propxplain_english_{split}.csv")
    d[LAB] = d[LAB].astype(str).str.strip()
    d = d[d[LAB].isin(["propagandistic", "non-propagandistic"])].copy()
    d[TEXT] = d[TEXT].astype(str)
    return d

def train_baseline():
    tr = load_propx("train")
    m = Pipeline([("v", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
                  ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    m.fit(tr[TEXT], tr[LAB])
    pc = list(m.classes_).index(POS)
    return lambda texts: m.predict_proba(list(texts))[:, pc]

def word_set(t):
    return set(re.findall(r"[a-z']+", str(t).lower()))

def jaccard(a, b):
    A, B = word_set(a), word_set(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0

def evaluate(score_fn, pairs):
    wide = pairs.pivot_table(index=["pair_id","technique"], columns="member",
                             values="text", aggfunc="first").reset_index()
    man = score_fn(wide["manipulative"].tolist())
    con = score_fn(wide["control"].tolist())
    wide["score_manip"] = man; wide["score_control"] = con
    wide["manip_higher"] = wide["score_manip"] > wide["score_control"]
    wide["lexical_jaccard"] = [jaccard(a,b) for a,b in zip(wide["manipulative"], wide["control"])]
    return wide

def report(wide, name):
    acc = wide["manip_higher"].mean()
    n = len(wide)
    # binomial 95% CI (normal approx) around pair accuracy
    se = (acc*(1-acc)/n)**0.5
    lo, hi = max(0, acc-1.96*se), min(1, acc+1.96*se)
    print(f"\n=== {name} ===")
    print(f"Pairs: {n} | mean lexical Jaccard between members: {wide['lexical_jaccard'].mean():.2f} "
          f"(1.0 = identical words; high = well matched)")
    print(f"Pair accuracy (manipulative scored higher): {acc:.3f}  [~95% CI {lo:.2f}, {hi:.2f}]")
    print("By technique:")
    for tech, g in wide.groupby("technique"):
        print(f"  {tech:26s} {g['manip_higher'].mean():.2f}  ({int(g['manip_higher'].sum())}/{len(g)})")
    print("\nInterpretation: ~0.50 means the model cannot tell structure apart when")
    print("vocabulary is matched; well above 0.50 means it is sensitive to structure.")

def main():
    pairs = pd.read_csv(PAIRS)
    # Option A: transformer scores supplied as CSV (pair_id, member, score)
    if len(sys.argv) > 1:
        sc = pd.read_csv(sys.argv[1])
        key = {(r.pair_id, r.member): r.score for r in sc.itertuples()}
        def score_fn(texts, _idx=[0]): raise SystemExit  # not used in CSV mode
        wide = pairs.pivot_table(index=["pair_id","technique"], columns="member",
                                 values="text", aggfunc="first").reset_index()
        wide["score_manip"] = [key[(pid,"manipulative")] for pid in wide["pair_id"]]
        wide["score_control"] = [key[(pid,"control")] for pid in wide["pair_id"]]
        wide["manip_higher"] = wide["score_manip"] > wide["score_control"]
        wide["lexical_jaccard"] = [jaccard(a,b) for a,b in zip(wide["manipulative"], wide["control"])]
        report(wide, f"Model from {sys.argv[1]}")
        return
    # Option B: train + score the lexical baseline here
    print("Training TF-IDF + logistic-regression baseline on PropXplain...")
    score_fn = train_baseline()
    wide = evaluate(score_fn, pairs)
    report(wide, "TF-IDF + Logistic Regression (lexical baseline)")
    wide.to_csv("pairs_baseline_scores.csv", index=False)

if __name__ == "__main__":
    main()
