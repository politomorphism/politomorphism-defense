# -*- coding: utf-8 -*-
"""
24_lexical_baseline.py
The experiment the review demanded: a purely lexical baseline (TF-IDF + logistic
regression) on the SAME items, to test whether the detector's results reflect
argumentative STRUCTURE or merely VOCABULARY. No API, no model calls.

Two tests:
  (A) LOGIC false_dilemma eval set (269 items): lexical 5-fold CV vs the detector's
      reported F1 0.895. If the lexical model nearly matches, the external LOGIC
      result alone does not demonstrate structural sensitivity.
  (B) Contrastive pairs (60 pairs, vocabulary held ~constant within a pair):
      pair-level accuracy for the detector (from saved predictions) vs a lexical
      baseline with GROUPED cross-validation (whole pairs held out). If the
      contrastive instrument truly isolates structure, the lexical model should
      do much worse than the detector here.

Honest note: a lexical baseline doing WELL is not a bug in the baseline; it is
evidence about what the evaluation can and cannot prove (cf. Vamvas & Sennrich,
2021, on lexical leakage in minimal-pair evaluation).
"""
import csv, os
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import precision_recall_fscore_support

TECH = {"presupposition","manufactured_equivalence","false_dilemma",
        "whataboutism","loaded_rhetorical_question","implied_causation"}

def lexclf():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"))

def test_logic(path, seed=20261107):
    rows=[r for r in csv.DictReader(open(path,encoding="utf-8")) if r.get("to_review")=="0"]
    X=[r["text"] for r in rows]; y=np.array([int(r["gold_false_dilemma"]) for r in rows])
    pred=np.zeros(len(y),int)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        c=lexclf(); c.fit([X[i] for i in tr],y[tr]); pred[te]=c.predict([X[i] for i in te])
    p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    print("(A) LOGIC false_dilemma, same items (n=%d, pos=%d)" % (len(y),y.sum()))
    print(f"    LEXICAL baseline: precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    print("    DETECTOR (reported): F1 0.895")
    print(f"    -> gap {0.895-f:+.3f}; a bag-of-words model nearly matches, so the")
    print("       external result alone does NOT prove structural sensitivity.\n")

def test_contrastive(v3_path, pred_path):
    v3=list(csv.DictReader(open(v3_path,encoding="utf-8")))
    pr=list(csv.DictReader(open(pred_path,encoding="utf-8")))
    if len(v3)!=len(pr):
        print("contrastive: row mismatch, skipping"); return
    pairs=defaultdict(dict)
    for a,b in zip(v3,pr):
        pairs[a["pair_id"]][a["member"]]={"det_flag":1 if b["pred"] in TECH else 0}
    full=[pid for pid,m in pairs.items() if "manipulative" in m and "control" in m]
    dpac=sum(1 for pid in full if pairs[pid]["manipulative"]["det_flag"]==1
             and pairs[pid]["control"]["det_flag"]==0)
    # lexical, grouped CV by pair
    X=[r["text"] for r in v3]; y=np.array([1 if r["member"]=="manipulative" else 0 for r in v3])
    groups=[r["pair_id"] for r in v3]; lp=np.zeros(len(y),int)
    for tr,te in GroupKFold(5).split(X,y,groups):
        c=lexclf(); c.fit([X[i] for i in tr],y[tr]); lp[te]=c.predict([X[i] for i in te])
    byid=defaultdict(dict)
    for r,v in zip(v3,lp): byid[r["pair_id"]][r["member"]]=v
    lpac=sum(1 for pid in full if byid[pid]["manipulative"]==1 and byid[pid]["control"]==0)
    n=len(full)
    print("(B) Contrastive pairs, pair-level (n=%d pairs, vocabulary ~constant within a pair)" % n)
    print(f"    DETECTOR pair accuracy: {dpac}/{n} = {dpac/n:.3f}")
    print(f"    LEXICAL pair accuracy (grouped CV): {lpac}/{n} = {lpac/n:.3f}")
    print("    -> if the lexical model matches or beats the detector here, the")
    print("       instrument has a lexical confound and structural sensitivity is")
    print("       NOT demonstrated (cf. Vamvas & Sennrich 2021).")

if __name__ == "__main__":
    here=os.path.dirname(os.path.abspath(__file__))
    logic=os.path.join(here,"logic_false_dilemma_eval.csv")
    if os.path.exists(logic): test_logic(logic)
    else: print("(A) skipped: logic_false_dilemma_eval.csv not present\n")
    test_contrastive(os.path.join(here,"contrastive_pairs_v3.csv"),
                     os.path.join(here,"contrastive_pairs_v3_v2_predictions.csv"))
