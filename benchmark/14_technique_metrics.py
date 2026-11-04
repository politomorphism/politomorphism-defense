# -*- coding: utf-8 -*-
"""
14_technique_metrics.py
Score the technique classifier's predictions against ground truth.

Reads technique_predictions.csv (produced by 13_technique_classifier.py) and
reports: naming accuracy on manipulative members, per-technique recall, the
false-device rate on plain controls, and the confusion matrix. No model calls.
"""
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TECH = ["presupposition", "manufactured_equivalence", "false_dilemma",
        "whataboutism", "loaded_rhetorical_question", "implied_causation"]
LABELS = TECH + ["none"]

d = pd.read_csv("technique_predictions.csv").dropna(subset=["pred"])
man = d[d.member == "manipulative"]
con = d[d.member == "control"]

acc = (man.true == man.pred).mean()
print(f"(1) Naming the device on {len(man)} manipulative members: accuracy {acc:.3f}")
for t in TECH:
    g = man[man.true == t]
    if len(g):
        print(f"    {t:28s} recall {(g.true==g.pred).mean():.2f}  ({int((g.true==g.pred).sum())}/{len(g)})")

none_ok = (con.pred == "none").mean()
print(f"\n(2) Controls correctly 'none': {none_ok:.3f}  ({int((con.pred=='none').sum())}/{len(con)})")
print(f"    false-device rate: {1-none_ok:.3f}  ({int((con.pred!='none').sum())}/{len(con)})")
print("    devices wrongly assigned to controls:",
      con[con.pred != 'none'].pred.value_counts().to_dict())

print("\n(3) 7-way report:")
print(classification_report(d.true, d.pred, labels=LABELS, zero_division=0))
print("Confusion matrix (rows=true, cols=pred); order:", ", ".join(LABELS))
print(confusion_matrix(d.true, d.pred, labels=LABELS))
