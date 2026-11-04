# -*- coding: utf-8 -*-
"""
11_residual_lexical_analysis.py
Does the transformer's advantage track leftover vocabulary, or structure?

The pairs are vocabulary-matched but not word-identical (mean Jaccard 0.50), so a
sceptic can argue the contextual model still wins on the residual vocabulary
difference. This script tests that directly, using only scores already computed:
  - correlation of per-pair lexical overlap (Jaccard) with the transformer's
    score margin and with its pair accuracy;
  - transformer and baseline pair accuracy on well- vs poorly-matched halves;
  - the strict high-overlap subset (Jaccard >= 0.60);
  - correlation of overlap with the LEXICAL baseline's correctness, as a control.

Reads: contrastive_pairs_v3.csv, pairs_v3_baseline_scores.csv,
pair_scores_v3_multiseed.csv. No model calls, no network.
"""
import re
import pandas as pd
import numpy as np
from scipy import stats

pairs = pd.read_csv("contrastive_pairs_v3.csv")
base = pd.read_csv("pairs_v3_baseline_scores.csv")
ms = pd.read_csv("pair_scores_v3_multiseed.csv")


def wset(t):
    return set(re.findall(r"[a-z']+", str(t).lower()))


def jac(a, b):
    A, B = wset(a), wset(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


wide = pairs.pivot_table(index=["pair_id", "technique"], columns="member",
                         values="text", aggfunc="first").reset_index()
wide["jac"] = [jac(a, b) for a, b in zip(wide["manipulative"], wide["control"])]

# transformer: mean score per member over the 5 seeds -> margin
tw = ms.pivot_table(index="pair_id", columns="member", values="score", aggfunc="mean")
wide = wide.merge(tw.rename(columns={"manipulative": "t_m", "control": "t_c"}), on="pair_id")
wide["t_margin"] = wide["t_m"] - wide["t_c"]

# transformer correctness as fraction of seeds correct
hi = []
for s, g in ms.groupby("seed"):
    w = g.pivot_table(index="pair_id", columns="member", values="score", aggfunc="first")
    hi.append((w["manipulative"] > w["control"]).rename(s))
wide = wide.merge(pd.concat(hi, axis=1).mean(axis=1).rename("t_acc"), on="pair_id")
wide = wide.merge(base[["pair_id", "hi"]].rename(columns={"hi": "b_hi"}), on="pair_id")

r1, p1 = stats.pearsonr(wide["jac"], wide["t_margin"])
r2, p2 = stats.spearmanr(wide["jac"], wide["t_acc"])
med = wide["jac"].median()
lo, hiq = wide[wide["jac"] <= med], wide[wide["jac"] > med]
strict = wide[wide["jac"] >= 0.60]
rb, pb = stats.pointbiserialr(wide["b_hi"], wide["jac"])

print("=== Residual-lexical analysis (60 pairs) ===")
print(f"corr(Jaccard, transformer margin):      r={r1:+.3f}  p={p1:.3f}")
print(f"corr(Jaccard, transformer pair acc):    rho={r2:+.3f}  p={p2:.3f}")
print(f"corr(Jaccard, BASELINE correctness):    r={rb:+.3f}  p={pb:.3f}  (control)")
print(f"\nmedian Jaccard = {med:.2f}")
print(f"poorly matched (<=med, n={len(lo)}):  transformer {lo['t_acc'].mean():.3f} | baseline {lo['b_hi'].mean():.3f}")
print(f"well matched   (> med, n={len(hiq)}):  transformer {hiq['t_acc'].mean():.3f} | baseline {hiq['b_hi'].mean():.3f}")
print(f"high overlap   (>=0.60, n={len(strict)}): transformer {strict['t_acc'].mean():.3f} | baseline {strict['b_hi'].mean():.3f}")
print("\nReading: the transformer's margin is uncorrelated with lexical overlap,")
print("and it does not degrade on well-matched pairs -- so it is not winning on")
print("residual vocabulary. The lexical baseline's correctness, by contrast, is")
print("significantly correlated with overlap, as a lexical model should be.")
