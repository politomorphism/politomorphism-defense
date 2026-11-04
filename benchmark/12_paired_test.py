# -*- coding: utf-8 -*-
"""
12_paired_test.py
Paired significance test of transformer vs lexical baseline on the 60 pairs.

Non-overlapping confidence intervals are an informal comparison. Because both
models are scored on the SAME sixty pairs, the right test is paired: McNemar on
pair-level correctness, plus a bootstrap CI on the paired margin. Uses the
headline transformer (seed 42) and the lexical baseline, from existing scores.
No model calls, no network.
"""
import pandas as pd
import numpy as np
from scipy import stats

base = pd.read_csv("pairs_v3_baseline_scores.csv")[["pair_id", "hi"]].rename(columns={"hi": "b"})
ms = pd.read_csv("pair_scores_v3_multiseed.csv")
g = ms[ms["seed"] == 42]
w = g.pivot_table(index="pair_id", columns="member", values="score", aggfunc="first")
tf = (w["manipulative"] > w["control"]).astype(int).rename("t").reset_index()
d = base.merge(tf, on="pair_id")
n = len(d)

ba, ta = d["b"].mean(), d["t"].mean()
b_only = int(((d["b"] == 1) & (d["t"] == 0)).sum())
t_only = int(((d["b"] == 0) & (d["t"] == 1)).sum())
both = int(((d["b"] == 1) & (d["t"] == 1)).sum())
none = int(((d["b"] == 0) & (d["t"] == 0)).sum())
disc = b_only + t_only
p_exact = stats.binomtest(min(b_only, t_only), disc, 0.5).pvalue if disc else 1.0

rng = np.random.default_rng(42)
diffs = d["t"].values - d["b"].values
boot = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"n pairs = {n}")
print(f"baseline {ba:.3f} | transformer(seed42) {ta:.3f} | margin {ta-ba:+.3f}")
print(f"both correct {both} | transformer-only {t_only} | baseline-only {b_only} | both wrong {none}")
print(f"exact McNemar p = {p_exact:.2e}  (discordant {disc})")
print(f"bootstrap 95% CI on margin: [{lo:+.3f}, {hi:+.3f}]  (excludes 0: {lo > 0})")
