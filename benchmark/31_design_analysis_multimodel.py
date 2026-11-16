#!/usr/bin/env python3
"""Design analysis for the multi-model CoCoLoFa pre-registration (draft v3).

Simulates the probability of each verdict of the pre-registered decision rule,
using only the writer structure of the frozen 440-comment manifest and writer
heterogeneity estimated from the public gpt-4o-mini (M0) predictions. No model
is called and no comment text is read.

Usage (from the repository root):
    python benchmark/31_design_analysis_multimodel.py
Writes benchmark/design_analysis_multimodel.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark" / "cocolofa_fd_sample_manifest.csv"
M0 = ROOT / "benchmark" / "cocolofa_fd_predictions.csv"
OUT = ROOT / "benchmark" / "design_analysis_multimodel.csv"

SEED = 20261116
N_SIM = 2000          # simulated studies per scenario
N_BOOT = 2000         # writer-bootstrap replicates per simulated study
                      # (the confirmatory analysis uses 10,000; 2,000 suffices
                      #  to estimate verdict probabilities)
HI, LO = 0.75, 0.60   # decision-rule thresholds (draft v3, Table 2)
G_THR = 0.10          # S1 threshold for the detection gap

B_TARGETS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]   # typical-writer recall in B
C_TARGETS = [0.70, 0.80, 0.90]                       # typical-writer accuracy in C
A_TARGET = 0.94                                      # typical-writer flag rate in A


# ---------------------------------------------------------------- data
man = pd.read_csv(MANIFEST)
writers = np.sort(man["worker_id"].unique())
widx = {w: i for i, w in enumerate(writers)}
n_w = len(writers)
counts = (
    man.groupby(["worker_id", "cell"]).size().unstack(fill_value=0)
    .reindex(writers).fillna(0).astype(int)
)
nA, nB, nC = counts["A"].values, counts["B"].values, counts["C"].values

m0 = pd.read_csv(M0)
m0["correct"] = ((m0.flag_fd == m0.gold_false_dilemma) & (m0.abstain == 0)).astype(int)


# ------------------------------------------------- writer heterogeneity
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()


def neg_loglik(sigma, k, n, mu):
    p = expit(mu + sigma * GH_X[None, :])                      # writers x nodes
    lik = (p ** k[:, None]) * ((1 - p) ** (n - k)[:, None])
    return -np.sum(np.log(lik @ GH_W + 1e-300))


def fit_sigma(cell, outcome):
    d = m0[m0.cell == cell].groupby("worker_id")[outcome].agg(["sum", "size"])
    k, n = d["sum"].values.astype(float), d["size"].values.astype(float)
    best = None
    for mu in np.linspace(-3, 4, 141):
        r = minimize_scalar(lambda s: neg_loglik(s, k, n, mu), bounds=(0.0, 5.0), method="bounded")
        if best is None or r.fun < best[0]:
            best = (r.fun, mu, r.x)
    return best[2]


SIG_B = fit_sigma("B", "correct")
SIG_C = fit_sigma("C", "correct")
SIG_A = fit_sigma("A", "flag_fd")


# ---------------------------------------------------------- bootstrap
rng = np.random.default_rng(SEED)
draws = rng.integers(0, n_w, size=(N_BOOT, n_w))
W = np.zeros((N_BOOT, n_w))
np.add.at(W, (np.repeat(np.arange(N_BOOT), n_w), draws.ravel()), 1.0)
WB, WC, WA = W @ nB, W @ nC, W @ nA
valid = (WB > 0) & (WC > 0) & (WA > 0)   # replicates with an empty cell are dropped


def interval(num, den):
    v = num[valid] / den[valid][:, None]
    return np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)


def verdicts(ba, lo, hi):
    sup = (ba >= HI) & (lo > LO)
    nsup = (ba <= LO) & (hi < HI)
    return sup, nsup, ~(sup | nsup)


def simulate(muB, muC, sB, sC, sA, n_sim):
    pB = expit(logit(muB) + sB * rng.standard_normal((n_w, n_sim)))
    pC = expit(logit(muC) + sC * rng.standard_normal((n_w, n_sim)))
    pA = expit(logit(A_TARGET) + sA * rng.standard_normal((n_w, n_sim)))
    kB = rng.binomial(nB[:, None], pB)
    kC = rng.binomial(nC[:, None], pC)
    kA = rng.binomial(nA[:, None], pA)
    true_ba = 0.5 * ((nB[:, None] * pB).sum(0) / nB.sum() + (nC[:, None] * pC).sum(0) / nC.sum())
    true_g = (nA[:, None] * pA).sum(0) / nA.sum() - (nB[:, None] * pB).sum(0) / nB.sum()
    ba = 0.5 * (kB.sum(0) / nB.sum() + kC.sum(0) / nC.sum())
    recB = (W @ kB)[valid] / WB[valid][:, None]
    accC = (W @ kC)[valid] / WC[valid][:, None]
    bab = 0.5 * (recB + accC)
    lo, hi = np.percentile(bab, 2.5, axis=0), np.percentile(bab, 97.5, axis=0)
    g = kA.sum(0) / nA.sum() - kB.sum(0) / nB.sum()
    gb = (W @ kA)[valid] / WA[valid][:, None] - recB
    glo, ghi = np.percentile(gb, 2.5, axis=0), np.percentile(gb, 97.5, axis=0)
    sup, nsup, inc = verdicts(ba, lo, hi)
    return {
        "true_BA": true_ba.mean(), "true_G": true_g.mean(),
        "P_SUPPORTED": sup.mean(), "P_NOT_SUPPORTED": nsup.mean(), "P_INCONCLUSIVE": inc.mean(),
        "median_CI_width": np.median(hi - lo),
        "P_G_PRESENT": (glo > G_THR).mean(), "P_G_BELOW": (ghi < G_THR).mean(),
    }


# ------------------------------------------------------- M0 check
k0B = np.zeros(n_w); k0C = np.zeros(n_w)
for w, c, ok in m0[["worker_id", "cell", "correct"]].itertuples(index=False):
    if c == "B":
        k0B[widx[w]] += ok
    elif c == "C":
        k0C[widx[w]] += ok
ba0 = 0.5 * (k0B.sum() / nB.sum() + k0C.sum() / nC.sum())
b0 = 0.5 * ((W @ k0B)[valid] / WB[valid] + (W @ k0C)[valid] / WC[valid])
m0_lo, m0_hi = np.percentile(b0, [2.5, 97.5])

rows = []
for het in ("estimated", "none"):
    sB, sC, sA = (SIG_B, SIG_C, SIG_A) if het == "estimated" else (0.0, 0.0, 0.0)
    for muB in B_TARGETS:
        for muC in C_TARGETS:
            r = simulate(muB, muC, sB, sC, sA, N_SIM)
            rows.append({"heterogeneity": het, "typical_B": muB, "typical_C": muC, **r})
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False, float_format="%.4f")

print(f"Writers: {n_w}; writers in B: {(nB > 0).sum()}; largest writer in B: {nB.max()} of {nB.sum()}")
print(f"Writer heterogeneity estimated from M0 (logit SD): B {SIG_B:.2f}, C {SIG_C:.2f}, A {SIG_A:.2f}")
print(f"Bootstrap replicates kept: {valid.sum()} of {N_BOOT} (others had an empty cell)")
print(f"M0 check: BA {ba0:.3f}, writer-clustered 95% interval {m0_lo:.3f}-{m0_hi:.3f}")
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
