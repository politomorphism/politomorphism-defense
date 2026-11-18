#!/usr/bin/env python3
"""Design analysis for the multi-model CoCoLoFa pre-registration (draft v4).

Draft v4 changes the primary outcome from comment-weighted to writer-equal
hard-cell balanced accuracy: in each hard cell, the proportion correct is first
computed per writer and then averaged over writers with equal weight. The
detection gap G is defined the same way (writer-equal flag rate in A minus
writer-equal flag rate in B).

This script simulates the probability of each verdict under the v4 rule and,
on the same simulated data, under the v3 (comment-weighted) rule, so that the
two can be compared. It uses only the writer structure of the frozen 440-comment
manifest and writer heterogeneity estimated from the public gpt-4o-mini (M0)
predictions. No model is called and no comment text is read.

Usage (from the repository root):
    python benchmark/32_design_analysis_writer_equal.py
Writes benchmark/design_analysis_writer_equal.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark" / "cocolofa_fd_sample_manifest.csv"
M0 = ROOT / "benchmark" / "cocolofa_fd_predictions.csv"
OUT = ROOT / "benchmark" / "design_analysis_writer_equal.csv"

SEED = 20261118
N_SIM = 2000          # simulated studies per scenario
N_BOOT = 2000         # writer-bootstrap replicates per simulated study
                      # (the confirmatory analysis uses 10,000)
HI, LO = 0.75, 0.60   # decision-rule thresholds (unchanged from v3)
G_THR = 0.10          # S1 threshold for the detection gap (unchanged)

B_TARGETS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]   # median-writer recall in B
C_TARGETS = [0.70, 0.80, 0.90]                       # median-writer accuracy in C
A_TARGET = 0.94                                      # median-writer flag rate in A


# ---------------------------------------------------------------- data
man = pd.read_csv(MANIFEST)
writers = np.sort(man["worker_id"].unique())
widx = {w: i for i, w in enumerate(writers)}
n_w = len(writers)
counts = (
    man.groupby(["worker_id", "cell"]).size().unstack(fill_value=0)
    .reindex(writers).fillna(0).astype(int)
)
nA, nB, nC = (counts[c].values.astype(float) for c in ("A", "B", "C"))
IA, IB, IC = (nA > 0).astype(float), (nB > 0).astype(float), (nC > 0).astype(float)

m0 = pd.read_csv(M0)
m0["correct"] = ((m0.flag_fd == m0.gold_false_dilemma) & (m0.abstain == 0)).astype(int)


# ------------------------------------------------- writer heterogeneity
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()


def neg_loglik(sigma, k, n, mu):
    p = expit(mu + sigma * GH_X[None, :])
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


def pop_mean(mu_prob, sigma):
    """Mean over the writer population of expit(logit(mu_prob) + sigma*Z)."""
    return float(expit(logit(mu_prob) + sigma * GH_X) @ GH_W)


SIG_B = fit_sigma("B", "correct")
SIG_C = fit_sigma("C", "correct")
SIG_A = fit_sigma("A", "flag_fd")


# ---------------------------------------------------------- bootstrap
rng = np.random.default_rng(SEED)
draws = rng.integers(0, n_w, size=(N_BOOT, n_w))
W = np.zeros((N_BOOT, n_w))
np.add.at(W, (np.repeat(np.arange(N_BOOT), n_w), draws.ravel()), 1.0)
valid = ((W @ IB) > 0) & ((W @ IC) > 0) & ((W @ IA) > 0)   # replicates with an empty cell dropped
Wv = W[valid]


def comment_weighted(k, n):
    """Point estimate and bootstrap replicates, each comment weighted equally."""
    n = np.asarray(n, dtype=float).ravel()
    return k.sum(0) / n.sum(), (Wv @ k) / (Wv @ n)[:, None]


def writer_equal(k, n, ind):
    """Point estimate and bootstrap replicates, each writer weighted equally."""
    rate = np.where(n[:, None] > 0, k / np.maximum(n, 1)[:, None], 0.0)
    return (rate * ind[:, None]).sum(0) / ind.sum(), (Wv @ rate) / (Wv @ ind)[:, None]


def verdicts(ba, lo, hi):
    sup = (ba >= HI) & (lo > LO)
    nsup = (ba <= LO) & (hi < HI)
    return sup, nsup, ~(sup | nsup)


def summarise(prefix, ba, bab, g, gb):
    lo, hi = np.percentile(bab, 2.5, axis=0), np.percentile(bab, 97.5, axis=0)
    glo, ghi = np.percentile(gb, 2.5, axis=0), np.percentile(gb, 97.5, axis=0)
    sup, nsup, inc = verdicts(ba, lo, hi)
    return {
        f"{prefix}_P_SUPPORTED": sup.mean(), f"{prefix}_P_NOT_SUPPORTED": nsup.mean(),
        f"{prefix}_P_INCONCLUSIVE": inc.mean(), f"{prefix}_median_CI_width": np.median(hi - lo),
        f"{prefix}_P_G_PRESENT": (glo > G_THR).mean(), f"{prefix}_P_G_BELOW": (ghi < G_THR).mean(),
    }


def simulate(muB, muC, sB, sC, sA, n_sim):
    pB = expit(logit(muB) + sB * rng.standard_normal((n_w, n_sim)))
    pC = expit(logit(muC) + sC * rng.standard_normal((n_w, n_sim)))
    pA = expit(logit(A_TARGET) + sA * rng.standard_normal((n_w, n_sim)))
    kB = rng.binomial(nB.astype(int)[:, None], pB).astype(float)
    kC = rng.binomial(nC.astype(int)[:, None], pC).astype(float)
    kA = rng.binomial(nA.astype(int)[:, None], pA).astype(float)

    out = {
        # estimand of v4: the writer-population quantity
        "pop_BA": 0.5 * (pop_mean(muB, sB) + pop_mean(muC, sC)),
        "pop_G": pop_mean(A_TARGET, sA) - pop_mean(muB, sB),
    }
    # v3: comment-weighted
    rB, rBb = comment_weighted(kB, nB); rC, rCb = comment_weighted(kC, nC)
    rA, rAb = comment_weighted(kA, nA)
    out.update(summarise("v3", 0.5 * (rB + rC), 0.5 * (rBb + rCb), rA - rB, rAb - rBb))
    # v4: writer-equal
    eB, eBb = writer_equal(kB, nB, IB); eC, eCb = writer_equal(kC, nC, IC)
    eA, eAb = writer_equal(kA, nA, IA)
    out.update(summarise("v4", 0.5 * (eB + eC), 0.5 * (eBb + eCb), eA - eB, eAb - eBb))
    return out


# ------------------------------------------------------- M0 check
def m0_counts(cell, col):
    k = np.zeros(n_w)
    for w, c, v in m0[["worker_id", "cell", col]].itertuples(index=False):
        if c == cell:
            k[widx[w]] += v
    return k[:, None]


k0B, k0C, k0A = m0_counts("B", "correct"), m0_counts("C", "correct"), m0_counts("A", "flag_fd")
k0Bflag = m0_counts("B", "flag_fd")
m0_rows = {}
for name, fn in (("comment-weighted (v3)", None), ("writer-equal (v4)", writer_equal)):
    if fn is None:
        b, bb = comment_weighted(k0B, nB); c, cb = comment_weighted(k0C, nC)
        a, ab = comment_weighted(k0A, nA); f, fb = comment_weighted(k0Bflag, nB)
    else:
        b, bb = writer_equal(k0B, nB, IB); c, cb = writer_equal(k0C, nC, IC)
        a, ab = writer_equal(k0A, nA, IA); f, fb = writer_equal(k0Bflag, nB, IB)
    ba, bab = 0.5 * (b + c), 0.5 * (bb + cb)
    g, gb = a - f, ab - fb
    m0_rows[name] = (b[0], c[0], ba[0], *np.percentile(bab, [2.5, 97.5]), g[0], *np.percentile(gb, [2.5, 97.5]))

rows = []
for het in ("estimated", "none"):
    sB, sC, sA = (SIG_B, SIG_C, SIG_A) if het == "estimated" else (0.0, 0.0, 0.0)
    for muB in B_TARGETS:
        for muC in C_TARGETS:
            rows.append({"heterogeneity": het, "median_B": muB, "median_C": muC,
                         **simulate(muB, muC, sB, sC, sA, N_SIM)})
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False, float_format="%.4f")

print(f"Writers: {n_w}; in B: {int(IB.sum())}; in C: {int(IC.sum())}; in A: {int(IA.sum())}; "
      f"largest writer in B: {int(nB.max())} of {int(nB.sum())}")
print(f"Writer heterogeneity estimated from M0 (logit SD): B {SIG_B:.2f}, C {SIG_C:.2f}, A {SIG_A:.2f}")
print(f"Bootstrap replicates kept: {int(valid.sum())} of {N_BOOT}")
for name, r in m0_rows.items():
    print(f"M0 {name}: recall B {r[0]:.3f}, accuracy C {r[1]:.3f}, BA {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}], "
          f"G {r[5]:.3f} [{r[6]:.3f}, {r[7]:.3f}]")
pd.set_option("display.width", 250)
cols = ["heterogeneity", "median_B", "median_C", "pop_BA", "pop_G",
        "v3_P_SUPPORTED", "v4_P_SUPPORTED", "v3_P_NOT_SUPPORTED", "v4_P_NOT_SUPPORTED",
        "v3_median_CI_width", "v4_median_CI_width", "v3_P_G_PRESENT", "v4_P_G_PRESENT",
        "v4_P_G_BELOW"]
print(res[cols].round(3).to_string(index=False))
