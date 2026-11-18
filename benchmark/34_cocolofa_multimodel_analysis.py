# -*- coding: utf-8 -*-
"""
34_cocolofa_multimodel_analysis.py
Frozen analysis for the pre-registered multi-model CoCoLoFa study (draft v5).
Run ONCE after the prediction files of all runs are committed.

PRIMARY (per arm, run 1, against corpus labels)
  Writer-equal hard-cell balanced accuracy: in cell B and in cell C the proportion
  correct is computed for each writer with comments in that cell, averaged over
  writers with equal weight, and the two averages are averaged. An abstention is
  an error in every cell. 95% interval: percentile writer bootstrap, 10,000 kept
  replicates, seed BOOT_SEED; 49 writers drawn with replacement, jointly for all
  cells; a replicate with no writer in B or C is discarded and redrawn.
  Verdict (Table 2): SUPPORTED if BA >= 0.75 and lower bound > 0.60;
  NOT SUPPORTED if BA <= 0.60 and upper bound < 0.75; otherwise INCONCLUSIVE.

SECONDARY, DESCRIPTIVE (Table 4 of the pre-registration), S1 to S10.\n  Every interval is conditional on the run it is computed from (run 1 unless stated).

Usage (from the repository root):
    python benchmark/34_cocolofa_multimodel_analysis.py            the study
    python benchmark/34_cocolofa_multimodel_analysis.py --test-m0  test on the public
                                                                   gpt-4o-mini file only
Writes benchmark/mm_report.txt (or mm_report_m0_test.txt).

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cocolofa_fd_sample_manifest.csv"
M0_FILE = HERE / "cocolofa_fd_predictions.csv"
LOCAL = HERE / "cocolofa_fd_eval_local.csv"
ARMS = {"M1": "gemini-3.1-flash-lite", "M2": "gemini-3.5-flash-lite", "M3": "gemma-4-31b-it"}
N_RUNS = 3
BOOT_SEED = 20261119
N_BOOT = 10_000
ITEM_BOOT = 2_000
HI, LO = 0.75, 0.60
G_THR = 0.10
STRICT_MARKER = re.compile(r"\beither\b|only two|two (options|choices|sides)|third way|"
                           r"no (other|middle)", re.I)
BIG_WRITER_MIN = 50          # the writer with 53 comments in B


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- data
man = pd.read_csv(MANIFEST)
WRITERS = np.sort(man["worker_id"].unique())
W_IDX = {w: i for i, w in enumerate(WRITERS)}
N_W = len(WRITERS)


def prepare(df):
    df = df.copy()
    df["correct"] = ((df.flag_fd == df.gold_false_dilemma) & (df.abstain == 0)).astype(int)
    return df


def per_writer(df, cell, col):
    """Return (k, n) arrays over all writers for one cell and one 0/1 column."""
    k, n = np.zeros(N_W), np.zeros(N_W)
    sub = df[df.cell == cell]
    for w, v in zip(sub.worker_id, sub[col]):
        k[W_IDX[w]] += v
        n[W_IDX[w]] += 1
    return k, n


def make_draws(seed, n_keep):
    """Writer-bootstrap weight matrix with replicates lacking B or C writers redrawn."""
    rng = np.random.default_rng(seed)
    nB = per_writer(man.assign(x=1), "B", "x")[1]
    nC = per_writer(man.assign(x=1), "C", "x")[1]
    nA = per_writer(man.assign(x=1), "A", "x")[1]
    keep, discarded = [], 0
    while sum(len(k) for k in keep) < n_keep:
        d = rng.integers(0, N_W, size=(n_keep, N_W))
        Wm = np.zeros((n_keep, N_W))
        np.add.at(Wm, (np.repeat(np.arange(n_keep), N_W), d.ravel()), 1.0)
        ok = ((Wm @ (nB > 0)) > 0) & ((Wm @ (nC > 0)) > 0) & ((Wm @ (nA > 0)) > 0)
        discarded += int((~ok).sum())
        keep.append(Wm[ok])
    return np.vstack(keep)[:n_keep], discarded


DRAWS, DISCARDED = make_draws(BOOT_SEED, N_BOOT)


def subset_draws(writers, seed=BOOT_SEED, n_keep=N_BOOT):
    """Bootstrap restricted to a subset of writers: each replicate draws len(writers)
    writers with replacement from that subset only. Returns a weight matrix over all
    writers (zero outside the subset)."""
    rng = np.random.default_rng(seed)
    idx = np.array([W_IDX[w] for w in sorted(writers)])
    d = rng.integers(0, len(idx), size=(n_keep, len(idx)))
    Wm = np.zeros((n_keep, N_W))
    np.add.at(Wm, (np.repeat(np.arange(n_keep), len(idx)), idx[d].ravel()), 1.0)
    return Wm


def we(k, n):
    """Writer-equal rate: point estimate and bootstrap replicates."""
    ind = (n > 0).astype(float)
    rate = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    return float((rate * ind).sum() / ind.sum()), (DRAWS @ rate) / (DRAWS @ ind)


def cw(k, n):
    """Comment-weighted rate: point estimate and writer-bootstrap replicates."""
    return float(k.sum() / n.sum()), (DRAWS @ k) / (DRAWS @ n)


def pct(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def verdict(ba, lo, hi):
    if ba >= HI and lo > LO:
        return "SUPPORTED"
    if ba <= LO and hi < HI:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def gap_class(lo, hi):
    if lo > G_THR:
        return "PRESENT"
    if hi < G_THR:
        return "BELOW THRESHOLD"
    return "UNDETERMINED"


def summary(df, rate=we):
    kB, nB = per_writer(df, "B", "correct")
    kC, nC = per_writer(df, "C", "correct")
    fA, nA = per_writer(df, "A", "flag_fd")
    fB, _ = per_writer(df, "B", "flag_fd")
    fC, _ = per_writer(df, "C", "flag_fd")
    fD, nD = per_writer(df, "D", "flag_fd")
    b, bb = rate(kB, nB)
    c, cb = rate(kC, nC)
    a, ab = rate(fA, nA)
    fb, fbb = rate(fB, nB)
    fc, fcb = rate(fC, nC)
    fd, fdb = rate(fD, nD)
    return {"recB": (b, bb), "accC": (c, cb), "BA": (0.5 * (b + c), 0.5 * (bb + cb)),
            "G": (a - fb, ab - fbb), "FA": (fc - fd, fcb - fdb)}


def item_ci(df, seed=0):
    """Item-level bootstrap of the comment-weighted BA, as in the first study (S8)."""
    rng = np.random.default_rng(seed)
    hard = df[df.cell.isin(["B", "C"])]
    g, ok = hard.gold_false_dilemma.values, hard.correct.values
    vals = []
    for _ in range(ITEM_BOOT):
        i = rng.integers(0, len(hard), len(hard))
        gg, oo = g[i], ok[i]
        if (gg == 1).any() and (gg == 0).any():
            vals.append(0.5 * (oo[gg == 1].mean() + oo[gg == 0].mean()))
    return pct(np.array(vals))


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-m0", action="store_true")
    a = ap.parse_args()
    lines = []

    def e(s=""):
        lines.append(s)
        print(s)

    m0 = prepare(pd.read_csv(M0_FILE))
    preds = {}
    if not a.test_m0:
        for arm in ARMS:
            for r in range(1, N_RUNS + 1):
                p = HERE / f"mm_pred_{arm}_run{r}.csv"
                if not p.exists():
                    raise SystemExit(f"{p.name} missing; the analysis runs once, after all runs")
                preds[(arm, r)] = prepare(pd.read_csv(p))
    arms = list(ARMS) if not a.test_m0 else []

    e("Multi-model CoCoLoFa study: frozen analysis" + ("  [TEST ON M0 ONLY]" if a.test_m0 else ""))
    e(f"manifest SHA-256 {sha256(MANIFEST)}")
    for (arm, r), _ in sorted(preds.items()):
        e(f"  {arm} run {r}: {sha256(HERE / f'mm_pred_{arm}_run{r}.csv')}")
    e(f"writer bootstrap: {N_BOOT} kept replicates, seed {BOOT_SEED}, {DISCARDED} discarded")
    e()

    # PRIMARY ------------------------------------------------------------
    e("PRIMARY: writer-equal hard-cell balanced accuracy, run 1, corpus labels")
    for arm in arms:
        s = summary(preds[(arm, 1)])
        ba, bab = s["BA"]
        lo, hi = pct(bab)
        e(f"  {arm} {ARMS[arm]:24s} BA {ba:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"recall B {s['recB'][0]:.3f}  accuracy C {s['accC'][0]:.3f}  -> {verdict(ba, lo, hi)}")
    s0 = summary(m0)
    lo0, hi0 = pct(s0["BA"][1])
    e(f"  M0 gpt-4o-mini (context, no verdict)  BA {s0['BA'][0]:.3f} [{lo0:.3f}, {hi0:.3f}]")
    e()

    every = [(arm, prepare(preds[(arm, 1)])) for arm in arms] + [("M0", m0)]

    # Per-cell table -----------------------------------------------------
    e("Per-cell table, run 1 (comment counts; abstention counted as an error)")
    for name, df in every:
        cells = []
        for cell in "ABCD":
            sub = df[df.cell == cell]
            cells.append(f"{cell} correct {sub.correct.sum()}/{len(sub)} flagged {sub.flag_fd.sum()} "
                         f"abstained {sub.abstain.sum()}")
        e(f"  {name}: " + " | ".join(cells))
    e()

    # S1, S2 -------------------------------------------------------------
    e("S1 detection gap G (writer-equal; comment-weighted in brackets)")
    for name, df in every:
        s, sc = summary(df), summary(df, rate=cw)
        lo, hi = pct(s["G"][1])
        e(f"  {name}: G {s['G'][0]:.3f} [{lo:.3f}, {hi:.3f}] -> {gap_class(lo, hi)}"
          f"   (comment-weighted {sc['G'][0]:.3f})")
    cntAB = man.groupby(["worker_id", "cell"]).size().unstack(fill_value=0)
    common_ab = set(cntAB[(cntAB.A > 0) & (cntAB.B > 0)].index)
    e(f"S1 sensitivity: writers present in both A and B: {len(common_ab)} "
      f"({int(cntAB.loc[list(common_ab), 'A'].sum())} comments in A, "
      f"{int(cntAB.loc[list(common_ab), 'B'].sum())} in B)")
    for name, df in every:
        sub = df[df.worker_id.isin(common_ab)]
        fA, nA = per_writer(sub, "A", "flag_fd")
        fB, nB = per_writer(sub, "B", "flag_fd")
        ind = ((nA > 0) & (nB > 0)).astype(float)
        diff = np.where(ind > 0, fA / np.maximum(nA, 1) - fB / np.maximum(nB, 1), 0.0)
        g = float((diff * ind).sum() / ind.sum())
        SD = subset_draws(common_ab)
        gb = (SD @ diff) / (SD @ ind)
        lo, hi = pct(gb)
        e(f"  {name}: within-writer G on common writers {g:.3f} [{lo:.3f}, {hi:.3f}] -> {gap_class(lo, hi)}")
    e("S2 false-alarm gap, flag rate C minus D (writer-equal; comment-weighted in brackets)")
    for name, df in every:
        s, sc = summary(df), summary(df, rate=cw)
        lo, hi = pct(s["FA"][1])
        e(f"  {name}: {s['FA'][0]:.3f} [{lo:.3f}, {hi:.3f}]   (comment-weighted {sc['FA'][0]:.3f})")
    e()

    # S3 -----------------------------------------------------------------
    if arms:
        e("S3 paired contrasts, run 1, same writer draws (descriptive, unadjusted)")
        for x, y in (("M3", "M1"), ("M3", "M2"), ("M2", "M1")):
            sx, sy = summary(preds[(x, 1)]), summary(preds[(y, 1)])
            for q, lab in (("recB", "B recall"), ("G", "gap G")):
                d = sx[q][0] - sy[q][0]
                lo, hi = pct(sx[q][1] - sy[q][1])
                note = "observed difference" if (lo > 0 or hi < 0) else "no difference detected"
                e(f"  {x} - {y} {lab}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {note}")
        e()

    # S4 -----------------------------------------------------------------
    e("S4 writer structure in B")
    nB_all = man[man.cell == "B"].worker_id.value_counts()
    big = nB_all.index[0]
    e(f"  largest writer in B: {big} with {nB_all.iloc[0]} of {nB_all.sum()} comments")
    for name, df in every:
        b = df[df.cell == "B"]
        s, sc = summary(df), summary(df, rate=cw)
        e(f"  {name}: B recall largest writer {b[b.worker_id == big].correct.mean():.3f}, "
          f"other writers {b[b.worker_id != big].correct.mean():.3f}; "
          f"writer-equal {s['recB'][0]:.3f} vs comment-weighted {sc['recB'][0]:.3f}; "
          f"BA writer-equal {s['BA'][0]:.3f} vs comment-weighted {sc['BA'][0]:.3f}")
    e()

    # S5 -----------------------------------------------------------------
    e("S5 abstentions, coverage, flag-based balanced accuracy")
    for name, df in every:
        codes = df.stage1_pred.value_counts()
        ab = {k: int(v) for k, v in codes.items() if k.startswith("ABSTAIN") or k == "insufficient_evidence"}
        ans = df[df.abstain == 0]
        hard_ans = ans[ans.cell.isin(["B", "C"])]
        acc_ans = hard_ans.correct.mean() if len(hard_ans) else float("nan")
        kB, nB = per_writer(df, "B", "flag_fd")
        kC, nC = per_writer(df, "C", "flag_fd")
        fb = we(kB, nB)[0]
        fc = we(kC, nC)[0]
        e(f"  {name}: abstentions {ab or 0}; coverage {len(ans) / len(df):.3f}; accuracy on "
          f"answered hard items {acc_ans:.3f}; flag-based BA (writer-equal) {0.5 * (fb + 1 - fc):.3f}")
    e()

    # S6 -----------------------------------------------------------------
    if arms:
        e("S6 stability across runs")
        for arm in arms:
            p = [preds[(arm, r)].set_index("id").stage1_pred for r in range(1, N_RUNS + 1)]
            same = float(((p[0] == p[1]) & (p[1] == p[2])).mean())
            vs, verdicts = [], []
            for r in range(1, N_RUNS + 1):
                s = summary(preds[(arm, r)])
                lo, hi = pct(s["BA"][1])
                verdicts.append(verdict(s["BA"][0], lo, hi))
                vs.append(f"run {r} BA {s['BA'][0]:.3f} {verdicts[-1]}")
            stable = len(set(verdicts)) == 1
            bas = [summary(preds[(arm, r)])["BA"][0] for r in range(1, N_RUNS + 1)]
            vs.append(f"mean over runs {np.mean(bas):.3f}, range {min(bas):.3f}-{max(bas):.3f}")
            e(f"  {arm}: identical labels in all runs {same:.3f}; " + "; ".join(vs)
              + ("" if stable else "  -> run 1 verdict stands, marked UNSTABLE"))
        e()

    # S7 -----------------------------------------------------------------
    if arms:
        e("S7 consensus misses in B, run 1")
        flag = {arm: preds[(arm, 1)].set_index("id").flag_fd for arm in arms}
        bids = man[man.cell == "B"].id
        none3 = [i for i in bids if all(flag[a].loc[i] == 0 for a in arms)]
        m0f = m0.set_index("id").flag_fd
        none4 = [i for i in none3 if m0f.loc[i] == 0]
        e(f"  missed by all of M1-M3: {len(none3)}/{len(bids)}; also by M0: {len(none4)}")
        e(f"  identifiers (all four): {', '.join(map(str, none4))}")
        e()

    # S9 -----------------------------------------------------------------
    e("S9 writers present in both B and C (sensitivity; writer-equal, run 1)")
    cnt = man.groupby(["worker_id", "cell"]).size().unstack(fill_value=0)
    for cell in "ABCD":
        col = cnt[cell][cnt[cell] > 0]
        e(f"  cell {cell}: {len(col)} writers, comments per writer median {col.median():.0f}, "
          f"max {col.max()}, writers with one comment {int((col == 1).sum())}")
    common = set(cnt[(cnt.B > 0) & (cnt.C > 0)].index)
    e(f"  writers in both B and C: {len(common)}; only B: {int(((cnt.B > 0) & (cnt.C == 0)).sum())}; "
      f"only C: {int(((cnt.C > 0) & (cnt.B == 0)).sum())}")
    SD9 = subset_draws(common)
    e(f"  bootstrap restricted to these {len(common)} writers, {N_BOOT} replicates, seed {BOOT_SEED}, "
      f"0 discarded (every drawn writer has comments in B and C)")
    for name, df in every:
        sub = df[df.worker_id.isin(common)]
        kB, nB = per_writer(sub, "B", "correct")
        kC, nC = per_writer(sub, "C", "correct")
        rB = np.where(nB > 0, kB / np.maximum(nB, 1), 0.0)
        rC = np.where(nC > 0, kC / np.maximum(nC, 1), 0.0)
        iB, iC = (nB > 0).astype(float), (nC > 0).astype(float)
        bab = 0.5 * ((SD9 @ rB) / (SD9 @ iB) + (SD9 @ rC) / (SD9 @ iC))
        s = summary(sub)
        lo, hi = pct(bab)
        e(f"  {name}: BA on common writers {s['BA'][0]:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"(recall B {s['recB'][0]:.3f}, accuracy C {s['accC'][0]:.3f})")
    e()

    # S10 ----------------------------------------------------------------
    e("S10 threshold sensitivity (descriptive; the verdict uses 0.75 / 0.60 only).")
    e("  Each pair h/l replaces both thresholds: SUPPORTED if BA >= h and lower bound > l;")
    e("  NOT SUPPORTED if BA <= l and upper bound < h; otherwise INCONCLUSIVE.")
    for name, df in every:
        s = summary(df)
        ba, (lo, hi) = s["BA"][0], pct(s["BA"][1])
        alt = []
        for h, l_ in ((0.70, 0.55), (0.75, 0.65), (0.80, 0.60), (0.80, 0.65)):
            v = "SUPPORTED" if (ba >= h and lo > l_) else "NOT SUPPORTED" if (ba <= l_ and hi < h) \
                else "INCONCLUSIVE"
            alt.append(f"{h:.2f}/{l_:.2f} {v}")
        e(f"  {name}: " + "; ".join(alt))
    e()

    # S8 -----------------------------------------------------------------
    e("S8 continuity with the first study (comment-weighted, rule of Table 2)")
    for name, df in every:
        sc = summary(df, rate=cw)
        lo, hi = pct(sc["BA"][1])
        ilo, ihi = item_ci(df)
        easy = df[df.cell.isin(["A", "D"])]
        eb = 0.5 * (easy[easy.cell == "A"].correct.mean() + easy[easy.cell == "D"].correct.mean())
        e(f"  {name}: comment-weighted BA {sc['BA'][0]:.3f}, writer CI [{lo:.3f}, {hi:.3f}] -> "
          f"{verdict(sc['BA'][0], lo, hi)}; item CI [{ilo:.3f}, {ihi:.3f}]; easy-cell BA {eb:.3f}")
        for cell in ("C", "D"):
            sub = df[df.cell == cell]
            neu = sub[sub.source_label == "none"].correct.mean()
            oth = sub[sub.source_label != "none"].correct.mean()
            e(f"     cell {cell}: neutral {neu:.3f}, other fallacy {oth:.3f}")
    if LOCAL.exists():
        text = pd.read_csv(LOCAL).set_index("id").text
        e("  stricter marker (ignores bare 'or'): cells recomputed from local text")
        for name, df in every:
            d = df.copy()
            d["mk"] = [int(bool(STRICT_MARKER.search(str(text.loc[i])))) for i in d.id]
            hb = d[(d.gold_false_dilemma == 1) & (d.mk == 0)].correct.mean()
            hc = d[(d.gold_false_dilemma == 0) & (d.mk == 1)].correct.mean()
            e(f"     {name}: hard-cell BA (comment-weighted) {0.5 * (hb + hc):.3f}")
    else:
        e("  stricter marker: local comment file not present; not computed")

    out = HERE / ("mm_report_m0_test.txt" if a.test_m0 else "mm_report.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written to {out.name}")


if __name__ == "__main__":
    main()
