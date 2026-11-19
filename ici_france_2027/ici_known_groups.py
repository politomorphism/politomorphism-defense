"""
ici_known_groups.py — pre-registered known-groups test of ICI
==============================================================
Șerban, G. F. (2026). Does ICI register ideological framing? A pre-registered
known-groups test on the 2027 French presidential campaign.

ICI is computed exactly as in compute_D.py (Politomorphism SRM pipeline):
    ICI(S) = 1 − mean_{i,j in S} cos(e_i, e_j)
over L2-normalised embeddings of the titles in S from
paraphrase-multilingual-MiniLM-L12-v2, the mean taken over all pairs
including i = j. For normalised vectors this equals 1 − ||mean(e)||².

Consequence (exact algebra, not an assumption): if corpus M is the union
of m titles from outlet a and m titles from outlet b, then
    ICI(M) − [ICI(a) + ICI(b)] / 2 = ||μ_a − μ_b||² / 4,
where ICI(a), ICI(b) are computed on the same m titles. Mixing two outlets
therefore always raises ICI. The test compares how much it rises when the
two outlets come from opposite ideological camps (cross-camp pairs) with
how much it rises when they come from the same camp (within-camp pairs).

Input: a CSV with columns candidate, outlet, title, date (one row per
article title, already filtered by the frozen collection rules).

Usage:
    python ici_known_groups.py --csv titles.csv --out results.json
    python ici_known_groups.py --selftest
"""

import argparse
import json
import sys
from itertools import combinations, product

import numpy as np

# ----------------------------------------------------------------------
# Frozen design constants
# ----------------------------------------------------------------------
LEFT = ["liberation.fr", "humanite.fr", "mediapart.fr"]
RIGHT = ["lefigaro.fr", "cnews.fr", "valeursactuelles.com"]
CENTRE = ["lemonde.fr", "francetvinfo.fr"]          # secondary analysis only
M_PER_CELL = 60                                     # titles per candidate x outlet
MIN_CANDIDATES = 5                                  # below this: no confirmatory verdict
SAMPLE_SEED = 20261201
HALF_SEED = 20261202
N_HALF = 2000                                       # half-samples without replacement
Q_MIN = 1.25                                        # pre-set minimum ratio X/W
Q_SENSITIVITY = (1.10, 1.50)


def share(q):
    """Convert a ratio Q = X/W into the share S = X/(X+W)."""
    return q / (1.0 + q)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CROSS_PAIRS = list(product(LEFT, RIGHT))                         # 9 pairs
WITHIN_PAIRS = list(combinations(LEFT, 2)) + list(combinations(RIGHT, 2))  # 6 pairs


# ----------------------------------------------------------------------
# ICI, identical to compute_D.py
# ----------------------------------------------------------------------
def ici_from_embeddings(emb):
    """1 − mean pairwise cosine similarity (diagonal included), as in compute_D.py."""
    emb = np.asarray(emb, dtype=float)
    if len(emb) < 2:
        return 0.0
    return float(1.0 - (emb @ emb.T).mean())


def make_encoder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    def encode(titles):
        return model.encode(titles, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True)
    return encode


# ----------------------------------------------------------------------
# Sampling and statistics
# ----------------------------------------------------------------------
def sample_cells(rows):
    """Deduplicate titles within each candidate x outlet cell, then draw
    M_PER_CELL titles without replacement with the frozen seed.
    Returns {candidate: {outlet: [titles]}} for eligible candidates and the
    per-cell counts for the report."""
    cells = {}
    for r in rows:
        key = (r["candidate"], r["outlet"])
        cells.setdefault(key, [])
        t = r["title"].strip()
        if t and t not in cells[key]:
            cells[key].append(t)
    counts = {f"{c}|{o}": len(v) for (c, o), v in cells.items()}
    rng = np.random.default_rng(SAMPLE_SEED)
    candidates = sorted({c for c, _ in cells})
    eligible = {}
    for c in candidates:
        camp_cells = [cells.get((c, o), []) for o in LEFT + RIGHT]
        if all(len(v) >= M_PER_CELL for v in camp_cells):
            eligible[c] = {}
            for o in LEFT + RIGHT + CENTRE:
                v = sorted(cells.get((c, o), []))
                if len(v) >= M_PER_CELL:
                    idx = rng.choice(len(v), M_PER_CELL, replace=False)
                    eligible[c][o] = [v[i] for i in sorted(idx)]
    return eligible, counts


def excess(mu_a, mu_b):
    """ICI(a ∪ b) − mean(ICI(a), ICI(b)) for equal-size corpora = ||μa − μb||² / 4."""
    d = mu_a - mu_b
    return float(d @ d) / 4.0


def candidate_stats(means, disp=None, m=M_PER_CELL):
    """Mean cross-camp and within-camp excess for one candidate.
    With disp (per-outlet ICI on the same m titles) also the noise-corrected
    versions. Sampling noise inflates the distance between two outlet means:
    E||μ̂a − μ̂b||² = ||μa − μb||² + tr(Σa)/m + tr(Σb)/m, and
    tr(Σ)/m is estimated without bias by ICI/(m − 1). The corrected excess is
    (||μ̂a − μ̂b||² − (ICI_a + ICI_b)/(m − 1)) / 4."""
    cross = [excess(means[a], means[b]) for a, b in CROSS_PAIRS]
    within = [excess(means[a], means[b]) for a, b in WITHIN_PAIRS]
    out = {"cross": float(np.mean(cross)), "within": float(np.mean(within))}
    if disp is not None:
        corr = lambda a, b: excess(means[a], means[b]) - (disp[a] + disp[b]) / (m - 1) / 4.0
        out["cross_bc"] = float(np.mean([corr(a, b) for a, b in CROSS_PAIRS]))
        out["within_bc"] = float(np.mean([corr(a, b) for a, b in WITHIN_PAIRS]))
    return out


def pooled(stats_by_cand, key_c="cross", key_w="within"):
    X = float(np.mean([s[key_c] for s in stats_by_cand.values()]))
    W = float(np.mean([s[key_w] for s in stats_by_cand.values()]))
    return X, W, (X / W if W > 0 else float("inf")), X - W


def verdict(s_point, s_lo, s_hi, q_min=Q_MIN):
    """Decision on the share S = X/(X+W); S = 0.5 means no camp effect."""
    s_min = share(q_min)
    if s_lo > 0.5 and s_point >= s_min:
        return "SUPPORTED"
    if s_hi < s_min:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def analyse(eligible, encode):
    emb = {c: {o: np.asarray(encode(t)) for o, t in cell.items()} for c, cell in eligible.items()}
    means = {c: {o: e.mean(axis=0) for o, e in d.items()} for c, d in emb.items()}
    disp = {c: {o: ici_from_embeddings(e) for o, e in d.items()} for c, d in emb.items()}
    stats = {c: candidate_stats(means[c], disp[c]) for c in emb}
    # Primary: noise-corrected excess (see candidate_stats); plain excess is secondary.
    X, W, Q, D = pooled(stats, "cross_bc", "within_bc")
    Xp, Wp, Qp, Dp = pooled(stats)

    # Identity check on real data: ICI of the mixed corpus minus mean single-outlet ICI
    c0 = next(iter(emb))
    a, b = CROSS_PAIRS[0]
    mixed = ici_from_embeddings(np.vstack([emb[c0][a], emb[c0][b]]))
    identity_gap = mixed - (disp[c0][a] + disp[c0][b]) / 2 - excess(means[c0][a], means[c0][b])

    # Uncertainty: half-sampling without replacement. In each replicate a random
    # half of the title positions is drawn for each outlet and applied to that
    # outlet's cell for every candidate; the statistic is recomputed with m/2. The SD of the half-sample estimates
    # approximates the sampling SD of the full-sample estimate (resampling with
    # replacement is not used: duplicated titles bias the corrected excess).
    # Normal 95% intervals: estimate ± 1.96·SD.
    rng = np.random.default_rng(HALF_SEED)
    h = M_PER_CELL // 2
    S0 = X / (X + W)
    ss, ds = [], []
    for _ in range(N_HALF):
        keep = {o: rng.permutation(M_PER_CELL)[:h] for o in LEFT + RIGHT}
        st = {}
        for c, d in emb.items():
            sub = {o: d[o][keep[o]] for o in LEFT + RIGHT}
            st[c] = candidate_stats({o: e.mean(axis=0) for o, e in sub.items()},
                                    {o: ici_from_embeddings(e) for o, e in sub.items()}, h)
        xh, wh, _, dh = pooled(st, "cross_bc", "within_bc")
        ss.append(xh / (xh + wh))
        ds.append(dh)
    sd_s, sd_d = float(np.std(ss)), float(np.std(ds))
    s_lo, s_hi = S0 - 1.96 * sd_s, S0 + 1.96 * sd_s
    d_lo, d_hi = D - 1.96 * sd_d, D + 1.96 * sd_d
    q_of = lambda v: v / (1.0 - v) if v < 1 else float("inf")

    # Secondary: centre outlets against each camp (exploratory, reported only)
    centre = {}
    for c in emb:
        cs = [o for o in CENTRE if o in means[c]]
        if cs:
            centre[c] = {
                "centre_left": float(np.mean([excess(means[c][x], means[c][l]) for x in cs for l in LEFT])),
                "centre_right": float(np.mean([excess(means[c][x], means[c][r]) for x in cs for r in RIGHT])),
            }

    n = len(emb)
    result = {
        "n_candidates": n,
        "candidates": sorted(emb),
        "per_candidate": stats,
        "per_candidate_Q_above_1": int(sum(s["cross_bc"] > s["within_bc"] for s in stats.values())),
        "primary": {"X_cross": X, "W_within": W, "S": S0, "S_ci95": [s_lo, s_hi],
                    "Q": Q, "Q_ci95": [q_of(s_lo), q_of(s_hi)],
                    "Delta": D, "Delta_ci95": [d_lo, d_hi]},
        "verdict": verdict(S0, s_lo, s_hi) if n >= MIN_CANDIDATES else "NOT RUN (fewer than 5 eligible candidates)",
        "sensitivity": {str(t): verdict(S0, s_lo, s_hi, t) for t in Q_SENSITIVITY},
        "secondary_plain_excess": {"X": Xp, "W": Wp, "Q": Qp, "Delta": Dp},
        "secondary_centre": centre,
        "identity_check_gap": identity_gap,
    }
    return result


# ----------------------------------------------------------------------
# Self-test on synthetic embeddings (no model download, no real data)
# ----------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    dim = 64

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def synth(camp_shift, outlet_shift):
        base = rng.normal(size=dim)
        camp = {"L": rng.normal(size=dim), "R": rng.normal(size=dim)}
        cells = {}
        for c in [f"cand{i}" for i in range(6)]:
            cand = rng.normal(size=dim)
            cells[c] = {}
            for o in LEFT + RIGHT + CENTRE:
                side = "L" if o in LEFT else "R" if o in RIGHT else None
                centre = 3 * base + cand + outlet_shift * rng.normal(size=dim)
                if side:
                    centre = centre + camp_shift * camp[side]
                cells[c][o] = unit(centre + 1.5 * rng.normal(size=(M_PER_CELL, dim)))
        return cells

    # exact identity on random data
    a = unit(rng.normal(size=(M_PER_CELL, dim)) + rng.normal(size=dim))
    b = unit(rng.normal(size=(M_PER_CELL, dim)) + rng.normal(size=dim))
    gap = ici_from_embeddings(np.vstack([a, b])) - (ici_from_embeddings(a) + ici_from_embeddings(b)) / 2 \
        - excess(a.mean(0), b.mean(0))
    print(f"identity gap (should be ~0): {gap:.2e}")

    for label, cs, os_ in (("strong camp effect", 1.2, 0.3), ("small camp effect", 0.4, 0.6),
                           ("no camp effect", 0.0, 0.6)):
        cells = synth(cs, os_)
        lookup = {}
        for c, d in cells.items():
            for o, e in d.items():
                lookup[(c, o)] = e
        global N_HALF
        saved, N_HALF = N_HALF, 300
        res = analyse({c: {o: [f"{c}|{o}|{i}" for i in range(M_PER_CELL)] for o in d} for c, d in cells.items()},
                      lambda titles: lookup[tuple(titles[0].split("|")[:2])])
        N_HALF = saved
        p = res["primary"]
        print(f"{label}: S = {p['S']:.3f} [{p['S_ci95'][0]:.3f}, {p['S_ci95'][1]:.3f}], "
              f"Q = {p['Q']:.2f} -> {res['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="ici_known_groups_results.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import csv
    with open(a.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    eligible, counts = sample_cells(rows)
    if not eligible:
        print("No eligible candidates.")
        sys.exit(1)
    res = analyse(eligible, make_encoder())
    res["cell_counts"] = counts
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("n_candidates", "primary", "verdict", "sensitivity")}, indent=2))


if __name__ == "__main__":
    main()
