"""
Priority-1 statistics for the clean, article-disjoint comparison.
Reads clean/baseline_test_scores.csv and (if present) clean/transformer_test_scores.csv.
Computes, with article-clustered bootstrap 95% CIs: per-model precision/recall/F1;
McNemar on paired default predictions; recall at matched precision and at matched
review budget (flagged count); PR average precision. Pure numpy/scipy/sklearn:
no network or model download needed, so it runs anywhere the two CSVs exist.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

OUT = Path("clean"); RNG = np.random.default_rng(42); B = 2000
base = pd.read_csv(OUT / "baseline_test_scores.csv")
tf_path = OUT / "transformer_test_scores.csv"
has_tf = tf_path.exists()

def prf(y, pred):
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

def cluster_boot_ci(df, metric_fn):
    arts = df["article"].unique()
    by = {a: df[df["article"]==a] for a in arts}
    vals = []
    for _ in range(B):
        samp = pd.concat([by[a] for a in RNG.choice(arts, size=len(arts), replace=True)], ignore_index=True)
        vals.append(metric_fn(samp))
    vals = np.array(vals)
    lo = np.nanpercentile(vals, 2.5, axis=0); hi = np.nanpercentile(vals, 97.5, axis=0)
    return lo, hi

def report_model(df, scorecol, predcol, name):
    y = df["y_true"].values; pred = df[predcol].values
    p, r, f = prf(y, pred)
    ap = average_precision_score(y, df[scorecol].values)
    lo, hi = cluster_boot_ci(df, lambda d: prf(d["y_true"].values, d[predcol].values))
    print(f"\n[{name}] default rule  (n={len(df)}, flagged={int(pred.sum())})")
    print(f"  precision {p:.3f}  [{lo[0]:.3f}, {hi[0]:.3f}]")
    print(f"  recall    {r:.3f}  [{lo[1]:.3f}, {hi[1]:.3f}]")
    print(f"  F1        {f:.3f}  [{lo[2]:.3f}, {hi[2]:.3f}]")
    print(f"  AP        {ap:.3f}")

report_model(base, "baseline_score", "baseline_pred", "BASELINE")

if not has_tf:
    print("\n[transformer scores not present yet — paired tests will run once clean/transformer_test_scores.csv is supplied]")
else:
    tf = pd.read_csv(tf_path)
    m = base.merge(tf, on="sample_id", suffixes=("_b", "_t"))
    # reconcile article/y_true columns
    art = m["article_b"] if "article_b" in m else m["article"]
    y = m["y_true_b"].values if "y_true_b" in m else m["y_true"].values
    m = m.assign(article=art, y_true=y)
    report_model(m.assign(), "transformer_score", "transformer_pred", "TRANSFORMER")

    # McNemar on paired default predictions
    bp = m["baseline_pred"].values; tp = m["transformer_pred"].values
    b01 = int(((bp==0)&(tp==1)).sum())  # baseline wrong-dir vs transformer: only-transformer-positive
    # McNemar should compare correctness, not raw positive. Use correctness vs y.
    bc = (bp==y).astype(int); tc = (tp==y).astype(int)
    b = int(((bc==1)&(tc==0)).sum())  # baseline right, transformer wrong
    c = int(((bc==0)&(tc==1)).sum())  # baseline wrong, transformer right
    from scipy.stats import binomtest
    pval = binomtest(min(b,c), b+c, 0.5).pvalue if (b+c)>0 else 1.0
    print(f"\n[McNemar on correctness]  baseline-only-correct={b}  transformer-only-correct={c}  exact p={pval:.3f}")

    # paired transition table (baseline -> transformer) among positives and negatives
    print("\n[paired default-prediction transitions]")
    for lab,yy in [("true positives", 1), ("true negatives", 0)]:
        sub = m[m["y_true"]==yy]
        a = int(((sub["baseline_pred"]==1)&(sub["transformer_pred"]==1)).sum())
        bb = int(((sub["baseline_pred"]==1)&(sub["transformer_pred"]==0)).sum())
        cc = int(((sub["baseline_pred"]==0)&(sub["transformer_pred"]==1)).sum())
        dd = int(((sub["baseline_pred"]==0)&(sub["transformer_pred"]==0)).sum())
        print(f"  {lab} (n={len(sub)}): both+={a} base+only={bb} tf+only={cc} both-={dd}")

    # recall at matched review budget: set transformer threshold so it flags the
    # same number of paragraphs the baseline flags, compare recall.
    nb = int(m["baseline_pred"].sum())
    order = np.argsort(-m["transformer_score"].values)
    tf_budget = np.zeros(len(m), int); tf_budget[order[:nb]] = 1
    _, r_b, _ = prf(m["y_true"].values, m["baseline_pred"].values)
    _, r_tb, _ = prf(m["y_true"].values, tf_budget)
    print(f"\n[matched review budget = {nb} flags]  baseline recall={r_b:.3f}  transformer recall={r_tb:.3f}")
    print("\n(Full corrected Paper II will report these with CIs and matched-precision points.)")
