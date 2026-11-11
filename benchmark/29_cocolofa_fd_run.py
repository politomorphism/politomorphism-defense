# -*- coding: utf-8 -*-
"""
29_cocolofa_fd_run.py
Pre-registered structural-sensitivity test of the false_dilemma detector on CoCoLoFa.
ONE confirmatory run. See Preregistration_CoCoLoFa_Structural_Test_RO_EN.docx.

WHAT IS FIXED (nothing here may change after the first model call)
  Detector  the frozen v2 prompt, system message, label parser and retry logic,
            IMPORTED from 23_logic_fd_eval.py (not retyped); model gpt-4o-mini,
            temperature 0. Each comment is passed whole in the "Sentence" slot.
  Sample    cocolofa_fd_sample_manifest.csv, produced by 28_cocolofa_sample.py; its
            SHA-256 must equal MANIFEST_SHA256 below or the script stops.
  Outcome   balanced accuracy on the HARD cells (B: gold false dilemma without
            marker; C: gold non-false-dilemma with marker),
              BA = 1/2 * (correct in B / all B + correct in C / all C).
            An abstention ("insufficient_evidence") is an incorrect answer in
            either cell. 95% CI: 2000 item-level bootstrap replicates, seed 0
            (the functions are imported from 26_logic_hard_cells.py).
  Rule      SUPPORTED if BA >= 0.75 and CI lower bound > 0.60;
            NOT SUPPORTED if BA <= 0.60 or CI lower bound <= 0.50;
            otherwise INCONCLUSIVE. Same thresholds as the LOGIC re-analysis; they
            are not adjusted to this corpus.
  Reported whatever the outcome.

SECONDARY, DESCRIPTIVE (they never change the verdict)
  easy-cell balanced accuracy; per-cell accuracy; cell C split into neutral
  ("none") and other-fallacy comments; abstentions per cell; a bootstrap clustered
  by writer (worker_id); the hard-cell score under a stricter marker without bare
  "or"; and a supervised lexical control (TF-IDF + logistic regression, 5-fold) on
  the hard cells, which is a different protocol from the zero-shot detector and is
  given for context only.

RUN ONCE. The script refuses to call the model if a predictions file already
exists. To re-print the report from saved predictions (no model calls):
    python benchmark/29_cocolofa_fd_run.py --rescore

Usage (from the repository root, with OPENAI_API_KEY set):
    python benchmark/29_cocolofa_fd_run.py

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import argparse
import csv
import hashlib
import importlib.util
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cocolofa_fd_sample_manifest.csv"
LOCAL = HERE / "cocolofa_fd_eval_local.csv"
PRED = HERE / "cocolofa_fd_predictions.csv"
REPORT = HERE / "cocolofa_fd_report.txt"
MANIFEST_SHA256 = "e35b540ec55962eb9de92b9887bd56647f563944b31050163c8fad531b2b9975"
STRICT_MARKER = re.compile(r"\beither\b|only two|two (options|choices|sides)|third way|"
                           r"no (other|middle)", re.I)


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det = load("fd_eval", "23_logic_fd_eval.py")
hc = load("hard_cells", "26_logic_hard_cells.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci(sample):
    reps = sorted(v for v in (hc.balacc(x) for x in hc.boot(sample)) if v == v)
    return reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]


def cluster_ci(rows, reps=2000, seed=0):
    """Bootstrap that resamples writers, keeping all of a writer's comments together."""
    by = {}
    for r in rows:
        by.setdefault(r["worker_id"], []).append((r["g"], r["correct"]))
    keys = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        s = []
        for _ in keys:
            s.extend(by[keys[rng.randrange(len(keys))]])
        v = hc.balacc(s)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def run_model(model, backend):
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("sample manifest differs from the pre-registered one; stopping")
    if PRED.exists():
        raise SystemExit(f"{PRED.name} already exists: the confirmatory run was already made. "
                         "Use --rescore to re-print the report.")
    if not LOCAL.exists():
        raise SystemExit(f"{LOCAL.name} not found; run 28_cocolofa_sample.py first")
    items = list(csv.DictReader(open(LOCAL, encoding="utf-8")))
    man = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if [r["id"] for r in items] != [r["id"] for r in man]:
        raise SystemExit("local evaluation file does not match the manifest; stopping")
    asker = det.make_asker(backend, model)
    out = []
    for i, r in enumerate(items):
        raw = det.call_with_retry(asker, det.SYSTEM,
                                  det.PROMPT.format(defs=det.DEFS, text=r["text"]))
        pred = det.parse_label(raw)
        out.append({"id": r["id"], "split": r["split"], "cell": r["cell"],
                    "gold_false_dilemma": r["gold_false_dilemma"],
                    "source_label": r["source_label"], "marker": r["marker"],
                    "worker_id": r["worker_id"], "stage1_pred": pred,
                    "flag_fd": int(pred == det.TARGET), "abstain": int(pred == det.ABSTAIN)})
        if (i + 1) % 40 == 0:
            print(f"   {i + 1}/{len(items)} done")
    with open(PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"saved {PRED.name} (no comment text)")


def lexical_control(rows):
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return None
    if not LOCAL.exists():
        return None
    text = {r["id"]: r["text"] for r in csv.DictReader(open(LOCAL, encoding="utf-8"))}
    hard = [r for r in rows if r["cell"] in ("B", "C")]
    X = [text[r["id"]] for r in hard]
    y = np.array([r["g"] for r in hard])
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=20261111).split(X, y):
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit([X[i] for i in tr], y[tr])
        pred[te] = clf.predict([X[i] for i in te])
    s = [(int(g), int(p == g)) for g, p in zip(y, pred)]
    return hc.balacc(s), ci(s)


def report(model):
    P = list(csv.DictReader(open(PRED, encoding="utf-8")))
    rows = []
    for r in P:
        g = int(r["gold_false_dilemma"])
        correct = int(int(r["flag_fd"]) == g and int(r["abstain"]) == 0)  # abstention = error
        rows.append({"id": r["id"], "cell": r["cell"], "g": g, "correct": correct,
                     "abstain": int(r["abstain"]), "source_label": r["source_label"],
                     "worker_id": r["worker_id"]})
    L = []

    def e(s=""):
        L.append(s)
        print(s)

    e("CoCoLoFa structural-sensitivity test, false_dilemma (pre-registered, one run)")
    e(f"model={model}  n={len(rows)}")
    e(f"sample manifest SHA-256: {sha256(MANIFEST)}")
    e(f"predictions file SHA-256: {sha256(PRED)}")
    e()
    e("Per cell (abstention counted as an error):")
    for cell, name in (("A", "A  false dilemma, marker      (easy)"),
                       ("B", "B  false dilemma, no marker   (HARD)"),
                       ("C", "C  not FD, marker             (HARD)"),
                       ("D", "D  not FD, no marker          (easy)")):
        rr = [r for r in rows if r["cell"] == cell]
        k = sum(r["correct"] for r in rr)
        e(f"  {name}  correct {k}/{len(rr)} = {k / len(rr):.3f}   "
          f"abstained {sum(r['abstain'] for r in rr)}")
    for cell in ("C", "D"):
        for kind, test in (("neutral ('none')", lambda r: r["source_label"] == "none"),
                           ("other fallacy  ", lambda r: r["source_label"] != "none")):
            rr = [r for r in rows if r["cell"] == cell and test(r)]
            k = sum(r["correct"] for r in rr)
            e(f"     cell {cell}, {kind}: correct {k}/{len(rr)} = {k / len(rr):.3f}")
    e()
    hard = [(r["g"], r["correct"]) for r in rows if r["cell"] in ("B", "C")]
    easy = [(r["g"], r["correct"]) for r in rows if r["cell"] in ("A", "D")]
    hb = hc.balacc(hard)
    hlo, hhi = ci(hard)
    eb = hc.balacc(easy)
    elo, ehi = ci(easy)
    e(f"PRIMARY  hard cells (B+C) balanced accuracy = {hb:.3f}  [95% CI {hlo:.3f}-{hhi:.3f}]")
    if hb >= 0.75 and hlo > 0.60:
        v = "SUPPORTED"
    elif hb <= 0.60 or hlo <= 0.50:
        v = "NOT SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    e(f"VERDICT under the pre-registered rule: {v}")
    e()
    e("Secondary, descriptive (do not change the verdict):")
    e(f"  easy cells (A+D) balanced accuracy = {eb:.3f}  [95% CI {elo:.3f}-{ehi:.3f}]")
    clo, chi = cluster_ci([r for r in rows if r["cell"] in ("B", "C")])
    e(f"  hard cells, bootstrap clustered by writer: 95% CI {clo:.3f}-{chi:.3f}")
    if LOCAL.exists():
        text = {r["id"]: r["text"] for r in csv.DictReader(open(LOCAL, encoding="utf-8"))}
        strict = []
        for r in rows:
            m = 1 if STRICT_MARKER.search(text[r["id"]]) else 0
            if (r["g"] == 1 and m == 0) or (r["g"] == 0 and m == 1):
                strict.append((r["g"], r["correct"]))
        pos = [c for g, c in strict if g == 1]
        neg = [c for g, c in strict if g == 0]
        if pos and neg:
            slo, shi = ci(strict)
            e(f"  hard cells under the stricter marker (no bare 'or'): n={len(strict)} "
              f"(FD {len(pos)}, non-FD {len(neg)})  balanced accuracy = "
              f"{hc.balacc(strict):.3f}  [95% CI {slo:.3f}-{shi:.3f}]")
        else:
            e(f"  hard cells under the stricter marker (no bare 'or'): FD {len(pos)}, "
              f"non-FD {len(neg)} -- one class empty, not computed")
        lex = lexical_control(rows)
        if lex:
            e(f"  supervised lexical control on hard cells (TF-IDF+LR, 5-fold; different "
              f"protocol, context only): balanced accuracy = {lex[0]:.3f}  "
              f"[95% CI {lex[1][0]:.3f}-{lex[1][1]:.3f}]")
    e("  marker-only rule on hard cells: 0.000 by construction.")
    e()
    e("Limits: gold = the fallacy the writer was ASSIGNED to produce, not an independent")
    e("reader's judgement; whole comments (median about 58 words) in a sentence-level")
    e("prompt; cells defined by a regular expression; one model, one run.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nsaved {REPORT.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="re-print the report from saved predictions")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--backend", default="openai")
    a = ap.parse_args()
    if not a.rescore:
        run_model(a.model, a.backend)
    elif not PRED.exists():
        raise SystemExit(f"{PRED.name} not found; nothing to rescore")
    report(a.model)


if __name__ == "__main__":
    main()
