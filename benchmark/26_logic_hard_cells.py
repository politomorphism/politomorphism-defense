# -*- coding: utf-8 -*-
"""
26_logic_hard_cells.py
Structural-sensitivity re-analysis on INDEPENDENTLY WRITTEN items (LOGIC), with NO new
model calls: it re-scores the predictions already saved by 23_logic_fd_eval.py.

WHY. Two author-written 2x2 sets (structural_2x2_fd.csv v1, _v2.csv) both leaked
register: a TF-IDF model scored 0.786 (v1) and 0.893 (v2) on the hard cells, so an
author-written probe cannot show the detector reads structure. LOGIC items were written
by others, so the natural marker/gold disagreement cells are a cleaner test.

CELLS (fixed here BEFORE looking at detector results on them):
  marker M(t) = regex below (either / or / only two / two options|choices|sides /
                third way / no other|middle).
  HARD cells: gold FD without marker (43 items) and gold non-FD with marker (14 items).
  EASY cells: marker agrees with gold.
PRIMARY OUTCOME: balanced accuracy on hard cells
  BA = 1/2 * ( correct FD-without-marker / all FD-without-marker
             + correct non-FD-with-marker / all non-FD-with-marker ),
  each ratio over ALL items of its class. An abstention is an incorrect answer in either
  class (a non-FD abstention is NOT credited as "non-FD"). Plain accuracy would be inflated
  by the unequal class mix. 95% CI: 2000 item-level bootstrap replicates (seed 0).
TIMING (stated precisely): the regex and the decision rule were fixed before the aggregate
  hard-cell score of the detector was computed; the detector predictions already existed
  (7 November run) and had not been inspected by the analyst. The thresholds were set after a
  supervised lexical control had been computed on the same cells (balanced accuracy ~0.75),
  so they are not independent of that information. None of the 20 hard-cell errors had been
  read when the rule was fixed. The script prints the SHA-256 of the predictions file so
  the exact input is recorded.
DECISION RULE (fixed before the aggregate hard-cell score was computed; a re-analysis plan, not an independent pre-registration): SUPPORTED if hard-cell balanced accuracy >= 0.75 and CI
  lower bound > 0.60; NOT SUPPORTED if <= 0.60 or CI lower bound <= 0.50; else INCONCLUSIVE.
  A marker-only rule scores 0 on hard cells by construction.
CAVEATS: pretraining exposure to LOGIC unresolved; 57 hard items, only 14 non-FD; hard
  cells defined by a regex, not by annotators; a supervised lexical model (shown for
  context, 5-fold) is a different protocol from the zero-shot detector.

Usage:  python 26_logic_hard_cells.py logic_fd_predictions.csv logic_false_dilemma_eval.csv [--exclude-review]
  --exclude-review keeps only rows with to_review=0 (joined by id), i.e. the 269-item set
  scored with the predictions in the given file (the 273-item run).
"""
import csv, hashlib, random, re, sys
import numpy as np

MK = re.compile(r"\b(either|or)\b|only two|two (options|choices|sides)|third way|no (other|middle)", re.I)

def boot(vals, reps=2000, seed=0):
    rng = random.Random(seed); n = len(vals); out = []
    for _ in range(reps):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        out.append(s)
    return out

def balacc(sample):
    pos = [c for g, c in sample if g == 1]; neg = [c for g, c in sample if g == 0]
    if not pos or not neg: return float("nan")
    return 0.5 * (sum(pos) / len(pos) + sum(neg) / len(neg))

def main(pred_path, data_path, exclude_review=False):
    gold_rows = {r["id"]: r for r in csv.DictReader(open(data_path, encoding="utf-8"))}
    P = [r for r in csv.DictReader(open(pred_path, encoding="utf-8"))]
    print("predictions file SHA-256:", hashlib.sha256(open(pred_path, "rb").read()).hexdigest())
    if exclude_review:
        P = [r for r in P if gold_rows[r["id"]].get("to_review") == "0"]
        print("--exclude-review: rows with to_review=0 only; n =", len(P))
    else:
        print("all rows in the predictions file; n =", len(P))
    items = []
    for r in P:
        g = int(r["gold_false_dilemma"]); m = 1 if MK.search(r["text"]) else 0
        correct = int(int(r["flag_fd"]) == g and int(r["abstain"]) == 0)  # abstention = error
        items.append((g, m, correct))
    # Overall confusion (all rows used), with abstentions as their own column
    for g_, name_ in ((1, "FD    "), (0, "non-FD")):
        rr = [r for r in P if int(r["gold_false_dilemma"]) == g_]
        a_ = sum(int(r["abstain"]) == 1 for r in rr)
        f_ = sum(int(r["abstain"]) == 0 and int(r["flag_fd"]) == 1 for r in rr)
        n_ = sum(int(r["abstain"]) == 0 and int(r["flag_fd"]) == 0 for r in rr)
        print(f"overall confusion  reference {name_}: predicted FD={f_}  predicted non-FD={n_}  abstained={a_}  total={len(rr)}")
    pos = [r for r in P if int(r["gold_false_dilemma"]) == 1]; neg = [r for r in P if int(r["gold_false_dilemma"]) == 0]
    TP = sum(int(r["abstain"]) == 0 and int(r["flag_fd"]) == 1 for r in pos)
    FN = sum(int(r["abstain"]) == 0 and int(r["flag_fd"]) == 0 for r in pos); AB = sum(int(r["abstain"]) == 1 for r in pos)
    FP = sum(int(r["abstain"]) == 0 and int(r["flag_fd"]) == 1 for r in neg)
    print(f"F1 on decided items (abstentions excluded): {2*TP/(2*TP+FP+FN):.4f}")
    print(f"F1 with abstentions on positives counted as misses: {2*TP/(2*TP+FP+FN+AB):.4f}")
    hard = [(g, c) for g, m, c in items if (g == 1 and m == 0) or (g == 0 and m == 1)]
    easy = [(g, c) for g, m, c in items if (g == 1 and m == 1) or (g == 0 and m == 0)]
    for name, s in (("HARD (marker disagrees)", hard), ("EASY (marker agrees)", easy)):
        pos = [c for g, c in s if g == 1]; neg = [c for g, c in s if g == 0]
        b = balacc(s); reps = sorted(v for v in (balacc(x) for x in boot(s)) if v == v)
        lo, hi = reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]
        print(f"[{name}] n={len(s)}  FD items {sum(pos)}/{len(pos)}  non-FD items {sum(neg)}/{len(neg)}"
              f"  balanced accuracy={b:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
        if name.startswith("HARD"):
            hb, hlo = b, lo
    if hb >= 0.75 and hlo > 0.60: v = "SUPPORTED (not marker-driven)"
    elif hb <= 0.60 or hlo <= 0.50: v = "NOT SUPPORTED"
    else: v = "INCONCLUSIVE"
    print(f"Verdict on hard cells under the fixed re-analysis rule: {v}")
    print("Marker-only rule on hard cells: 0.000 by construction. Supervised lexical (5-fold) on the same hard cells: balanced accuracy ~0.75 (context only, different protocol).")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], exclude_review="--exclude-review" in sys.argv)
