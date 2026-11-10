# -*- coding: utf-8 -*-
"""
25_structural_2x2.py
Structural-sensitivity test for false_dilemma, designed to close the gap exposed by
the lexical baseline (24_lexical_baseline.py).

DESIGN. 2x2 = gold structure (false dilemma yes/no) x lexical marker (either/or,
"only two", etc. present/absent). Cells:
  A marked + FD      B unmarked + FD
  C marked + not-FD  D unmarked + not-FD
Marker and gold AGREE in A and D; they DISAGREE in B and C ("hard cells").
A system reading vocabulary does well on A+D and fails on B+C; a system reading
structure does well on all four. Primary outcome: accuracy on B+C.

PRE-REGISTERED RULE (fixed before any detector run):
  structural sensitivity SUPPORTED   if detector accuracy on B+C >= 0.75 AND the
      lower bound of its 95% bootstrap CI exceeds the lexical control's B+C accuracy;
  NOT SUPPORTED (vocabulary-driven)  if detector B+C accuracy <= 0.60 or its CI
      overlaps the lexical control's B+C accuracy;
  otherwise INCONCLUSIVE.
Abstentions count as errors on B+C (conservative) and are also reported separately.

CAVEATS: 56 author-written items (14/cell) -> a probe, not a benchmark; the author
wrote both gold and items, so independent annotation is still required before any
strong claim. Cell B items use "with us / against us" and guilt-by-association
phrasing; whether those are false dilemmas is itself a construct choice (cell B is
the most contestable cell and should be checked by independent annotators).

Usage:
  python 25_structural_2x2.py --lexical            # no API
  python 25_structural_2x2.py --detector           # needs OPENAI_API_KEY
"""
import argparse, csv, importlib.util, os, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "structural_2x2_fd.csv")


def load(path=None):
    return list(csv.DictReader(open(path or DATA, encoding="utf-8")))


def boot_ci(correct, reps=2000, seed=0):
    rng = random.Random(seed); n = len(correct); v = []
    for _ in range(reps):
        v.append(sum(correct[rng.randrange(n)] for _ in range(n)) / n)
    v.sort()
    return v[int(0.025 * reps)], v[int(0.975 * reps) - 1]


def report(name, pred, rows):
    gold = np.array([int(r["gold_false_dilemma"]) for r in rows])
    cell = np.array([r["cell"] for r in rows])
    ok = (np.array(pred) == gold).astype(int)
    print(f"\n[{name}]")
    for c in "ABCD":
        m = cell == c
        print(f"  cell {c}: {ok[m].sum()}/{m.sum()} = {ok[m].mean():.3f}")
    ad = np.isin(cell, ["A", "D"]); bc = np.isin(cell, ["B", "C"])
    lo, hi = boot_ci(ok[bc].tolist())
    print(f"  A+D (marker agrees with gold): {ok[ad].mean():.3f}")
    print(f"  B+C (marker disagrees)  PRIMARY: {ok[bc].mean():.3f}  [95% CI {lo:.3f}-{hi:.3f}]  n={bc.sum()}")
    return ok[bc].mean(), lo, hi


def lexical(rows):
    X = [r["text"] for r in rows]; y = np.array([int(r["gold_false_dilemma"]) for r in rows])
    pred = np.zeros(len(y), int)
    # v2 sets carry a topic column: hold out whole topics (all 4 cells) so topic
    # vocabulary cannot be learned across folds. v1 has no topic: plain LOO.
    if "topic" in rows[0]:
        splits = LeaveOneGroupOut().split(X, y, [r["topic"] for r in rows])
        print("Lexical control: held-out TOPICS (leave-one-topic-out).")
    else:
        splits = LeaveOneOut().split(X)
    for tr, te in splits:
        c = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
        c.fit([X[i] for i in tr], y[tr]); pred[te] = c.predict([X[i] for i in te])
    print("Lexical control: TF-IDF 1-2gram + LR, leave-one-out, supervised on this same set")
    print("(favours the lexical model: it sees in-domain labels; the detector is zero-shot).")
    m, lo, hi = report("LEXICAL (LOO)", pred, rows)
    # marker-only rule: predicts FD iff marker present
    mk = [int(r["marker"]) for r in rows]
    report("MARKER-ONLY RULE (FD iff marker present)", mk, rows)
    return m, lo, hi


def detector(rows, model="gpt-4o-mini"):
    spec = importlib.util.spec_from_file_location("fd23", os.path.join(HERE, "23_logic_fd_eval.py"))
    m23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m23)
    asker = m23.make_asker("openai", model)
    pred, ab = [], 0
    out = []
    for r in rows:
        lab = m23.parse_label(m23.call_with_retry(asker, m23.SYSTEM,
              m23.PROMPT.format(defs=m23.DEFS, text=r["text"])))
        ab += lab == m23.ABSTAIN
        pred.append(1 if lab == m23.TARGET else 0)   # abstention -> 0 (error on FD items)
        out.append({"id": r["id"], "cell": r["cell"], "gold": r["gold_false_dilemma"],
                    "label": lab, "text": r["text"]})
    with open(os.path.join(HERE, "structural_2x2_detector_predictions.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"Detector: {model}, frozen v2 prompt, temperature 0. Abstentions: {ab}")
    report("DETECTOR", pred, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical", action="store_true")
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(); rows = load(a.data)
    if a.lexical or not a.detector:
        lexical(rows)
    if a.detector:
        detector(rows, a.model)
