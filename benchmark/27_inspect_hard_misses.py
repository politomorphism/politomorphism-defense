# -*- coding: utf-8 -*-
"""
27_inspect_hard_misses.py
Lists the hard-cell errors from the LOGIC re-analysis so they can be read by hand:
  (a) gold false dilemma WITHOUT a marker that the detector did not flag (or abstained on)
  (b) gold non-false-dilemma WITH a marker that the detector flagged (or abstained on)
Purpose: decide, by reading, whether each error is a detector miss, an arguable LOGIC
label, or a definitional/annotation artifact. This is qualitative and must not be used
to re-draw the cells or change the pre-registered verdict.
Usage:  python 27_inspect_hard_misses.py logic_fd_predictions.csv
Writes: hard_misses_for_review.csv next to the predictions file.
"""
import csv, os, re, sys
MK = re.compile(r"\b(either|or)\b|only two|two (options|choices|sides)|third way|no (other|middle)", re.I)
path = sys.argv[1]
rows = list(csv.DictReader(open(path, encoding="utf-8")))
out = []
for r in rows:
    g = int(r["gold_false_dilemma"]); m = 1 if MK.search(r["text"]) else 0
    flag = int(r["flag_fd"]); ab = int(r["abstain"])
    if g == 1 and m == 0 and not (flag == 1 and ab == 0):
        out.append(("FD_without_marker_not_flagged", r["stage1_pred"], r["source_label"], r["text"]))
    if g == 0 and m == 1 and not (flag == 0 and ab == 0):
        out.append(("nonFD_with_marker_flagged_or_abstained", r["stage1_pred"], r["source_label"], r["text"]))
for kind, pred, src, text in out:
    print(f"[{kind}] detector={pred} | LOGIC label={src}\n   {text}\n")
dest = os.path.join(os.path.dirname(os.path.abspath(path)), "hard_misses_for_review.csv")
with open(dest, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["kind", "detector_label", "logic_label", "text", "my_reading"]); 
    for k, p, s, t in out: w.writerow([k, p, s, t, ""])
print(f"{len(out)} errors listed; blank 'my_reading' column saved to {dest}")
