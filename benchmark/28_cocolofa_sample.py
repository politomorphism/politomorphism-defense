# -*- coding: utf-8 -*-
"""
28_cocolofa_sample.py
Fix the evaluation sample for the pre-registered structural-sensitivity test of the
false_dilemma detector on CoCoLoFa (Yeh, Wan & Huang, EMNLP 2024). NO model calls.

WHAT THIS SCRIPT DOES
  1. Verifies that the three CoCoLoFa files are byte-identical to the pinned release
     (github.com/Crowd-AI-Lab/cocolofa, commit c39d45fdd57401e1f6cb674f25113dfa0304e734,
     6 October 2024) by SHA-256. It refuses to continue on any other file.
  2. Assigns every comment to one of four cells using the SAME marker object as
     26_logic_hard_cells.py (imported from that file, not retyped):
        A  gold false dilemma,      marker present   (easy)
        B  gold false dilemma,      marker absent    (HARD)
        C  gold not false dilemma,  marker present   (HARD)
        D  gold not false dilemma,  marker absent    (easy)
     Gold = the CoCoLoFa `fallacy` field ("false dilemma" vs anything else).
  3. Draws the fixed sample (seed 20261111, comment ids sorted numerically first):
        B  all 110
        A  110 at random
        C  55 labelled "none" + 55 labelled with another fallacy
        D  55 labelled "none" + 55 labelled with another fallacy
     Total 440 comments.
  4. Writes cocolofa_fd_sample_manifest.csv: comment id, split, cell, gold, source
     label, marker. It contains NO comment text, so the dataset is not re-hosted.
     Its SHA-256 is printed and must equal the value in the pre-registration.
  5. Writes cocolofa_fd_eval_local.csv (manifest + text) for the run script. That
     file is git-ignored and stays on the local machine.

The cell counts on the full corpus are printed so they can be checked against the
pre-registration: A 444, B 110, C 1346 (656 "none" + 690 other fallacy),
D 5806 (2474 "none" + 3332 other fallacy).

Usage (from the repository root):
    git clone https://github.com/Crowd-AI-Lab/cocolofa external/cocolofa
    python benchmark/28_cocolofa_sample.py external/cocolofa

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261111
PINNED = {
    "train.json": "51c9d03425ea9e065905b87d3d1f37b41fef0d09a0f37241594076282e7077a3",
    "dev.json": "23fd1fdc57930de5a5a05405d0c1084d447c98ece036be5e0b0c83793505ff65",
    "test.json": "366ae5d321616599cbe2d259a2be7bd529388d0a0107d37741dc2fa029169281",
}
MARKER_PATTERN = (r"\b(either|or)\b|only two|two (options|choices|sides)|third way|"
                  r"no (other|middle)")
N_PER_CELL = 110
N_HALF = 55
MANIFEST = HERE / "cocolofa_fd_sample_manifest.csv"
LOCAL = HERE / "cocolofa_fd_eval_local.csv"


def load_marker():
    spec = importlib.util.spec_from_file_location("hard_cells", HERE / "26_logic_hard_cells.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.MK.pattern != MARKER_PATTERN:
        raise SystemExit("marker in 26_logic_hard_cells.py differs from the pre-registered pattern")
    return mod.MK


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(corpus_dir):
    corpus_dir = Path(corpus_dir)
    for name, want in PINNED.items():
        got = sha256(corpus_dir / name)
        if got != want:
            raise SystemExit(f"{name}: SHA-256 {got} differs from the pinned release; stopping")
    print("corpus files match the pinned CoCoLoFa release (3/3 SHA-256 identical)")

    mk = load_marker()
    rows = []
    for split in ("train", "dev", "test"):
        for art in json.loads((corpus_dir / f"{split}.json").read_text(encoding="utf-8")):
            for c in art["comments"]:
                gold = 1 if c["fallacy"] == "false dilemma" else 0
                marker = 1 if mk.search(c["comment"]) else 0
                cell = {(1, 1): "A", (1, 0): "B", (0, 1): "C", (0, 0): "D"}[(gold, marker)]
                rows.append({"id": c["id"], "split": split, "cell": cell,
                             "gold_false_dilemma": gold, "source_label": c["fallacy"],
                             "marker": marker, "worker_id": c["worker_id"],
                             "text": c["comment"]})
    rows.sort(key=lambda r: int(r["id"]))

    def pick(cell, kind=None):
        out = [r for r in rows if r["cell"] == cell]
        if kind == "none":
            out = [r for r in out if r["source_label"] == "none"]
        elif kind == "other":
            out = [r for r in out if r["source_label"] != "none"]
        return out

    print("full-corpus cell counts:",
          f"A={len(pick('A'))}", f"B={len(pick('B'))}",
          f"C={len(pick('C'))} (none {len(pick('C', 'none'))}, other {len(pick('C', 'other'))})",
          f"D={len(pick('D'))} (none {len(pick('D', 'none'))}, other {len(pick('D', 'other'))})")

    rng = random.Random(SEED)
    sample = list(pick("B"))
    if len(sample) != N_PER_CELL:
        raise SystemExit(f"cell B has {len(sample)} items, expected {N_PER_CELL}; stopping")
    sample += rng.sample(pick("A"), N_PER_CELL)
    sample += rng.sample(pick("C", "none"), N_HALF)
    sample += rng.sample(pick("C", "other"), N_HALF)
    sample += rng.sample(pick("D", "none"), N_HALF)
    sample += rng.sample(pick("D", "other"), N_HALF)
    sample.sort(key=lambda r: int(r["id"]))

    cols = ["id", "split", "cell", "gold_false_dilemma", "source_label", "marker", "worker_id"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in sample:
            w.writerow({k: r[k] for k in cols})
    with open(LOCAL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["text"], lineterminator="\n")
        w.writeheader()
        for r in sample:
            w.writerow(r)

    print(f"sample: {len(sample)} comments "
          + ", ".join(f"{c}={sum(1 for r in sample if r['cell'] == c)}" for c in "ABCD")
          + f"; distinct writers: {len({r['worker_id'] for r in sample})}")
    print(f"wrote {MANIFEST.name} (no text)  SHA-256: {sha256(MANIFEST)}")
    print(f"wrote {LOCAL.name} (with text, git-ignored, local only)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python benchmark/28_cocolofa_sample.py <path to cocolofa clone>")
    main(sys.argv[1])
