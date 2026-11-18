# -*- coding: utf-8 -*-
"""
35_annotation_manifest.py
Draws the annotation set of the deferred human-labelling stage (Appendix A of the
multi-model pre-registration). Nothing else of that stage is executed.

Set: all 110 comments of cell B, all 110 of cell D, 50 of cell A drawn by seed, and
50 of cell C drawn by seed (25 neutral, source_label "none", and 25 with another
fallacy). The output holds identifiers and cells only, no comment text, in an order
shuffled by seed.

Usage (from the repository root):  python benchmark/35_annotation_manifest.py
Writes benchmark/mm_annotation_manifest.csv and prints its SHA-256.

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import hashlib
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cocolofa_fd_sample_manifest.csv"
OUT = HERE / "mm_annotation_manifest.csv"
MANIFEST_SHA256 = "e35b540ec55962eb9de92b9887bd56647f563944b31050163c8fad531b2b9975"
SEED = 20261118


def main():
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise SystemExit("sample manifest differs from the pre-registered one; stopping")
    if OUT.exists():
        raise SystemExit(f"{OUT.name} already exists; it is drawn once")
    m = pd.read_csv(MANIFEST).sort_values("id")
    rng = random.Random(SEED)
    pick = lambda df, k: rng.sample(sorted(df.id.tolist()), k)
    ids = (m[m.cell == "B"].id.tolist() + m[m.cell == "D"].id.tolist()
           + pick(m[m.cell == "A"], 50)
           + pick(m[(m.cell == "C") & (m.source_label == "none")], 25)
           + pick(m[(m.cell == "C") & (m.source_label != "none")], 25))
    rng.shuffle(ids)
    cell = m.set_index("id").cell
    out = pd.DataFrame({"order": range(1, len(ids) + 1), "id": ids, "cell": [cell[i] for i in ids]})
    assert len(out) == 320 and out.id.is_unique
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"saved {OUT.name}: {out.cell.value_counts().to_dict()}  "
          f"SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
