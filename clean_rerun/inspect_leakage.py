"""Inspect PropXplain English: ID scheme, class balance, and leakage."""
from pathlib import Path
import re
import pandas as pd

DATA = Path("data")
TEXT, LAB, SID = "input", "Label", "sample_id"

def load(split):
    df = pd.read_csv(DATA / f"propxplain_english_{split}.csv")
    df = df[df[LAB].astype(str).str.strip().isin(["propagandistic", "non-propagandistic"])].copy()
    df[LAB] = df[LAB].astype(str).str.strip()
    df[TEXT] = df[TEXT].astype(str)
    return df

tr, va, te = load("train"), load("validation"), load("test")
print("rows: train", len(tr), "val", len(va), "test", len(te))
for name, d in [("train", tr), ("val", va), ("test", te)]:
    vc = d[LAB].value_counts()
    print(f"  {name}: prop={vc.get('propagandistic',0)} nonprop={vc.get('non-propagandistic',0)}")

print("\nsample_id examples (test):", list(te[SID].head(8)))

def norm(s):  # audit-style normalization
    return re.sub(r"\s+", " ", str(s).strip().lower())

def prefix(sid):  # remove last underscore component
    s = str(sid)
    return s.rsplit("_", 1)[0] if "_" in s else s

# exact-text overlap train vs test (normalized)
tr_norm = set(tr[TEXT].map(norm))
te_norm_list = te[TEXT].map(norm)
exact_dup_mask = te_norm_list.isin(tr_norm)
print("\n[exact-text] test rows whose normalized text appears in train:", int(exact_dup_mask.sum()), "/", len(te))

# article-prefix overlap
tr_pref = set(tr[SID].map(prefix))
te_pref = te[SID].map(prefix)
te_pref_in_train = te_pref.isin(tr_pref)
print("[prefix] distinct test prefixes:", te_pref.nunique(),
      "| distinct train prefixes:", tr[SID].map(prefix).nunique())
print("[prefix] test rows whose article-prefix also in train:", int(te_pref_in_train.sum()), "/", len(te))
print("[prefix] distinct test prefixes also in train:",
      len(set(te_pref) & tr_pref), "/", te_pref.nunique())

# what would survive an article-disjoint purge of test?
survive = (~te_pref_in_train) & (~exact_dup_mask)
print("\ntest rows surviving article-disjoint + exact-dedup purge:", int(survive.sum()), "/", len(te))
if survive.sum() > 0:
    sv = te[survive]
    print("  surviving class balance:", dict(sv[LAB].value_counts()))

# pooled article-level view for a possible re-split
alldf = pd.concat([tr.assign(orig="train"), va.assign(orig="val"), te.assign(orig="test")], ignore_index=True)
alldf["pref"] = alldf[SID].map(prefix)
print("\npooled rows:", len(alldf), "| distinct articles (prefixes):", alldf["pref"].nunique())
# articles that span multiple original splits
span = alldf.groupby("pref")["orig"].nunique()
print("articles spanning >1 original split:", int((span > 1).sum()), "/", alldf["pref"].nunique())
