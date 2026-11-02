"""Baseline across multiple article-disjoint split seeds.
LR is deterministic on a fixed split, so the baseline's variability comes from
WHICH article-disjoint partition is drawn. We vary the split seed and report the
spread of propagandistic-class metrics. (Transformer seed-variability is separate
and must be run where the model can be downloaded.)"""
from pathlib import Path
import re, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

DATA=Path("data"); TEXT,LAB,SID="input","Label","sample_id"; POS="propagandistic"
def load(s):
    d=pd.read_csv(DATA/f"propxplain_english_{s}.csv"); d[LAB]=d[LAB].astype(str).str.strip()
    d=d[d[LAB].isin(["propagandistic","non-propagandistic"])].copy(); d[TEXT]=d[TEXT].astype(str)
    return d[[SID,TEXT,LAB]]
pool=pd.concat([load("train"),load("validation"),load("test")],ignore_index=True)
pool["_n"]=pool[TEXT].map(lambda s:re.sub(r"\s+"," ",s.strip().lower()))
pool=pool.drop_duplicates("_n").reset_index(drop=True)
pool["article"]=pool[SID].map(lambda s:s.rsplit("_",1)[0] if "_" in str(s) else str(s))

rows=[]
for seed in range(10):
    y=(pool[LAB]==POS).astype(int).values; g=pool["article"].values
    itr,ite=next(GroupShuffleSplit(1,test_size=0.15,random_state=seed).split(pool,y,g))
    tr=pool.iloc[itr]; te=pool.iloc[ite]
    m=Pipeline([("v",TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,sublinear_tf=True)),
                ("c",LogisticRegression(max_iter=2000,class_weight="balanced"))]).fit(tr[TEXT],tr[LAB])
    pc=list(m.classes_).index(POS); s=m.predict_proba(te[TEXT])[:,pc]; pred=m.predict(te[TEXT])
    yte=(te[LAB]==POS).astype(int).values
    rows.append(dict(seed=seed,n_test=len(te),prop=int(yte.sum()),
        precision=precision_score(te[LAB],pred,pos_label=POS,zero_division=0),
        recall=recall_score(te[LAB],pred,pos_label=POS,zero_division=0),
        f1=f1_score(te[LAB],pred,pos_label=POS,zero_division=0),
        ap=average_precision_score(yte,s)))
df=pd.DataFrame(rows); pd.set_option("display.width",120)
print(df.round(3).to_string(index=False))
print("\nAcross 10 article-disjoint splits (mean ± sd; min–max):")
for c in ["precision","recall","f1","ap"]:
    v=df[c]; print(f"  {c:10s} {v.mean():.3f} ± {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]")
df.to_csv("clean/baseline_multisplit.csv",index=False)
