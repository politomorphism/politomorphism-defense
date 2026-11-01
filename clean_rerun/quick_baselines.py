"""Alternative lexical baselines on the SAME clean article-disjoint split,
to give a small comparison table (reviewer request). Prop-class metrics."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

OUT = Path("clean"); TEXT, LAB, POS = "input", "Label", "propagandistic"
tr = pd.read_csv(OUT/"clean_train.csv"); te = pd.read_csv(OUT/"clean_test.csv")

def ev(name, model=None, trivial=False):
    if trivial:
        pred = np.array(["non-propagandistic"]*len(te))
    else:
        model.fit(tr[TEXT], tr[LAB]); pred = model.predict(te[TEXT])
    p = precision_score(te[LAB], pred, pos_label=POS, zero_division=0)
    r = recall_score(te[LAB], pred, pos_label=POS, zero_division=0)
    f = f1_score(te[LAB], pred, pos_label=POS, zero_division=0)
    print(f"{name:32s} P={p:.3f} R={r:.3f} F1={f:.3f}")

ev("trivial (always non-prop)", trivial=True)
ev("TF-IDF unigram + LR", Pipeline([("v",TfidfVectorizer(lowercase=True,ngram_range=(1,1),min_df=2,sublinear_tf=True)),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))]))
ev("TF-IDF char 3-5grams + LR", Pipeline([("v",TfidfVectorizer(lowercase=True,analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True)),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))]))
ev("TF-IDF uni+bigram + LR (main)", Pipeline([("v",TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,sublinear_tf=True)),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))]))
