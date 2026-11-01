"""Article-clustered bootstrap CIs on transformer - baseline differences."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
OUT = Path("clean"); RNG = np.random.default_rng(42); B = 2000
b = pd.read_csv(OUT/"baseline_test_scores.csv"); t = pd.read_csv(OUT/"transformer_test_scores.csv")
m = b.merge(t, on="sample_id", suffixes=("_b","_t"))
art = m["article_b"].values; y = m["y_true_b"].values
bs = m["baseline_score"].values; ts = m["transformer_score"].values
bp = m["baseline_pred"].values; tp = m["transformer_pred"].values

def f1(yv, pred):
    tpp=((pred==1)&(yv==1)).sum(); fp=((pred==1)&(yv==0)).sum(); fn=((pred==0)&(yv==1)).sum()
    p=tpp/(tpp+fp) if tpp+fp else 0; r=tpp/(tpp+fn) if tpp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
def recall(yv,pred):
    tpp=((pred==1)&(yv==1)).sum(); fn=((pred==0)&(yv==1)).sum()
    return tpp/(tpp+fn) if tpp+fn else 0
def matched_budget_recall(yv, base_pred, tscore):
    nb=int(base_pred.sum()); k=min(nb,len(tscore))
    thr_idx=np.argsort(-tscore)[:k]; pred=np.zeros(len(tscore),int); pred[thr_idx]=1
    return recall(yv,base_pred), recall(yv,pred)

arts=np.unique(art); by={a:np.where(art==a)[0] for a in arts}
dAP=[]; dF1=[]; dRB=[]
for _ in range(B):
    idx=np.concatenate([by[a] for a in RNG.choice(arts,size=len(arts),replace=True)])
    yy=y[idx]
    if yy.sum()==0 or yy.sum()==len(yy): continue
    dAP.append(average_precision_score(yy,ts[idx])-average_precision_score(yy,bs[idx]))
    dF1.append(f1(yy,tp[idx])-f1(yy,bp[idx]))
    rb_b,rb_t=matched_budget_recall(yy,bp[idx],ts[idx]); dRB.append(rb_t-rb_b)
def ci(v): v=np.array(v); return np.percentile(v,2.5),np.percentile(v,97.5),np.mean(v)
for nm,v in [("ΔAP (tf-base)",dAP),("ΔF1 default",dF1),("Δrecall @ matched budget",dRB)]:
    lo,hi,mean=ci(v); print(f"{nm:26s} mean={mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (frac>0 = {np.mean(np.array(v)>0):.3f})")
