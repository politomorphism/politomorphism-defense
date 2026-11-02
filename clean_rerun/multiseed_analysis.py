"""Full multi-seed paired analysis: transformer (5 seeds) vs baseline (fixed, split 42)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import binomtest

base = pd.read_csv("clean/baseline_test_scores.csv")  # split 42 baseline
seeds = [42,1,2,3,4]
tf = {s: pd.read_csv(f"ms/seed{s}.csv") for s in seeds}

# align everything on sample_id
yb = base.set_index("sample_id")["y_true"]
bs = base.set_index("sample_id")["baseline_score"]
bp = base.set_index("sample_id")["baseline_pred"]
art = base.set_index("sample_id")["article"]
ids = base["sample_id"].values

def prf(y,pred):
    tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return p,r,(2*p*r/(p+r) if p+r else 0)

# baseline reference
yv=yb.values; p_b,r_b,f_b=prf(yv,bp.values); ap_b=average_precision_score(yv,bs.values)
nb=int(bp.sum())
print(f"BASELINE (split 42): P={p_b:.3f} R={r_b:.3f} F1={f_b:.3f} AP={ap_b:.3f} flagged={nb}\n")

rows=[]
for s in seeds:
    d=tf[s].set_index("sample_id")
    ts=d["transformer_score"].reindex(ids).values
    tp_=d["transformer_pred"].reindex(ids).values
    p,r,f=prf(yv,tp_); ap=average_precision_score(yv,ts)
    # matched review budget: flag top-nb by transformer score
    order=np.argsort(-ts); mb=np.zeros(len(ts),int); mb[order[:nb]]=1
    _,r_mb,_=prf(yv,mb)
    # McNemar on correctness vs baseline
    bc=(bp.values==yv).astype(int); tc=(tp_==yv).astype(int)
    b_only=int(((bc==1)&(tc==0)).sum()); t_only=int(((bc==0)&(tc==1)).sum())
    pval=binomtest(min(b_only,t_only),b_only+t_only,0.5).pvalue if (b_only+t_only)>0 else 1.0
    rows.append(dict(seed=s,P=p,R=r,F1=f,AP=ap,dAP=ap-ap_b,R_matched=r_mb,
                     dR_matched=r_mb-r_b,base_only=b_only,tf_only=t_only,mcnemar_p=pval,
                     beats_AP=ap>ap_b, beats_matched=r_mb>r_b))
df=pd.DataFrame(rows)
pd.set_option("display.width",160)
print(df.round(3).to_string(index=False))
print("\n--- Summary across 5 seeds ---")
print(f"AP:        {df.AP.mean():.3f} ± {df.AP.std():.3f}   (baseline {ap_b:.3f}); ΔAP mean {df.dAP.mean():+.3f}")
print(f"  transformer AP > baseline AP in {df.beats_AP.sum()}/5 seeds")
print(f"Recall @ matched budget: {df.R_matched.mean():.3f} ± {df.R_matched.std():.3f} (baseline {r_b:.3f}); Δ mean {df.dR_matched.mean():+.3f}")
print(f"  transformer matched-recall > baseline in {df.beats_matched.sum()}/5 seeds")
print(f"McNemar: transformer-only-correct {df.tf_only.mean():.0f} vs baseline-only {df.base_only.mean():.0f} (mean); max p = {df.mcnemar_p.max():.4f}")
print(f"  significant (p<0.05) in {(df.mcnemar_p<0.05).sum()}/5 seeds")
