"""
05_transformer_multiseed.py
Multi-seed transformer runs for statistical robustness (reviewer request).

Fixes the article-disjoint SPLIT (seed 42, identical to Working Paper II) and
varies only the TRAINING seed, so the spread measured here is training-seed
variability — the quantity a single run cannot estimate. Fine-tunes DistilBERT
once per seed and saves that seed's per-paragraph test scores.

Run on your machine (it can reach Hugging Face). ~35 min per seed on CPU, so
SEEDS = [42, 1, 2, 3, 4] is ~3 hours. You can stop after any number of seeds
and send whatever clean/transformer_scores_seed*.csv files exist; the analysis
scales to however many are present.
"""
import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

DATA=Path("data"); OUT=Path("clean"); OUT.mkdir(exist_ok=True)
TEXT,LAB,SID="input","Label","sample_id"; POS="propagandistic"
LABELS=["non-propagandistic","propagandistic"]
label2id={n:i for i,n in enumerate(LABELS)}; id2label={i:n for i,n in enumerate(LABELS)}
MODEL="distilbert-base-uncased"; MAXLEN=256; EPOCHS=3; TB=8; EB=16; LR=2e-5
SPLIT_SEED=42                       # fixed: same split as Working Paper II
SEEDS=[42,1,2,3,4]                  # training seeds to sweep (edit to taste)

def load(s):
    d=pd.read_csv(DATA/f"propxplain_english_{s}.csv"); d[LAB]=d[LAB].astype(str).str.strip()
    d=d[d[LAB].isin(LABELS)].copy(); d[TEXT]=d[TEXT].astype(str); return d[[SID,TEXT,LAB]]

def clean_split():
    pool=pd.concat([load("train"),load("validation"),load("test")],ignore_index=True)
    pool["_n"]=pool[TEXT].map(lambda s:re.sub(r"\s+"," ",s.strip().lower()))
    pool=pool.drop_duplicates("_n").reset_index(drop=True)
    pool["article"]=pool[SID].map(lambda s:s.rsplit("_",1)[0] if "_" in str(s) else str(s))
    y=(pool[LAB]==POS).astype(int).values; g=pool["article"].values
    itrf,ite=next(GroupShuffleSplit(1,test_size=0.15,random_state=SPLIT_SEED).split(pool,y,g))
    trf=pool.iloc[itrf].reset_index(drop=True)
    itr,iva=next(GroupShuffleSplit(1,test_size=0.12,random_state=SPLIT_SEED).split(trf,None,trf["article"].values))
    return trf.iloc[itr].reset_index(drop=True), trf.iloc[iva].reset_index(drop=True), pool.iloc[ite].reset_index(drop=True)

class WT(Trainer):
    def __init__(self,*a,cw=None,**k): super().__init__(*a,**k); self.cw=cw
    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        y=inputs.pop("labels"); o=model(**inputs); lg=o.logits
        w=self.cw.to(lg.device) if self.cw is not None else None
        loss=nn.CrossEntropyLoss(weight=w)(lg.view(-1,model.config.num_labels),y.view(-1))
        return (loss,o) if return_outputs else loss

def run_seed(seed, tr, va, te, cw, tok):
    set_seed(seed)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2,id2label=id2label,label2id=label2id)
    def ds(d):
        d=d.copy(); d["labels"]=d[LAB].map(label2id).astype(int)
        return Dataset.from_pandas(d[[TEXT,"labels"]],preserve_index=False).map(
            lambda b: tok(b[TEXT],truncation=True,max_length=MAXLEN),batched=True,remove_columns=[TEXT])
    args=TrainingArguments(output_dir=f"ms_out_{seed}",num_train_epochs=EPOCHS,per_device_train_batch_size=TB,
        per_device_eval_batch_size=EB,learning_rate=LR,logging_steps=200,save_strategy="no",report_to="none",seed=seed)
    tr_t=ds(tr); te_t=ds(te)
    trainer=WT(model=model,args=args,train_dataset=tr_t,eval_dataset=te_t,
               data_collator=DataCollatorWithPadding(tokenizer=tok),cw=cw)
    trainer.train()
    o=trainer.predict(te_t); lg=o.predictions; lg=lg[0] if isinstance(lg,tuple) else lg
    prob=torch.softmax(torch.tensor(lg),dim=-1)[:,1].numpy()
    yhat=np.array([id2label[int(i)] for i in lg.argmax(-1)]); yte=(te[LAB].values==POS).astype(int)
    P=precision_score(te[LAB],yhat,pos_label=POS,zero_division=0)
    R=recall_score(te[LAB],yhat,pos_label=POS,zero_division=0)
    F=f1_score(te[LAB],yhat,pos_label=POS,zero_division=0); AP=average_precision_score(yte,prob)
    pd.DataFrame({"sample_id":te[SID],"article":te["article"],"y_true":yte,
                  "transformer_score":prob,"transformer_pred":(yhat==POS).astype(int)}
                 ).to_csv(OUT/f"transformer_scores_seed{seed}.csv",index=False)
    print(f"[seed {seed}] P={P:.3f} R={R:.3f} F1={F:.3f} AP={AP:.3f}  -> clean/transformer_scores_seed{seed}.csv")
    return dict(seed=seed,precision=P,recall=R,f1=F,ap=AP)

def main():
    tr,va,te=clean_split()
    print(f"split train={len(tr)} val={len(va)} test={len(te)} (test prop={int((te[LAB]==POS).sum())})")
    counts=tr[LAB].value_counts(); n=len(tr); k=len(LABELS)
    cw=torch.tensor([n/(k*int(counts.get(x,1))) for x in LABELS],dtype=torch.float)
    tok=AutoTokenizer.from_pretrained(MODEL)
    res=[run_seed(s,tr,va,te,cw,tok) for s in SEEDS]
    df=pd.DataFrame(res); df.to_csv(OUT/"transformer_multiseed_summary.csv",index=False)
    print("\nAcross",len(SEEDS),"training seeds (mean ± sd):")
    for c in ["precision","recall","f1","ap"]:
        v=df[c]; print(f"  {c:10s} {v.mean():.3f} ± {v.std():.3f}  [{v.min():.3f}, {v.max():.3f}]")
    print("\nSend the clean/transformer_scores_seed*.csv files back for the paired analysis.")

if __name__=="__main__":
    main()
