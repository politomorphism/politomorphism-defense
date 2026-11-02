"""
06_transformer_pairs.py
Score the contrastive-pairs benchmark with the fine-tuned transformer.

Trains DistilBERT on the clean article-disjoint PropXplain split (same as
Working Paper II), then scores each member of each contrastive pair and writes
pair_scores_v2.csv (columns: pair_id, member, score). Run evaluate_pairs.py
with that CSV to get the transformer's pair accuracy and compare it with the
lexical baseline.

Run on your machine (needs Hugging Face). ~35 min on CPU. Put this file and
contrastive_pairs_v2.csv in the project folder, next to data/.
"""
import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

DATA=Path("data"); PAIRS=Path("contrastive_pairs_v2.csv")
TEXT,LAB,SID="input","Label","sample_id"; POS="propagandistic"
LABELS=["non-propagandistic","propagandistic"]
label2id={n:i for i,n in enumerate(LABELS)}; id2label={i:n for i,n in enumerate(LABELS)}
MODEL="distilbert-base-uncased"; MAXLEN=256; EPOCHS=3; TB=8; EB=16; LR=2e-5; SEED=42

def load(s):
    d=pd.read_csv(DATA/f"propxplain_english_{s}.csv"); d[LAB]=d[LAB].astype(str).str.strip()
    d=d[d[LAB].isin(LABELS)].copy(); d[TEXT]=d[TEXT].astype(str); return d[[SID,TEXT,LAB]]

def clean_train():
    pool=pd.concat([load("train"),load("validation"),load("test")],ignore_index=True)
    pool["_n"]=pool[TEXT].map(lambda s:re.sub(r"\s+"," ",s.strip().lower()))
    pool=pool.drop_duplicates("_n").reset_index(drop=True)
    pool["article"]=pool[SID].map(lambda s:s.rsplit("_",1)[0] if "_" in str(s) else str(s))
    y=(pool[LAB]==POS).astype(int).values; g=pool["article"].values
    itrf,_=next(GroupShuffleSplit(1,test_size=0.15,random_state=SEED).split(pool,y,g))
    trf=pool.iloc[itrf].reset_index(drop=True)
    itr,_=next(GroupShuffleSplit(1,test_size=0.12,random_state=SEED).split(trf,None,trf["article"].values))
    return trf.iloc[itr].reset_index(drop=True)

class WT(Trainer):
    def __init__(self,*a,cw=None,**k): super().__init__(*a,**k); self.cw=cw
    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        y=inputs.pop("labels"); o=model(**inputs); lg=o.logits
        w=self.cw.to(lg.device) if self.cw is not None else None
        loss=nn.CrossEntropyLoss(weight=w)(lg.view(-1,model.config.num_labels),y.view(-1))
        return (loss,o) if return_outputs else loss

def main():
    set_seed(SEED)
    tr=clean_train()
    counts=tr[LAB].value_counts(); n=len(tr); k=len(LABELS)
    cw=torch.tensor([n/(k*int(counts.get(x,1))) for x in LABELS],dtype=torch.float)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2,id2label=id2label,label2id=label2id)
    def ds(df,textcol):
        d=df.copy()
        if LAB in d: d["labels"]=d[LAB].map(label2id).astype(int)
        cols=[textcol]+(["labels"] if "labels" in d else [])
        return Dataset.from_pandas(d[cols].rename(columns={textcol:TEXT}),preserve_index=False).map(
            lambda b: tok(b[TEXT],truncation=True,max_length=MAXLEN),batched=True,remove_columns=[TEXT])
    args=TrainingArguments(output_dir="pairs_out",num_train_epochs=EPOCHS,per_device_train_batch_size=TB,
        per_device_eval_batch_size=EB,learning_rate=LR,logging_steps=200,save_strategy="no",report_to="none",seed=SEED)
    trainer=WT(model=model,args=args,train_dataset=ds(tr,TEXT),
               data_collator=DataCollatorWithPadding(tokenizer=tok),cw=cw)
    print("Training on the clean split..."); trainer.train()

    pairs=pd.read_csv(PAIRS)
    scored=pairs[["pair_id","member","text"]].copy()
    out=trainer.predict(ds(scored,"text")); lg=out.predictions; lg=lg[0] if isinstance(lg,tuple) else lg
    scored["score"]=torch.softmax(torch.tensor(lg),dim=-1)[:,1].numpy()
    scored[["pair_id","member","score"]].to_csv("pair_scores_v2.csv",index=False)
    print("Saved pair_scores_v2.csv. Now run: python evaluate_pairs_v2.py pair_scores_v2.csv")

if __name__=="__main__":
    main()
