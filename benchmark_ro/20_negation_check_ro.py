# -*- coding: utf-8 -*-
"""
20_negation_check_ro.py
Romanian negation verifier applied to SAVED predictions (attributable).

Mirrors 18_apply_negation_saved.py but the negation question is in Romanian.
It touches ONLY rows the base classifier labelled manufactured_equivalence, so
by construction it cannot change any other class. For each such row it asks, in
Romanian, whether the comparison is negated; if yes, it overrides to 'none'.

Reports, honestly and without assuming the result: the equivalence false alarms
on controls before/after, individual positive losses, and that nothing outside
manufactured_equivalence changed. Run it on the saved predictions of any RO set
(the parallel set, or a fresh negation test set) with that set's text file.

    python 20_negation_check_ro.py --pred <saved_v2ro_predictions.csv> --data <ro_set.csv>

Needs OPENAI_API_KEY (or ollama).
"""
import argparse, json, time, urllib.request
import pandas as pd
ME="manufactured_equivalence"
NEG_SYSTEM=("Verifici un singur lucru ingust: daca o comparatie dintr-o propozitie "
            "este afirmata sau negata. Raspunde doar da sau nu.")
NEG_PROMPT=("In propozitia de mai jos, comparatia dintre cele doua lucruri este "
            "NEGATA sau RESPINSA (de ex. 'nu ca', 'nu asa cum', 'spre deosebire de', "
            "'nu seamana', 'dar nu')? Raspunde 'da' daca cele doua lucruri sunt spuse "
            "a fi DIFERITE, sau 'nu' daca sunt spuse a fi la FEL / echivalente.\n\n"
            "Propozitie:\n\"{text}\"\n\nRaspunde da sau nu.")

def ask_openai(system,prompt,model):
    from openai import OpenAI
    return OpenAI().chat.completions.create(model=model,temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}]).choices[0].message.content
def ask_ollama(system,prompt,model,host="http://localhost:11434"):
    body=json.dumps({"model":model,"system":system,"prompt":prompt,"stream":False,"options":{"temperature":0}}).encode()
    req=urllib.request.Request(host+"/api/generate",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read().decode()).get("response","")
def asker(b,m):
    if b=="openai": return lambda s,p: ask_openai(s,p,m)
    if b=="ollama": return lambda s,p: ask_ollama(s,p,m)
    raise SystemExit("unknown backend")
def call(a,s,p,tries=3):
    for i in range(tries):
        try: return a(s,p)
        except Exception:
            if i==tries-1: raise
            time.sleep(2*(i+1))
def denied(raw):
    t=(raw or "").strip().lower()
    return t.startswith("da") or t.startswith("y") or t[:5].find("yes")>=0
def goldstruct(row):
    if "gold_structure" in row and pd.notna(row.get("gold_structure")): return row["gold_structure"]
    return row["technique"] if row.get("member")=="manipulative" else "none"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred",required=True); ap.add_argument("--data",required=True)
    ap.add_argument("--backend",default="openai",choices=["openai","ollama"]); ap.add_argument("--model",default="gpt-4o-mini")
    a=ap.parse_args(); ask=asker(a.backend,a.model)
    pred=pd.read_csv(a.pred).reset_index(drop=True); data=pd.read_csv(a.data).reset_index(drop=True)
    if len(pred)!=len(data): raise SystemExit(f"row mismatch {len(pred)} vs {len(data)}")
    pred["text"]=data["text"].values
    pred["gold_structure"]=[goldstruct(r) for _,r in data.iterrows()]
    pred["pred_final"]=pred["pred"]; pred["neg"]=""
    for i,r in pred.iterrows():
        if r["pred"]==ME:
            ans=call(ask,NEG_SYSTEM,NEG_PROMPT.format(text=r["text"])).strip()
            pred.at[i,"neg"]=ans[:10]
            if denied(ans): pred.at[i,"pred_final"]="none"
    out=a.pred.replace(".csv","_ro_negfinal.csv"); pred.to_csv(out,index=False)
    ctrl=pred[pred.gold_structure=="none"]; me=pred[pred.gold_structure==ME]
    changed_outside=int(((pred.pred!=pred.pred_final)&(pred.pred!=ME)).sum())
    print("\n==========  VERIFICATOR NEGATIE (RO), pe predictii salvate  ==========")
    print(f"pred: {a.pred}")
    print(f"Alarme false echivalenta pe controale: {int((ctrl.pred==ME).sum())} -> {int((ctrl.pred_final==ME).sum())}")
    before=int((me.pred==ME).sum()); lost=int(((me.pred==ME)&(me.pred_final!=ME)).sum())
    print(f"Echivalente pozitive corecte: {before} -> {int((me.pred_final==ME).sum())}  (pierderi individuale: {lost})")
    print(f"Predictii schimbate in afara echivalentei: {changed_outside}  (trebuie 0)")
    print(f"Salvat {out}.")

if __name__=="__main__": main()
