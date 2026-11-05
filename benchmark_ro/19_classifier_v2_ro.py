# -*- coding: utf-8 -*-
"""
19_classifier_v2_ro.py
Romanian version of the v2 classifier (abstention, negation-aware).

Same design and output labels as the English 16_classifier_v2_abstention.py, but
the system prompt, the technique definitions and the rules are in Romanian, and
the sentences are Romanian. This tests whether the explanation-layer approach
transfers to Romanian political text. The output labels stay in English tokens
so that scoring is identical to the English pipeline.

Run on contrastive_pairs_ro_v4parallel.csv. Needs OPENAI_API_KEY (or ollama).
    python 19_classifier_v2_ro.py --file contrastive_pairs_ro_v4parallel.csv
"""
import argparse, json, time, urllib.request
from pathlib import Path
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]
ABSTAIN = "insufficient_evidence"

DEFS = (
    "- presupposition: o afirmatie contestata codificata ca deja data.\n"
    "- manufactured_equivalence: doua lucruri diferite afirmate ca identice.\n"
    "- false_dilemma: o alegere redusa la doua optiuni cand exista mai multe.\n"
    "- whataboutism: o critica deviata aratand spre adversar.\n"
    "- loaded_rhetorical_question: o intrebare care isi afirma propriul raspuns.\n"
    "- implied_causation: cauzalitate dedusa din simpla succesiune sau coincidenta.\n"
    "- none: propozitia enunta continutul simplu, fara niciunul dintre procedee.\n"
    "- insufficient_evidence: nu poti decide fara mai mult context."
)
SYSTEM = ("Esti un analist atent al limbajului politic. Detectezi un procedeu "
          "retoric doar cand autorul chiar il realizeaza si si-l asuma; nu deduci "
          "intentia si nu tratezi vocabularul incarcat drept dovada in sine.")
PROMPT = (
    "Etichete si definitii:\n\n{defs}\n\n"
    "Tinta. Identifica doar procedeul pe care AUTORUL il realizeaza si si-l asuma "
    "in aceasta propozitie, nu orice structura care apare. Cuvintele incarcate "
    "emotional NU sunt suficiente. Un procedeu negat ('nu ca o dictatura'), citat "
    "sau atribuit altora si contestat, atenuat sau prezentat ca incert, ori pus "
    "ca intrebare genuin deschisa NU este realizat de autor. Judeca structura pe "
    "care autorul si-o asuma; nu judeca daca continutul e adevarat si nu deduce "
    "intentia. Numeste un procedeu doar daca poti identifica si fragmentul, si "
    "inferenta facuta de autor. Daca autorul enunta continutul simplu, raspunde "
    "'none'. Daca nu poti decide fara mai mult context, raspunde "
    "'insufficient_evidence'.\n\n"
    "Propozitie:\n\"{text}\"\n\n"
    "Raspunde cu exact o eticheta din: " + ", ".join(OUTPUTS) + "."
)


def ask_openai(system, prompt, model):
    from openai import OpenAI
    out = OpenAI().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return out.choices[0].message.content


def ask_ollama(system, prompt, model, host="http://localhost:11434"):
    body = json.dumps({"model": model, "system": system, "prompt": prompt,
                       "stream": False, "options": {"temperature": 0}}).encode("utf-8")
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def make_asker(backend, model):
    if backend == "openai":
        return lambda s, p: ask_openai(s, p, model)
    if backend == "ollama":
        return lambda s, p: ask_ollama(s, p, model)
    raise SystemExit("unknown backend")


def call_with_retry(asker, system, prompt, tries=3):
    for i in range(tries):
        try:
            return asker(system, prompt)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def parse_label(raw):
    t = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    for lab in sorted(OUTPUTS, key=len, reverse=True):
        if lab in t:
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="contrastive_pairs_ro_v4parallel.csv")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    data = pd.read_csv(args.file)
    out_path = Path(f"{Path(args.file).stem}_v2ro_predictions.csv")
    print(f"Romanian v2 classifier on {args.file}: {len(data)} sentences, {args.model}")

    rows = []
    for i, r in data.iterrows():
        true = r.technique if r.member == "manipulative" else "none"
        raw = call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=r.text))
        pred = parse_label(raw)
        rows.append({"pair_id": r.pair_id, "member": r.member, "true": true,
                     "pred": pred, "raw": (raw or "").strip().replace("\n", " ")[:120]})
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(data)} done")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    def isdev(p): return p in TECHNIQUES
    man = df[df.member == "manipulative"]; con = df[df.member == "control"]
    print("\n==================  v2 ROMANIAN  ==================")
    fa = int(con.pred.map(isdev).sum())
    det = int(man.pred.map(isdev).sum()); ct = int((man.pred == man.true).sum())
    print(f"Controls: false alarms {fa}/60 | correct 'none' {int((con.pred=='none').sum())}/60 | abstain {int((con.pred==ABSTAIN).sum())}/60")
    print(f"Positives: detection recall {det}/60 | correct technique {ct}/60")
    print("\nPer technique (correct on positives):")
    for t in TECHNIQUES:
        g = man[man.true == t]
        print(f"   {t:28s} {int((g.pred==t).sum())}/{len(g)}")
    print(f"\nSaved {out_path.name}.")
    print("Compare with the English v4 held-out run to see cross-lingual transfer.")


if __name__ == "__main__":
    main()
