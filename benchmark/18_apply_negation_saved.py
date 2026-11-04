# -*- coding: utf-8 -*-
"""
18_apply_negation_saved.py
Apply the targeted negation check to SAVED predictions (no re-classification).

The attributable way to measure the negation verifier: hold the base
classifier's predictions fixed and apply the check only to the items it already
labelled 'manufactured_equivalence'. This script does exactly that. It reads a
saved predictions file (from 16_classifier_v2_abstention.py), asks the focused
negation question ONLY on manufactured_equivalence rows, and overrides those to
'none' when the comparison is denied. By construction it can change NOTHING
outside the manufactured_equivalence class -- the implementation guarantees the
acceptance condition "no prediction other than manufactured_equivalence may be
modified".

Input: a *_v2_predictions.csv with columns including (id|pair_id), gold_structure
(or the pairs schema), pred, and the original text source so the sentence can be
fetched. Because the saved predictions file does not carry the text, pass the
dataset file too with --data so the text is looked up by row order.

    python 18_apply_negation_saved.py --pred contrastive_pairs_v4_heldout_v2_predictions.csv --data contrastive_pairs_v4_heldout.csv

Needs OPENAI_API_KEY (or --backend ollama). It calls the model only for the
manufactured_equivalence rows, so it is cheap.

Reporting (attributable): on controls (gold_structure=='none'), equivalence
false alarms before vs after; on equivalence positives (gold_structure==
'manufactured_equivalence'), correct kept before vs after, counting individual
losses; and a check that zero predictions outside manufactured_equivalence
changed. For the negated-comparison subset it also prints the relative reduction
and warns if the base has too few such errors to evaluate.
"""
import argparse, json, time, urllib.request
import pandas as pd

ME = "manufactured_equivalence"
NEG_SYSTEM = ("You check one narrow thing: whether a comparison in a sentence is "
              "affirmed or denied. Answer only yes or no.")
NEG_PROMPT = (
    "In the sentence below, is the comparison between the two things DENIED, "
    "NEGATED, or REJECTED (e.g. 'unlike', 'not like', 'but not', 'differently "
    "from', 'is nothing like')? Answer 'yes' if the two things are said to be "
    "DIFFERENT, or 'no' if they are said to be the SAME or equivalent.\n\n"
    "Sentence:\n\"{text}\"\n\nAnswer yes or no."
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


def asker_for(backend, model):
    if backend == "openai":
        return lambda s, p: ask_openai(s, p, model)
    if backend == "ollama":
        return lambda s, p: ask_ollama(s, p, model)
    raise SystemExit("unknown backend")


def denied(raw):
    t = (raw or "").strip().lower()
    return t.startswith("y") or t[:6].find("yes") >= 0


def gold_structure(row):
    if "gold_structure" in row and pd.notna(row.get("gold_structure")):
        return row["gold_structure"]
    return row["technique"] if row.get("member") == "manipulative" else "none"


def is_negated_comparison(text):
    t = str(text).lower()
    cues = ["unlike", "not like", "but not", "differently from", "nothing like",
            "is not the same", "not the same", "though not", "rather than"]
    return any(c in t for c in cues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="saved *_v2_predictions.csv")
    ap.add_argument("--data", required=True, help="dataset file with the texts")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    ask = asker_for(args.backend, args.model)

    pred = pd.read_csv(args.pred).reset_index(drop=True)
    data = pd.read_csv(args.data).reset_index(drop=True)
    if len(pred) != len(data):
        raise SystemExit(f"row mismatch: {len(pred)} predictions vs {len(data)} data rows")
    pred["text"] = data["text"].values
    pred["gold_structure"] = [gold_structure(r) for _, r in data.iterrows()]

    pred["pred_final"] = pred["pred"]
    pred["neg_answer"] = ""
    for i, r in pred.iterrows():
        if r["pred"] == ME:                                   # ONLY equivalence rows
            a = ask(NEG_SYSTEM, NEG_PROMPT.format(text=r["text"])).strip()
            pred.at[i, "neg_answer"] = a[:12]
            if denied(a):
                pred.at[i, "pred_final"] = "none"
    out = args.pred.replace(".csv", "_negfinal.csv")
    pred.to_csv(out, index=False)

    ctrl = pred[pred.gold_structure == "none"]
    me = pred[pred.gold_structure == ME]
    # implementation guarantee check
    changed_outside = int(((pred.pred != pred.pred_final) & (pred.pred != ME)).sum())

    print("\n==========  NEGATION CHECK ON SAVED PREDICTIONS  ==========")
    print(f"pred file: {args.pred}")
    print(f"\nEquivalence false alarms on controls: "
          f"{int((ctrl.pred==ME).sum())} -> {int((ctrl.pred_final==ME).sum())}")
    before_pos = int((me.pred == ME).sum()); after_pos = int((me.pred_final == ME).sum())
    lost = int(((me.pred == ME) & (me.pred_final != ME)).sum())
    print(f"Equivalence positives kept correct: {before_pos} -> {after_pos}  "
          f"(individual losses: {lost})")
    print(f"Predictions changed OUTSIDE manufactured_equivalence: {changed_outside}  "
          f"(must be 0)")

    # negated-comparison subset among controls
    negctrl = ctrl[ctrl.text.map(is_negated_comparison)]
    fp_before = int((negctrl.pred == ME).sum())
    fp_after = int((negctrl.pred_final == ME).sum())
    print(f"\nNegated-comparison controls: {len(negctrl)} items")
    print(f"  equivalence false alarms on them: {fp_before} -> {fp_after}")
    if fp_before == 0:
        print("  relative reduction: NOT EVALUABLE (base has no such false alarms).")
    else:
        print(f"  relative reduction: {(fp_before-fp_after)/fp_before:.0%}  "
              f"(raw {fp_before}->{fp_after}); treat % as fragile if fp_before is small.")
    print(f"\nSaved {out}.")


if __name__ == "__main__":
    main()
