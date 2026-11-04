# -*- coding: utf-8 -*-
"""
16_classifier_v2_abstention.py
Technique classifier, version 2: negation-aware, with abstention.

The version-1 classifier (13_technique_classifier.py) named the device well when
one was present (0.917) but invented a device on plain controls 65% of the time,
because it keyed on charged vocabulary. This version tests a fix suggested by
review: tell the model that charged words are not enough, make it check whether
the relevant claim is actually asserted or instead negated, quoted, attributed,
hedged, or questioned, and let it answer "none" or "insufficient_evidence"
instead of being forced to pick a device.

Run it on two files to see whether the fix helps:
  --file contrastive_pairs_v3.csv   the original 60 pairs (compare to v1's 65%)
  --file hard_controls.csv          hard controls (negated/quoted/open/cautious/
                                    verified) plus inverse cases (a real device
                                    with mild, non-charged words) -- tests the
                                    MECHANISM, not the vocabulary.

Both files share the schema pair_id,technique,member,text,rationale. Ground
truth: a manipulative member's technique, and "none" for every control.

Backends: --backend openai (default, needs OPENAI_API_KEY) or --backend ollama.
Usage:
    python 16_classifier_v2_abstention.py --file hard_controls.csv
    python 16_classifier_v2_abstention.py --file contrastive_pairs_v3.csv
"""
import argparse, json, time, urllib.request
from pathlib import Path
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]

DEFS = (
    "- presupposition: a contested claim encoded as already given.\n"
    "- manufactured_equivalence: two unlike things asserted to be identical.\n"
    "- false_dilemma: a choice collapsed to only two options when more exist.\n"
    "- whataboutism: a criticism deflected by pointing at the opponent.\n"
    "- loaded_rhetorical_question: a question that asserts its own answer.\n"
    "- implied_causation: causation inferred merely from sequence or co-occurrence.\n"
    "- none: the sentence states its content plainly, with none of the devices.\n"
    "- insufficient_evidence: you cannot tell without more context."
)

SYSTEM = ("You are a careful analyst of political language. You detect a "
          "manipulative rhetorical device only when the text actually performs "
          "it. You never infer intent, and you never treat charged vocabulary as "
          "proof on its own.")
PROMPT = (
    "Labels and definitions:\n\n{defs}\n\n"
    "Rules. Emotionally charged words are NOT sufficient. Before naming a device, "
    "check whether the relevant claim is actually ASSERTED by the author, or "
    "instead negated (\"not like a dictatorship\"), quoted or attributed to others "
    "and possibly disputed (\"was called a 'thief', but...\"), hedged or presented "
    "as uncertain, or discussed as an open question. Those are NOT the device. "
    "Name a device only if you can identify BOTH the specific fragment AND the "
    "problematic inference the author is actually making. If the sentence states "
    "its content plainly, answer 'none'. If you cannot decide without more "
    "context, answer 'insufficient_evidence'.\n\n"
    "Sentence:\n\"{text}\"\n\n"
    "Answer with exactly one label from: " + ", ".join(OUTPUTS) + "."
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
    raise SystemExit(f"unknown backend: {backend}")


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
    ap.add_argument("--file", default="hard_controls.csv")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    data = pd.read_csv(args.file)
    stem = Path(args.file).stem
    out_path = Path(f"{stem}_v2_predictions.csv")
    print(f"Classifier v2 (abstention) on {args.file}: {len(data)} sentences, model {args.model}")

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

    valid = df.dropna(subset=["pred"])
    man = valid[valid.member == "manipulative"]
    con = valid[valid.member == "control"]

    print("\n==================  CLASSIFIER v2 (abstention)  ==================")
    print(f"File: {args.file} | model: {args.model}")

    if len(man):
        named = (man.true == man.pred).mean()
        abst = man.pred.isin(["none", "insufficient_evidence"]).mean()
        print(f"\nManipulative members ({len(man)}): correct device named {named:.3f}, "
              f"abstained/none {abst:.3f}")
        for t in TECHNIQUES:
            g = man[man.true == t]
            if len(g):
                print(f"   {t:28s} named {(g.true==g.pred).mean():.2f}  ({int((g.true==g.pred).sum())}/{len(g)})")

    if len(con):
        safe = con.pred.isin(["none", "insufficient_evidence"]).mean()
        false_dev = (~con.pred.isin(["none", "insufficient_evidence"])).mean()
        print(f"\nControls ({len(con)}): did NOT invent a device {safe:.3f} "
              f"(none/insufficient) | false-device rate {false_dev:.3f}")
        bad = con[~con.pred.isin(["none", "insufficient_evidence"])]
        if len(bad):
            print("   devices wrongly assigned to controls:", bad.pred.value_counts().to_dict())

    print(f"\nabstention rate overall: {valid.pred.eq('insufficient_evidence').mean():.3f}")
    print(f"Saved {out_path.name}.")
    print("\nCompare the control false-device rate here with v1's 0.65. If it falls")
    print("sharply, abstention + negation-awareness fixes much of the over-flagging")
    print("in a single stage; if not, the gate (detect-then-explain) is needed.")


if __name__ == "__main__":
    main()
