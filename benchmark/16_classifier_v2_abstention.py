# -*- coding: utf-8 -*-
"""
16_classifier_v2_abstention.py
Technique classifier, version 2: negation-aware, with abstention.

The version-1 classifier (13) named the device well when one was present (0.917)
but invented a device on plain controls 65% of the time, keying on charged
vocabulary. This version tests a single-stage fix: tell the model that charged
words are not proof, make it check whether the relevant claim is actually
asserted or instead negated, quoted, hedged, or questioned, and let it answer
"none" or "insufficient_evidence" instead of being forced to pick a device.

WHAT THIS DOES AND DOES NOT TEST. It tests classifier v2 (instructions +
abstention), NOT the two-stage detect-then-explain gate. If false alarms fall
here, that is evidence the instructions help, not yet evidence for the gate.

Three outcomes are counted SEPARATELY, never merged: a correct answer, a false
alarm (a device named where none is used), and an abstention
("insufficient_evidence"). Abstention is not a correct classification.

Ground truth has two axes, because a rhetorical structure is not the same as a
manipulative use:
  gold_structure : which device is structurally present (a technique) or "none".
  gold_use       : "manipulative", "legitimate" (device present but not
                   deceptive, e.g. a confirmed presupposition), or "absent".
This lets us keep "no device" distinct from "a device used legitimately".

Run it on two files:
  --file contrastive_pairs_v3.csv   the 60 pairs. Reports, from one run, BOTH
                                    the control false-alarm rate (compare to
                                    v1's 0.65) AND the recall on the 60 positives
                                    -- so a drop in false alarms that is really
                                    just blanket refusal shows up as collapsed
                                    recall.
  --file hard_controls.csv          hard controls (negated/quoted/open/cautious/
                                    genuine question), one legitimate-but-present
                                    device, and inverse cases (a real device with
                                    mild words). Tests the mechanism, not the
                                    vocabulary.

For contrastive_pairs_v3.csv the two gold axes are derived (control -> absent;
manipulative -> manipulative). hard_controls.csv carries them explicitly.

Backends: --backend openai (default, needs OPENAI_API_KEY) or --backend ollama.
"""
import argparse, json, time, urllib.request
from pathlib import Path
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]
ABSTAIN = "insufficient_evidence"

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
          "rhetorical device only when the text actually performs it. You never "
          "infer intent, and never treat charged vocabulary as proof on its own.")
PROMPT = (
    "Labels and definitions:\n\n{defs}\n\n"
    "Rules. Emotionally charged words are NOT sufficient. Before naming a device, "
    "check whether the relevant claim is actually ASSERTED by the author, or "
    "instead negated (\"not like a dictatorship\"), quoted or attributed to others "
    "and possibly disputed, hedged or presented as uncertain, or asked as an open "
    "question. Those are NOT the device. Name a device only if you can identify "
    "BOTH the specific fragment AND the inference the author is actually making. "
    "If the sentence states its content plainly, answer 'none'. If you cannot "
    "decide without more context, answer 'insufficient_evidence'.\n\n"
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


def gold(row):
    """Return (gold_structure, gold_use) whether columns are explicit or derived."""
    if "gold_use" in row and pd.notna(row.get("gold_use")):
        return row["gold_structure"], row["gold_use"]
    # derive from the pairs schema
    if row.get("member") == "manipulative":
        return row["technique"], "manipulative"
    return "none", "absent"


def rate(mask_num, n):
    return f"{(mask_num/n):.3f} ({mask_num}/{n})" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="hard_controls.csv")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    data = pd.read_csv(args.file)
    text_col = "text"
    out_path = Path(f"{Path(args.file).stem}_v2_predictions.csv")
    print(f"Classifier v2 (abstention) on {args.file}: {len(data)} sentences, model {args.model}")

    rows = []
    for i, r in data.iterrows():
        gs, gu = gold(r)
        raw = call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=r[text_col]))
        pred = parse_label(raw)
        rows.append({"id": r.get("item_id", r.get("pair_id")), "gold_structure": gs,
                     "gold_use": gu, "pred": pred,
                     "raw": (raw or "").strip().replace("\n", " ")[:120]})
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(data)} done")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    v = df.dropna(subset=["pred"])

    def is_device(p): return p in TECHNIQUES

    print("\n==================  CLASSIFIER v2 (abstention)  ==================")
    print(f"File: {args.file} | model: {args.model}")
    print("Three outcomes counted separately: correct / false-alarm / abstention.\n")

    ab = v[v.gold_use == "absent"]
    if len(ab):
        n = len(ab)
        correct = int((ab.pred == "none").sum())
        false_alarm = int(ab.pred.map(is_device).sum())
        abstain = int((ab.pred == ABSTAIN).sum())
        print(f"[DEVICE ABSENT] n={n}  (plain text; correct answer is 'none')")
        print(f"   correct 'none':   {rate(correct,n)}")
        print(f"   FALSE ALARM:      {rate(false_alarm,n)}   <- compare to v1's 0.65")
        print(f"   abstention:       {rate(abstain,n)}")
        if false_alarm:
            print("   devices wrongly named:", ab[ab.pred.map(is_device)].pred.value_counts().to_dict())

    lg = v[v.gold_use == "legitimate"]
    if len(lg):
        n = len(lg)
        named = int((lg.pred == lg.gold_structure).sum())
        said_none = int((lg.pred == "none").sum())
        abstain = int((lg.pred == ABSTAIN).sum())
        print(f"\n[DEVICE PRESENT BUT LEGITIMATE] n={n}  (structure there, use not deceptive)")
        print(f"   named the structure: {rate(named,n)}   (expected if v2 detects structure)")
        print(f"   said 'none':         {rate(said_none,n)}   (expected if v2 tracks manipulative use)")
        print(f"   abstention:          {rate(abstain,n)}")
        print("   note: this axis is reported, not scored as error either way.")

    mp = v[v.gold_use == "manipulative"]
    if len(mp):
        n = len(mp)
        named = int((mp.pred == mp.gold_structure).sum())
        wrong = int((mp.pred.map(is_device) & (mp.pred != mp.gold_structure)).sum())
        miss_none = int((mp.pred == "none").sum())
        abstain = int((mp.pred == ABSTAIN).sum())
        print(f"\n[DEVICE MANIPULATIVE] n={n}  (a real device is used; name it)")
        print(f"   correct device:   {rate(named,n)}   <- recall; watch this does NOT collapse")
        print(f"   wrong device:     {rate(wrong,n)}")
        print(f"   missed ('none'):  {rate(miss_none,n)}")
        print(f"   abstention:       {rate(abstain,n)}")
        for t in TECHNIQUES:
            g = mp[mp.gold_structure == t]
            if len(g):
                print(f"      {t:28s} {rate(int((g.pred==t).sum()), len(g))}")

    print(f"\nSaved {out_path.name}.")
    print("Read it together: a real fix lowers FALSE ALARM on absent items WITHOUT")
    print("collapsing recall on manipulative items. A drop in false alarms paid for")
    print("by lost recall is blanket refusal, not understanding.")


if __name__ == "__main__":
    main()
