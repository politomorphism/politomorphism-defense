# -*- coding: utf-8 -*-
"""
17_negation_check.py
A targeted negation-scope check on top of the v2 classifier.

Error analysis of v2 showed that its remaining false alarms on manufactured
equivalence are all NEGATED comparisons ("... UNLIKE the way dictatorships ...",
"... BUT NOT the way a thief ..."): the model names the device although the
comparison is explicitly denied. A general instruction did not fix this. This
script tests a targeted fix instead of another global prompt change.

Two passes:
  1. Classify each sentence with the v2 prompt (same as 16).
  2. ONLY for sentences the classifier labelled 'manufactured_equivalence', ask a
     single focused yes/no question: is the comparison inside the scope of a
     negation / rejection ("unlike", "not like", "but not", "differently from")?
     If yes, the author does not perform the equivalence -> override to 'none'.

Because the second pass touches only comparison-flagged items, it cannot change
any other technique. The script reports manufactured_equivalence before and
after the override, so we can see whether the 6 negated-control false alarms are
removed WITHOUT breaking the 10 genuine manufactured_equivalence positives.

Run on contrastive_pairs_v3.csv (the regression set) and on hard_controls.csv
(H01, H02 are negated comparisons). Needs OPENAI_API_KEY (or --backend ollama).

    python 17_negation_check.py --file contrastive_pairs_v3.csv
    python 17_negation_check.py --file hard_controls.csv
"""
import argparse, json, time, urllib.request
from pathlib import Path
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]
COMPARISON = "manufactured_equivalence"

DEFS = (
    "- presupposition: a contested claim encoded as already given.\n"
    "- manufactured_equivalence: two unlike things asserted to be identical.\n"
    "- false_dilemma: a choice collapsed to only two options when more exist.\n"
    "- whataboutism: a criticism deflected by pointing at the opponent.\n"
    "- loaded_rhetorical_question: a question that asserts its own answer.\n"
    "- implied_causation: causation inferred merely from sequence or co-occurrence.\n"
    "- none: the sentence states its content plainly.\n"
    "- insufficient_evidence: you cannot tell without more context."
)
SYSTEM = ("You are a careful analyst of political language. You detect a rhetorical "
          "device only when the author actually performs and endorses it; you never "
          "infer intent or treat charged words as proof.")
PROMPT = (
    "Labels and definitions:\n\n{defs}\n\n"
    "Identify only the device the author performs and endorses. A device that is "
    "negated, quoted and disputed, hedged, or asked as an open question is NOT "
    "performed. Answer with exactly one label from: " + ", ".join(OUTPUTS) + ".\n\n"
    "Sentence:\n\"{text}\""
)
NEG_SYSTEM = ("You check one narrow thing about a sentence: whether a comparison it "
              "contains is affirmed or denied. Answer only yes or no.")
NEG_PROMPT = (
    "In the sentence below, is the comparison between the two things DENIED, "
    "NEGATED, or REJECTED -- for example by 'unlike', 'not like', 'but not', "
    "'differently from', 'is nothing like'? Answer 'yes' if the comparison is "
    "denied (the two things are said to be DIFFERENT), or 'no' if the comparison "
    "is affirmed (the two things are said to be the SAME or equivalent).\n\n"
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


def is_denied(raw):
    return (raw or "").strip().lower().startswith("y") or "yes" in (raw or "").lower()[:6]


def gold(row):
    if "gold_structure" in row and pd.notna(row.get("gold_structure")):
        return row["gold_structure"]
    return row["technique"] if row.get("member") == "manipulative" else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="contrastive_pairs_v3.csv")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    data = pd.read_csv(args.file)
    out_path = Path(f"{Path(args.file).stem}_negcheck.csv")
    print(f"Pass 1 + targeted negation check on {args.file}: {len(data)} sentences, {args.model}")

    rows = []
    for i, r in data.iterrows():
        gs = gold(r)
        p1 = parse_label(call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=r.text)))
        final = p1
        neg = ""
        if p1 == COMPARISON:                                   # second pass only here
            neg = call_with_retry(asker, NEG_SYSTEM, NEG_PROMPT.format(text=r.text)).strip()
            if is_denied(neg):
                final = "none"
        rows.append({"id": r.get("item_id", r.get("pair_id")), "gold_structure": gs,
                     "pred_v2": p1, "negation_denied": neg[:12], "pred_final": final})
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(data)} done")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    me = df[df.gold_structure == COMPARISON]           # genuine equivalence positives
    ctrl = df[df.gold_structure == "none"]             # should never be a device

    def fa(col):  # false alarms on controls for the comparison label
        return int((ctrl[col] == COMPARISON).sum())

    def rec(col):  # positives still labelled equivalence
        return int((me[col] == COMPARISON).sum())

    print("\n==================  TARGETED NEGATION CHECK  ==================")
    print(f"File: {args.file} | model: {args.model}")
    print(f"\nmanufactured_equivalence false alarms on controls:")
    print(f"   before negation check (v2):  {fa('pred_v2')}/{len(ctrl)}")
    print(f"   after  negation check:       {fa('pred_final')}/{len(ctrl)}")
    print(f"\ngenuine manufactured_equivalence positives still caught:")
    print(f"   before: {rec('pred_v2')}/{len(me)}")
    print(f"   after:  {rec('pred_final')}/{len(me)}   <- must NOT drop")
    overridden = df[(df.pred_v2 == COMPARISON) & (df.pred_final == "none")]
    print(f"\noverridden to 'none' by the negation check: {len(overridden)}")
    print(f"Saved {out_path.name}.")
    print("\nSuccess looks like: control false alarms for equivalence fall toward 0")
    print("while genuine positives stay caught. If positives drop, the check is")
    print("over-firing and needs a tighter negation question.")


if __name__ == "__main__":
    main()
