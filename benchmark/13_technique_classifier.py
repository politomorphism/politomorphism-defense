# -*- coding: utf-8 -*-
"""
13_technique_classifier.py
From scoring to explanation: name WHICH manipulative device a sentence uses.

Working Papers I-III score a sentence as manipulative or plain. A defensive
audit tool needs more than a score: it needs to say which device is present, so
a human reader can see the mechanism ("this is a false dilemma", "this is a
loaded rhetorical question"). This script turns the binary detector into a
technique classifier.

Why zero-shot and not a trained model: the benchmark has only ten manipulative
examples per technique, far too few to fine-tune a reliable six-way classifier
without overfitting. Instead, a model that already knows the technique
definitions is asked to NAME the device, with the public definitions supplied in
the prompt. This is also closer to genuine explanation: the model applies a
stated definition rather than a memorised pattern.

What it does: every member of every pair (120 sentences) is shown alone and the
model must answer with exactly one label from the six techniques or "none".
Ground truth is known for all of them -- each manipulative member's technique,
and "none" for every control -- so the script computes, on its own:
  - accuracy at naming the technique on the 60 manipulative members;
  - per-technique precision / recall / F1 (7-way: six techniques + none);
  - how often the model invents a technique on a plain control (false-device
    rate), which matters as much as naming the real ones.
Every raw answer is saved to technique_predictions.csv for audit.

Backends
--------
--backend openai   (default) OpenAI API. Needs OPENAI_API_KEY and `pip install
                   openai`. Choose the model with --model (e.g. gpt-4o-mini).
--backend ollama   a local model via Ollama (free, offline, no key); set the
                   model with --model (e.g. llama3.1:8b).

Usage
-----
    python 13_technique_classifier.py
    python 13_technique_classifier.py --backend openai --model gpt-4o-mini
    python 13_technique_classifier.py --backend ollama --model llama3.1:8b

Put this file next to contrastive_pairs_v3.csv. About 120 short calls; a few
minutes on the OpenAI API, a few more with a local model.
"""
import argparse, json, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PAIRS = Path("contrastive_pairs_v3.csv")
OUT = Path("technique_predictions.csv")

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
LABELS = TECHNIQUES + ["none"]

DEFS = (
    "- presupposition: a contested claim is encoded as already given "
    "(e.g. \"the council's failed plan\" treats the failure as settled).\n"
    "- manufactured_equivalence: two unlike things are asserted to be identical "
    "(e.g. a policy equated with theft or with a dictatorship).\n"
    "- false_dilemma: a choice is collapsed to only two options when more exist "
    "(e.g. \"either we act now or we accept disaster\").\n"
    "- whataboutism: a criticism is deflected by pointing at the opponent "
    "instead of answering it.\n"
    "- loaded_rhetorical_question: a question that asserts its own answer rather "
    "than genuinely asking (e.g. \"how long will people tolerate this corrupt "
    "council?\").\n"
    "- implied_causation: causation is inferred merely from sequence or "
    "co-occurrence (e.g. \"X happened after Y, so Y caused X\").\n"
    "- none: the sentence states its content plainly, as an open, questioned, or "
    "attributed claim, with none of the devices above."
)

SYSTEM = ("You are a careful analyst of political language. You identify which "
          "single manipulative rhetorical device, if any, a sentence uses. Judge "
          "the structure, not the topic or whether you agree with it.")
PROMPT = (
    "Here are seven labels and their definitions:\n\n{defs}\n\n"
    "Read the sentence below and decide which ONE label fits it best.\n\n"
    "Sentence:\n\"{text}\"\n\n"
    "Answer with exactly one label from this list, nothing else: "
    + ", ".join(LABELS) + "."
)


def ask_openai(system, prompt, model):
    from openai import OpenAI
    client = OpenAI()
    out = client.chat.completions.create(
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
    # longest label names first, so "manufactured_equivalence" wins before "none"
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab in t:
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    print(f"Technique classifier: backend={args.backend}, model={args.model}")

    pairs = pd.read_csv(PAIRS)
    rows = []
    print(f"Naming the device for {len(pairs)} sentences...")
    for i, r in pairs.iterrows():
        true = r.technique if r.member == "manipulative" else "none"
        raw = call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=r.text))
        pred = parse_label(raw)
        rows.append({"pair_id": r.pair_id, "member": r.member, "true": true,
                     "pred": pred, "raw": (raw or "").strip().replace("\n", " ")[:120]})
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(pairs)} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    valid = df.dropna(subset=["pred"])
    miss = len(df) - len(valid)
    man = valid[valid["member"] == "manipulative"]
    con = valid[valid["member"] == "control"]

    print("\n==================  TECHNIQUE CLASSIFIER  ==================")
    print(f"Backend / model: {args.backend} / {args.model}")
    if miss:
        print(f"({miss} unparseable answers excluded)")

    acc = (man["true"] == man["pred"]).mean()
    print(f"\n(1) Naming the device on the 60 manipulative members:")
    print(f"    exact-technique accuracy: {acc:.3f}  ({int((man['true']==man['pred']).sum())}/{len(man)})")
    print("\n    Per-technique recall (did it name the right device?):")
    for tech in TECHNIQUES:
        g = man[man["true"] == tech]
        if len(g):
            print(f"      {tech:28s} {(g['true']==g['pred']).mean():.2f}  ({int((g['true']==g['pred']).sum())}/{len(g)})")

    none_rate = (con["pred"] == "none").mean()
    print(f"\n(2) Plain controls (should be 'none'):")
    print(f"    labelled 'none' correctly: {none_rate:.3f}  ({int((con['pred']=='none').sum())}/{len(con)})")
    print(f"    invented a device on a plain control (false-device rate): {1-none_rate:.3f}")
    if (con["pred"] != "none").any():
        print("    devices wrongly assigned to controls:")
        print(con[con["pred"] != "none"]["pred"].value_counts().to_string().replace("\n", "\n      "))

    print("\n(3) Full report (7-way, six techniques + none):")
    print(classification_report(valid["true"], valid["pred"], labels=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted); order:")
    print("  " + ", ".join(LABELS))
    print(confusion_matrix(valid["true"], valid["pred"], labels=LABELS))
    print(f"\nSaved {OUT.name} (every sentence, true label, predicted label, raw answer).")
    print("\nReading: accuracy in (1) is how often the tool names the right device —")
    print("the 'explanation' quality. The false-device rate in (2) is how often it")
    print("cries manipulation on a plain sentence, which a defensive tool must keep")
    print("low. The confusion matrix shows which devices it mixes up.")


if __name__ == "__main__":
    main()
