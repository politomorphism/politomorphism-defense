# -*- coding: utf-8 -*-
"""
19_study2_diagnostic.py
Study 2: a targeted CONTRASTIVE DIAGNOSTIC of structural sensitivity.

WHAT THIS IS, STATED HONESTLY. This is NOT proof that the detector "understands"
or reasons over argumentative structure. It is a targeted diagnostic on 30
author-built, author-labelled examples, testing whether predictions respond to a
change in argumentative structure (affirm vs negate, assert vs quote, justified
vs unjustified comparison, presupposition kept vs cancelled) rather than merely
to the presence of a negation cue. Success here is PRELIMINARY evidence of
structural sensitivity on these cases; it does not establish a general mechanism,
exclude lexical strategies, or demonstrate generalisation. Because the set targets
previously observed weaknesses, it is a diagnostic set: any later tuning requires
a fresh, untouched test set.

FROZEN CONFIGURATION. The classifier (stage 1) and negation verifier (stage 2)
prompts below are copied verbatim from 16_classifier_v2_abstention.py and
18_apply_negation_saved.py. Do not edit them here; edit the originals and re-copy,
then re-freeze (see 19_freeze_manifest.json / analysis of hashes).

TWO STAGES LOGGED SEPARATELY (reviewer rule B). For every item we save: the
classifier's original prediction, whether the verifier activated, the verifier's
decision, and the final prediction. This separates classifier failures from
verifier-induced failures.

WHAT IT REPORTS.
  (C) Critical-transition table: for each case type, the desired verifier
      behaviour and the observed counts, with denominators -- especially the
      number of originally correct positives the verifier suppresses.
  (D) Contrast scoring on the paired items:
        pair accuracy          : both members of a pair correct (performed/absent);
        required-change accuracy: the prediction flips in the right direction when
                                  the structure changes between the two members.
      Invariance accuracy is reported as NOT APPLICABLE: this set contains no
      invariance pairs (same structure under an irrelevant surface change). That
      is a limitation of the set, stated, not hidden.
  Item level: device-performed detection, false alarms on 'none', abstentions
      (kept in the denominator, never counted as a correct rejection).

Abstentions are never counted as "absence".

Run (needs OPENAI_API_KEY, or --backend ollama --model <name>):
    python 19_study2_diagnostic.py --data study2_mechanism_families.csv
Outputs next to the data file:
    study2_predictions.csv   per-item, both stages
    study2_report.txt        the tables below, as text
"""
import argparse, json, time, hashlib, datetime, urllib.request, os
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]
ABSTAIN = "insufficient_evidence"
ME = "manufactured_equivalence"

# ---- frozen stage-1 prompt (verbatim from script 16) ----
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
    "Target. Identify only the device the AUTHOR performs and endorses in this "
    "sentence, not every structure that merely appears in it. Emotionally charged "
    "words are NOT sufficient. A device that is negated (\"not like a "
    "dictatorship\"), quoted or attributed to others and disputed, hedged or "
    "presented as uncertain, or asked as a genuinely open question is NOT "
    "performed by the author. Judge the structure the author commits to; do not "
    "judge whether its content is true, and do not infer intent. Name a device "
    "only if you can identify BOTH the fragment AND the inference the author "
    "makes. If the author states the content plainly, answer 'none'. If you "
    "cannot decide without more context, answer 'insufficient_evidence'.\n\n"
    "Sentence:\n\"{text}\"\n\n"
    "Answer with exactly one label from: " + ", ".join(OUTPUTS) + "."
)
# ---- frozen stage-2 prompt (verbatim from script 18) ----
NEG_SYSTEM = ("You check one narrow thing: whether a comparison in a sentence is "
              "affirmed or denied. Answer only yes or no.")
NEG_PROMPT = (
    "In the sentence below, is the comparison between the two things DENIED, "
    "NEGATED, or REJECTED (e.g. 'unlike', 'not like', 'but not', 'differently "
    "from', 'is nothing like')? Answer 'yes' if the two things are said to be "
    "DIFFERENT, or 'no' if they are said to be the SAME or equivalent.\n\n"
    "Sentence:\n\"{text}\"\n\nAnswer yes or no."
)

# explicit contrast pairs (required-change: structure differs between members)
PAIRS = [("A1a", "A1b"), ("A2a", "A2b"), ("A3a", "A3b"),
         ("B1a", "B1b"), ("B2a", "B2b"), ("B3a", "B3b"),
         ("C1a", "C1b"), ("C2a", "C2b"), ("C3a", "C3b"),
         ("D1a", "D1b"), ("D2a", "D2b"), ("D3a", "D3b"),
         ("E2", "E1"), ("E3", "E4"), ("E5", "E6")]  # (performed, absent) by topic


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
    for lab in OUTPUTS:
        if lab in t:
            return lab
    return "none"


def denied(raw):
    t = (raw or "").strip().lower()
    return t.startswith("yes")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="study2_mechanism_families.csv")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.data)) or "."
    asker = make_asker(a.backend, a.model)
    df = pd.read_csv(a.data)

    # provenance freeze
    manifest = {
        "frozen_on": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_file": os.path.basename(a.data),
        "data_sha256_16": sha256(a.data),
        "this_script_sha256_16": sha256(os.path.abspath(__file__)),
        "model": a.model, "backend": a.backend, "temperature": 0,
        "stage1": "16_classifier_v2_abstention (verbatim prompt)",
        "stage2": "18_apply_negation_saved (verbatim prompt)",
    }
    json.dump(manifest, open(os.path.join(here, "study2_freeze_manifest.json"), "w"), indent=2)

    rows = []
    for _, r in df.iterrows():
        s1 = parse_label(call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=r["text"])))
        activated = (s1 == ME)
        decision, final = "na", s1
        if activated:
            raw = call_with_retry(asker, NEG_SYSTEM, NEG_PROMPT.format(text=r["text"]))
            decision = "denied" if denied(raw) else "affirmed"
            final = "none" if decision == "denied" else s1
        rows.append({"item_id": r["item_id"], "family": r["family"], "property": r["property"],
                     "gold_structure": r["gold_structure"], "gold_use": r["gold_use"],
                     "stage1_pred": s1, "verifier_activated": activated,
                     "verifier_decision": decision, "final_pred": final, "text": r["text"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(here, "study2_predictions.csv"), index=False)

    def performed(gold): return gold != "none"
    def flagged(pred): return pred in TECHNIQUES  # a device named (abstain/none are not flags)

    lines = []
    def emit(s=""): lines.append(s); print(s)

    emit("STUDY 2 - targeted contrastive diagnostic of structural sensitivity")
    emit(f"model={a.model}  data={manifest['data_file']}  sha={manifest['data_sha256_16']}  frozen={manifest['frozen_on']}")
    emit("Preliminary evidence of sensitivity on these 30 author-labelled items;")
    emit("not proof of a general structural-reasoning mechanism.\n")

    # item level
    pos = out[out.gold_structure != "none"]; neg = out[out.gold_structure == "none"]
    emit("ITEM LEVEL (final prediction)")
    emit(f"  device-performed items: {len(pos)}")
    emit(f"    detected (a device named)   : {sum(flagged(p) for p in pos.final_pred)}")
    emit(f"    exact structure correct     : {sum(p==g for p,g in zip(pos.final_pred,pos.gold_structure))}")
    emit(f"    abstained                   : {sum(p==ABSTAIN for p in pos.final_pred)}")
    emit(f"  no-device (control) items: {len(neg)}")
    emit(f"    correct 'none'              : {sum(p=='none' for p in neg.final_pred)}")
    emit(f"    false alarm (device named)  : {sum(flagged(p) for p in neg.final_pred)}")
    emit(f"    abstained                   : {sum(p==ABSTAIN for p in neg.final_pred)}\n")

    # (C) critical-transition table (verifier stage)
    emit("(C) CRITICAL-TRANSITION TABLE (negation verifier)")
    act = out[out.verifier_activated]
    neg_cancels = act[(act.gold_structure == "none")]           # should remove flag
    neg_intact = act[(act.gold_structure == ME)]                # should keep flag
    removed_ok = sum((neg_cancels.final_pred == "none"))
    kept_wrong = sum((neg_cancels.final_pred == ME))
    kept_ok = sum((neg_intact.final_pred == ME))
    suppressed = sum((neg_intact.final_pred == "none"))
    emit(f"  verifier activated on {len(act)} item(s) (stage-1 = manufactured_equivalence)")
    emit(f"  case: negation cancels equivalence (gold none)  n={len(neg_cancels)}  correctly removed={removed_ok}  wrongly kept={kept_wrong}")
    emit(f"  case: equivalence intact (gold equivalence)      n={len(neg_intact)}  correctly kept={kept_ok}  WRONGLY SUPPRESSED={suppressed}")
    emit(f"  >> originally-correct positives suppressed by verifier: {suppressed}  (this is the collateral-damage count)\n")

    # (D) contrast scoring
    byid = {r.item_id: r for r in out.itertuples()}
    pair_ok = req_ok = n_pairs = 0
    for pf, pa in PAIRS:
        if pf not in byid or pa not in byid: continue
        n_pairs += 1
        f_perf = flagged(byid[pf].final_pred)   # performed member should flag
        a_perf = flagged(byid[pa].final_pred)   # absent member should NOT flag
        if f_perf and not a_perf: pair_ok += 1; req_ok += 1
    emit("(D) CONTRAST SCORING (performed/absent axis)")
    emit(f"  pairs scored: {n_pairs}")
    emit(f"  pair accuracy (both members correct)     : {pair_ok}/{n_pairs}")
    emit(f"  required-change accuracy (flips correctly): {req_ok}/{n_pairs}")
    emit("  invariance accuracy: NOT APPLICABLE - this set has no invariance pairs")
    emit("    (same structure under an irrelevant surface change). Adding such")
    emit("    pairs is required to measure invariance.\n")

    emit("LIMITATION. Author-built, author-labelled diagnostic set. Running the")
    emit("software does not establish independent label validity; 30 related")
    emit("examples are not 30 independent tests of broad generalisation.")

    open(os.path.join(here, "study2_report.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
