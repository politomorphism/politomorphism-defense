# -*- coding: utf-8 -*-
"""
23_logic_fd_eval.py
Evaluate the v2 detector's FALSE_DILEMMA flag against independent labels from the
LOGIC dataset (Jin et al., Findings of EMNLP 2022; github.com/causalNLP/
logical-fallacy). This is the first evaluation on labels NOT built by the author.

SCOPE, stated honestly:
 - Conceptual match is close: LOGIC false dilemma = "presenting only two options
   or sides when there are many options or sides"; project definition = "a choice
   collapsed to only two options when more exist." (See the compatibility audit.)
 - Unit: single sentence / short snippet, which matches the detector's unit.
 - NEGATIVES ARE OTHER-FALLACY SENTENCES, NOT NEUTRAL TEXT. So the false-positive
   rate here is "flags false_dilemma on a DIFFERENT fallacy", and specificity is
   discrimination among fallacies -- NOT a false-alarm rate on neutral political
   discourse. This evaluation therefore does not feed the prevalence correction.
 - Single-label assumption: each LOGIC item carries one fallacy type; a non-FD
   item is treated as FD-negative. A few items flagged to_review=1 (definitional/
   very short) should be checked by hand.
 - A public benchmark may be present in the model's pretraining: independent of
   the author is not guaranteed unseen by the model. Record model + date.

Only stage 1 (the classifier) matters here: the negation verifier touches only
manufactured_equivalence, so it cannot change a false_dilemma decision.

Run (needs OPENAI_API_KEY, or --backend ollama):
    python 23_logic_fd_eval.py --data logic_false_dilemma_eval.csv
    # add --split test to restrict to the strict held-out subset (few positives)
    # add --exclude-review to drop the to_review=1 rows
Outputs next to the data: logic_fd_predictions.csv and logic_fd_report.txt
"""
import argparse, json, time, os, random
import pandas as pd

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]
ABSTAIN = "insufficient_evidence"
TARGET = "false_dilemma"

# ---- frozen v2 prompt (verbatim from 16_classifier_v2_abstention.py) ----
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


def ask_openai(system, prompt, model):
    from openai import OpenAI
    out = OpenAI().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return out.choices[0].message.content


def ask_ollama(system, prompt, model, host="http://localhost:11434"):
    import urllib.request
    body = json.dumps({"model": model, "system": system, "prompt": prompt,
                       "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode()).get("response", "")


def make_asker(backend, model):
    if backend == "openai":
        return lambda s, p: ask_openai(s, p, model)
    if backend == "ollama":
        return lambda s, p: ask_ollama(s, p, model)
    raise SystemExit("unknown backend")


def call_with_retry(asker, s, p, tries=3):
    for i in range(tries):
        try:
            return asker(s, p)
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


def boot_ci(pairs, stat, reps=2000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    vals = []
    n = len(pairs)
    for _ in range(reps):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        v = stat(sample)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (vals[int(alpha / 2 * len(vals))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logic_false_dilemma_eval.csv")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--split", default=None, help="restrict to one split (e.g. test)")
    ap.add_argument("--exclude-review", action="store_true", help="drop to_review=1 rows")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.data)) or "."
    df = pd.read_csv(a.data)
    if a.split:
        df = df[df["split"] == a.split]
    if a.exclude_review and "to_review" in df.columns:
        df = df[df["to_review"] == 0]
    asker = make_asker(a.backend, a.model)

    rows = []
    for _, r in df.iterrows():
        pred = parse_label(call_with_retry(asker, SYSTEM,
                           PROMPT.format(defs=DEFS, text=r["text"])))
        rows.append({"id": r["id"], "gold_false_dilemma": int(r["gold_false_dilemma"]),
                     "source_label": r.get("source_label", ""), "split": r.get("split", ""),
                     "stage1_pred": pred,
                     "flag_fd": int(pred == TARGET), "abstain": int(pred == ABSTAIN),
                     "text": r["text"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(here, "logic_fd_predictions.csv"), index=False)

    def f1_stat(s):
        t = sum(1 for g, p in s if g == 1 and p == 1)
        fpv = sum(1 for g, p in s if g == 0 and p == 1)
        fnv = sum(1 for g, p in s if g == 1 and p == 0)
        pr = t / (t + fpv) if (t + fpv) else 0
        rc = t / (t + fnv) if (t + fnv) else 0
        return 2 * pr * rc / (pr + rc) if (pr + rc) else float("nan")

    L = []
    def e(s=""):
        L.append(s); print(s)

    def block(sub, label):
        tp = int(((sub.gold_false_dilemma == 1) & (sub.flag_fd == 1)).sum())
        fn = int(((sub.gold_false_dilemma == 1) & (sub.flag_fd == 0)).sum())
        fp = int(((sub.gold_false_dilemma == 0) & (sub.flag_fd == 1)).sum())
        tn = int(((sub.gold_false_dilemma == 0) & (sub.flag_fd == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        lo, hi = boot_ci(list(zip(sub.gold_false_dilemma.tolist(), sub.flag_fd.tolist())), f1_stat)
        e(f"[{label}]  n={len(sub)}  positives={tp+fn}  negatives={fp+tn}")
        e(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        e(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
        e(f"  specificity-vs-other-fallacies={spec:.3f}")
        e(f"  abstentions: on positives={int(((sub.gold_false_dilemma==1)&(sub.abstain==1)).sum())} "
          f"on negatives={int(((sub.gold_false_dilemma==0)&(sub.abstain==1)).sum())} (kept separate)")
        conf = sub[(sub.gold_false_dilemma == 0) & (sub.flag_fd == 1)]
        if len(conf):
            from collections import Counter
            e(f"  false FD-flags land on: {dict(Counter(conf.source_label))}")
        if (tp + fn) < 25:
            e(f"  NOTE: only {tp+fn} positives here -> sensitivity estimate is unstable.")
        e()

    e("LOGIC false_dilemma evaluation (independent labels; Jin et al. 2022)")
    e(f"model={a.model}")
    e("CONTAMINATION: the detector was not trained on LOGIC, but LOGIC is public and")
    e("may be in the base model's PRETRAINING; independent-of-author is not")
    e("independent-of-model. SCOPE: negatives are OTHER-FALLACY sentences, not neutral")
    e("text, so 'specificity' is fallacy discrimination, NOT a neutral-text false-alarm")
    e("rate, and this result does not feed the prevalence correction.\n")
    # Report the pooled set and the official test split SEPARATELY.
    block(out, "pooled (all splits)" if not a.split else f"split={a.split}")
    if not a.split and "split" in out.columns and (out["split"] == "test").any():
        block(out[out["split"] == "test"], "official LOGIC test split")
    open(os.path.join(here, "logic_fd_report.txt"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
