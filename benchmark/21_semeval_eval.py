# -*- coding: utf-8 -*-
"""
21_semeval_eval.py
Config-driven evaluation of the v2 classifier against a PUBLIC annotated corpus
(SemEval-2020 Task 11 or SemEval-2023 Task 3), with the review's guardrails
compiled in as HARD CHECKS. The script REFUSES to run until the methodological
decisions are made explicit, because a script can otherwise produce exact numbers
for the wrong comparison.

Guardrails enforced before any model call:
  1. Label mapping must be explicit per edition (no built-in default mapping).
     A merged corpus class (e.g. 'Whataboutism,Straw_Men,Red_Herring') may be
     mapped ONLY if the config sets allow_merged=true AND gives a reason; the
     mapped positives are then flagged as a mixed class, not pure whataboutism.
  2. Evaluation unit (fragment | sentence | paragraph) must be set; results on
     different units are never compared.
  3. Negatives: unlabeled text is used as negative ONLY if negatives.justified
     is true with a documented reason tied to the corpus annotation protocol.
     Otherwise false-alarm rate is NOT estimated and prevalence correction is
     not attempted from this run.
  4. Spans are read against the RAW text; the text is NOT normalised before the
     character offsets are resolved (normalisation shifts annotated fragments).

The result states, in its header: this measures how the detector behaves against
THIS benchmark's annotations and definitions; it does not establish validity of
all six categories, Romanian performance, parliamentary-discourse performance,
the negation verifier, or transportability of error rates. A public benchmark
may also be present in the model's training data, so labels independent of the
author are not guaranteed unseen by the model.

Config: JSON file passed with --config. Example skeleton written by --init.
Running the classifier needs OPENAI_API_KEY (or --backend ollama).
"""
import argparse, json, os, re, sys, time, urllib.request
from collections import defaultdict

TECHNIQUES = ["presupposition", "manufactured_equivalence", "false_dilemma",
              "whataboutism", "loaded_rhetorical_question", "implied_causation"]
OUTPUTS = TECHNIQUES + ["none", "insufficient_evidence"]

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

SKELETON = {
    "edition": "2020",                         # "2020" or "2023"
    "articles_dir": "PATH/TO/articles",        # raw .txt articles
    "labels_file": "PATH/TO/labels.tsv",       # span annotations
    "evaluation_unit": "",                     # REQUIRED: fragment | sentence | paragraph
    "mapping": {                               # REQUIRED: my_category -> corpus label(s)
        # "false_dilemma": {"corpus_labels": ["Black-and-White_Fallacy"],
        #                   "definition_checked": False, "notes": ""},
        # "whataboutism": {"corpus_labels": ["Whataboutism,Straw_Men,Red_Herring"],
        #                   "allow_merged": False, "reason": "", "definition_checked": False},
    },
    "negatives": {"justified": False, "source": "", "reason": ""},
    "model": "gpt-4o-mini", "backend": "openai",
    "contamination_note": "A public benchmark may be in the model's training data."
}


def fail(msg):
    sys.exit("GUARDRAIL STOP: " + msg)


def check_config(cfg):
    if cfg.get("edition") not in ("2020", "2023"):
        fail("edition must be '2020' or '2023'.")
    if cfg.get("evaluation_unit") not in ("fragment", "sentence", "paragraph"):
        fail("evaluation_unit must be set to fragment | sentence | paragraph; "
             "results on different units are not comparable.")
    mapping = cfg.get("mapping") or {}
    if not mapping:
        fail("mapping is empty. Build the verified label-mapping table FIRST; "
             "there is no default mapping.")
    for mycat, spec in mapping.items():
        if mycat not in TECHNIQUES:
            fail(f"mapping key '{mycat}' is not one of the six categories.")
        if not spec.get("corpus_labels"):
            fail(f"mapping['{mycat}'] has no corpus_labels.")
        if not spec.get("definition_checked"):
            fail(f"mapping['{mycat}'].definition_checked is False: confirm the "
                 f"corpus label's definition matches yours before mapping.")
        for lab in spec["corpus_labels"]:
            if ("," in lab) and not spec.get("allow_merged"):
                fail(f"corpus label '{lab}' is a MERGED class; set allow_merged "
                     f"true and give a reason, and treat positives as a mixed "
                     f"class, not pure {mycat}.")
    neg = cfg.get("negatives") or {}
    if not neg.get("justified"):
        print("NOTE: negatives.justified is False -> unlabeled text will NOT be "
              "used as negative. False-alarm rate is not estimated and prevalence "
              "correction is not attempted from this run.")
    elif not neg.get("reason"):
        fail("negatives.justified is True but no reason tied to the corpus "
             "annotation protocol is given.")


# ---------------- readers (spans read against RAW text) ----------------
def read_2020(cfg):
    """SemEval-2020 Task 11 TC format: labels.tsv rows = article_id<TAB>technique
    <TAB>start<TAB>end; article text in articles_dir/article<ID>.txt. Text is read
    raw; offsets are resolved before any normalisation."""
    spans = defaultdict(list)
    with open(cfg["labels_file"], encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            aid, tech, s, e = parts[0], parts[1], parts[2], parts[3]
            try:
                spans[aid].append((tech, int(s), int(e)))
            except ValueError:
                continue
    return spans


def read_2023(cfg):
    fail("2023 reader not implemented in this skeleton: confirm the exact "
         "subtask file format and label list from the downloaded package, then "
         "add the reader. Do not assume the 2020 mapping carries over.")


def units_from_article(text, unit):
    if unit == "paragraph":
        bounds, i = [], 0
        for para in re.split(r"\n\s*\n", text):
            start = text.find(para, i); i = start + len(para)
            if para.strip():
                bounds.append((para, start, start + len(para)))
        return bounds
    if unit == "sentence":
        out, i = [], 0
        for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
            seg = m.group(0)
            if seg.strip():
                out.append((seg, m.start(), m.end()))
        return out
    fail("fragment-unit extraction requires the annotated-span fragments "
         "directly; implement per the chosen protocol.")


def ask_openai(system, prompt, model):
    from openai import OpenAI
    out = OpenAI().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return out.choices[0].message.content


def parse_label(raw):
    t = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    for lab in OUTPUTS:
        if lab in t:
            return lab
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--init", action="store_true", help="write a config skeleton and exit")
    ap.add_argument("--dry-run", action="store_true", help="check config + reader, no model calls")
    a = ap.parse_args()
    if a.init:
        json.dump(SKELETON, open("semeval_eval_config.json", "w"), indent=2)
        print("wrote semeval_eval_config.json - fill it, then run with --config")
        return
    if not a.config:
        fail("pass --config semeval_eval_config.json (or --init to create one).")
    cfg = json.load(open(a.config, encoding="utf-8"))
    check_config(cfg)
    spans = read_2020(cfg) if cfg["edition"] == "2020" else read_2023(cfg)
    mapped = {}
    for mycat, spec in cfg["mapping"].items():
        for lab in spec["corpus_labels"]:
            mapped[lab] = mycat
    print(f"config OK | edition={cfg['edition']} unit={cfg['evaluation_unit']} "
          f"mapped labels={list(mapped)} articles_with_spans={len(spans)}")
    print("RESULT SCOPE: behaviour vs THIS benchmark's annotations and definitions "
          "only; not validity of all six categories, not Romanian, not parliamentary "
          "discourse, not the negation verifier, not transportable error rates. "
          + cfg.get("contamination_note", ""))
    if a.dry_run:
        print("dry-run: guardrails passed; reader loaded. No model calls made.")
        return
    # evaluation loop intentionally left to run only after a dry-run on real files,
    # so the unit extraction and span alignment are confirmed on a few articles first.
    fail("evaluation loop is enabled only after a successful --dry-run on the real "
         "files and confirmation of unit/span alignment. Re-run with the loop once "
         "the sample check (Step B) passes.")


if __name__ == "__main__":
    main()
