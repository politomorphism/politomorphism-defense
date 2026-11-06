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
     Otherwise the FULL evaluation is blocked. CHANGE FROM THE EARLIER PROTOCOL:
     the earlier protocol excluded any positives-only run; this version adds a
     separate, clearly bounded --mode exploratory that reports the flag rate on
     compatible positives only. It does NOT relax the full-evaluation rule, does
     NOT report precision/specificity/false-alarm, and is NEVER used for
     prevalence correction. The two are documented as distinct modes, not two
     contradictory rules.
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
        # Each mapping needs a justification RECORD (existence checked here, not
        # scientific validity), and merged classes need explicit handling.
        # "false_dilemma": {
        #   "corpus_labels": ["Black-and-White_Fallacy"],
        #   "justification": {"your_definition": "", "corpus_definition": "",
        #                     "source": "version/section/page", "differences": "",
        #                     "decision": "", "rationale": ""}},
        # "whataboutism": {
        #   "corpus_labels": ["Whataboutism,Straw_Men,Red_Herring"],
        #   "merged_handling": "exclude",          # or "combined_task"
        #   "combined_outputs": [],                # required iff combined_task
        #   "task_note": "",
        #   "justification": {...}},
    },
    # Three operational classes (never two): positive; verified_negative (meets
    # the corpus annotation protocol's conditions); unlabeled_uncertain (status
    # not confirmed -> EXCLUDED from the main binary evaluation).
    "negatives": {"justified": False, "rule": "", "reason": "",
                  "exclude_uncertain": True},
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
    JREQ = ["your_definition", "corpus_definition", "source", "differences",
            "decision", "rationale"]
    for mycat, spec in mapping.items():
        if mycat not in TECHNIQUES:
            fail(f"mapping key '{mycat}' is not one of the six categories.")
        if not spec.get("corpus_labels"):
            fail(f"mapping['{mycat}'] has no corpus_labels.")
        # a justification RECORD must exist (existence checked, not its validity)
        j = spec.get("justification") or {}
        missing = [k for k in JREQ if not str(j.get(k, "")).strip()]
        if missing:
            fail(f"mapping['{mycat}'].justification is incomplete (missing/empty: "
                 f"{missing}). A ticked flag is not evidence; record your "
                 f"definition, the corpus definition, the exact source "
                 f"(version/section/page), the differences, the decision and its "
                 f"rationale. The script checks this record exists, not that the "
                 f"mapping is scientifically correct.")
        for lab in spec["corpus_labels"]:
            if "," in lab:  # merged corpus class
                mh = spec.get("merged_handling")
                if mh not in ("exclude", "combined_task"):
                    fail(f"corpus label '{lab}' is a MERGED class; set "
                         f"merged_handling to 'exclude' (drop it from the "
                         f"{mycat}-specific evaluation) or 'combined_task'. "
                         f"Allowing it does not make it a valid reference for "
                         f"{mycat} alone.")
                if mh == "combined_task":
                    comps = [c.strip() for c in lab.split(",") if c.strip()]
                    cout = spec.get("combined_outputs") or {}
                    if not cout:
                        fail(f"merged_handling='combined_task' for '{lab}' needs "
                             f"combined_outputs: a detector output for EACH "
                             f"component of the merged class. Absence of a {mycat} "
                             f"flag on a non-{mycat} member is NOT scored as a "
                             f"{mycat} miss.")
                    # the detector outputs must actually cover the SAME union of
                    # techniques as the corpus label; an explicit list is not
                    # enough if the detector cannot represent every component.
                    uncovered = [c for c in comps if not str(cout.get(c, "")).strip()]
                    if uncovered:
                        fail(f"combined_task for '{lab}' is invalid: the detector "
                             f"has no output mapped for component(s) {uncovered}. "
                             f"The combined task must span the same union of "
                             f"techniques as the corpus label; since the detector "
                             f"does not cover all components, use "
                             f"merged_handling='exclude' instead.")
                    bad = [c for c, o in cout.items() if o not in OUTPUTS]
                    if bad:
                        fail(f"combined_outputs maps component(s) to non-detector "
                             f"labels: {bad}. Use one of {OUTPUTS}.")
    neg = cfg.get("negatives") or {}
    mode = cfg.get("_mode", "full")
    if mode == "full" and not neg.get("justified"):
        fail("full evaluation is BLOCKED without justified negatives: unlabeled "
             "text is not automatically negative. Either justify negatives "
             "(negatives.justified=true with a reason tied to the corpus protocol) "
             "or run --mode exploratory (flag rate on compatible positives only; "
             "no precision/specificity/false-alarm; not used for prevalence).")
    if neg.get("justified") and not neg.get("reason"):
        fail("negatives.justified is True but no reason tied to the corpus "
             "annotation protocol is given.")


# ---------------- readers (spans read against RAW text) ----------------
def read_2020(cfg):
    """SemEval-2020 Task 11 TC format: labels.tsv rows = article_id<TAB>technique
    <TAB>start<TAB>end; article text in articles_dir/article<ID>.txt. Text is read
    raw; offsets are resolved before any normalisation."""
    if not os.path.exists(cfg.get("labels_file", "")):
        fail(f"labels_file not found: {cfg.get('labels_file')!r}. Bring the sample "
             f"annotation file(s) and set the path before the dry-run.")
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


# ---------------- statistics (reported only in full mode) ----------------
def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p and r and (p + r)) else float("nan")
    return p, r, f1


def spec_fpr(tn, fp):
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    return sp, (1 - sp) if sp == sp else float("nan")


def macro_micro(per_cat):
    """per_cat: {cat: (tp,fp,fn)}. Returns macro-F1 and micro-F1."""
    f1s, TP, FP, FN = [], 0, 0, 0
    for tp, fp, fn in per_cat.values():
        _, _, f1 = prf(tp, fp, fn)
        if f1 == f1:
            f1s.append(f1)
        TP += tp; FP += fp; FN += fn
    macro = sum(f1s) / len(f1s) if f1s else float("nan")
    _, _, micro = prf(TP, FP, FN)
    return macro, micro


def bootstrap_ci_by_document(items, statistic, reps=2000, seed=0, alpha=0.05):
    """items: list of (doc_id, record). Resamples DOCUMENTS (clusters), not
    fragments, so fragments of one article are not treated as independent."""
    import random
    rng = random.Random(seed)
    by_doc = defaultdict(list)
    for doc_id, rec in items:
        by_doc[doc_id].append(rec)
    docs = list(by_doc)
    if not docs:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(reps):
        sample = []
        for _ in range(len(docs)):
            sample.extend(by_doc[docs[rng.randrange(len(docs))]])
        vals.append(statistic(sample))
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return (float("nan"), float("nan"))
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


def verify_span_alignment(articles_dir, spans, sample=50):
    """Reconstruct annotated fragments from offsets against RAW text; report
    mismatches (empty or out-of-range). No normalisation is applied."""
    checked = bad = 0
    for aid, sp in list(spans.items())[:sample]:
        path = os.path.join(articles_dir, f"article{aid}.txt")
        if not os.path.exists(path):
            path = os.path.join(articles_dir, f"{aid}.txt")
        if not os.path.exists(path):
            bad += len(sp); checked += len(sp); continue
        raw = open(path, encoding="utf-8").read()
        for tech, s, e in sp:
            checked += 1
            if not (0 <= s < e <= len(raw)) or not raw[s:e].strip():
                bad += 1
    return checked, bad


def run_manifest(cfg, extra):
    import subprocess
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=os.path.dirname(os.path.abspath(__file__)),
                                    stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = "unknown"
    man = {"git_commit": h, "model": cfg.get("model"), "edition": cfg.get("edition"),
           "evaluation_unit": cfg.get("evaluation_unit"), "mode": cfg.get("_mode"),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    man.update(extra)
    return man


def selftest():
    """Prove the statistics on synthetic data, with no corpus or model."""
    per_cat = {"false_dilemma": (8, 2, 2), "implied_causation": (3, 1, 6)}
    assert abs(prf(8, 2, 2)[0] - 0.8) < 1e-9
    sp, fpr = spec_fpr(90, 10); assert abs(sp - 0.9) < 1e-9 and abs(fpr - 0.1) < 1e-9
    macro, micro = macro_micro(per_cat)
    items = [(f"doc{i%5}", 1 if i % 4 else 0) for i in range(200)]
    lo, hi = bootstrap_ci_by_document(items, lambda s: sum(s) / len(s), reps=500)
    print(f"selftest OK: macroF1={macro:.3f} microF1={micro:.3f} "
          f"doc-bootstrap mean-CI=[{lo:.3f},{hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--init", action="store_true", help="write a config skeleton and exit")
    ap.add_argument("--selftest", action="store_true", help="prove the statistics on synthetic data, no corpus/model")
    ap.add_argument("--dry-run", action="store_true", help="check config + reader + span alignment, no model calls")
    ap.add_argument("--mode", choices=["full", "exploratory"], default="full",
                    help="full needs justified negatives; exploratory reports flag "
                         "rate on compatible positives only (no precision/specificity/"
                         "false-alarm; not used for prevalence correction)")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.init:
        json.dump(SKELETON, open("semeval_eval_config.json", "w"), indent=2)
        print("wrote semeval_eval_config.json - fill it, then run with --config")
        return
    if not a.config:
        fail("pass --config semeval_eval_config.json (or --init to create one).")
    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg["_mode"] = a.mode
    check_config(cfg)
    if a.mode == "exploratory":
        print("MODE=exploratory: flag rate on compatible positives ONLY. This mode "
              "does NOT report precision, specificity, or false-alarm rate, and is "
              "NOT used for prevalence correction.")
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
        # mechanical rows of the go/no-go table (conceptual compatibility stays a
        # human judgement and is NOT decided here)
        if os.path.isdir(cfg.get("articles_dir", "")) and spans:
            checked, bad = verify_span_alignment(cfg["articles_dir"], spans)
            print(f"span alignment: {checked-bad}/{checked} fragments reconstruct "
                  f"from raw-text offsets ({'OK' if bad==0 else str(bad)+' MISMATCH'})")
        else:
            print("span alignment: not checked (articles_dir or labels not present "
                  "yet) - bring sample files to run this check.")
        neg = cfg.get("negatives") or {}
        print(f"negatives: justified={bool(neg.get('justified'))} "
              f"exclude_uncertain={neg.get('exclude_uncertain', True)} "
              f"(unlabeled_uncertain is excluded from the main binary evaluation)")
        print("conceptual compatibility (your definition vs corpus definition): "
              "HUMAN JUDGEMENT - not decided by this script.")
        print("dry-run: guardrails passed; reader loaded. No model calls made.")
        return
    # evaluation loop intentionally left to run only after a dry-run on real files,
    # so the unit extraction and span alignment are confirmed on a few articles first.
    fail("evaluation loop is enabled only after a successful --dry-run on the real "
         "files and confirmation of unit/span alignment. Re-run with the loop once "
         "the sample check (Step B) passes.")


if __name__ == "__main__":
    main()
