# -*- coding: utf-8 -*-
"""
30_dfa_classifier.py
Digital Fairness Act (DFA) component of the defensive audit layer.

On 11 November 2026 the European Commission presented its proposal for a
Digital Fairness Act: a Regulation against unfair B2C practices in the digital
environment (dark patterns, addictive design, unfair personalisation,
influencer marketing, virtual currencies and variable rewards, subscriptions).
This script adds a component that reads consumer-facing text (interface
microcopy, offer banners, subscription terms, influencer posts, in-game store
labels, customer-service notices) and names WHICH DFA-relevant practice, if
any, the trader's own text performs, together with the DFA provision group it
falls under.

SCOPE AND HONEST BOUNDARY
-------------------------
1. The DFA is consumer law (B2C). It does not regulate political speech; paid
   political advertising sits under Regulation (EU) 2024/900 (TTPA). This
   component therefore audits COMMERCIAL text. Its link to the Politomorphism
   programme is methodological (same target definition, same abstention, same
   separate outcome counting) and through the AI-evaluation arm: it can score
   whether an AI system's generated interface copy or customer-service answer
   reproduces a DFA-relevant practice.
2. Article references below follow the LEAKED draft reported on 9 November 2026
   (Contexte; summarised by law-firm notes). They are provision GROUPS, not
   legal citations, and must be re-checked against the official COM text
   before any publication.
3. An output label is a screening signal for a human reviewer, NOT a legal
   determination that a practice is unlawful. Several practices (false
   urgency, false scarcity) depend on facts outside the text (is the timer
   real? is stock really low?). The taxonomy marks these as
   requires_verification, and abstention ('insufficient_evidence') is an
   acceptable answer for them.
4. The seed set (dfa_seed_items.csv) is AUTHOR-BUILT: 20 vocabulary-matched
   problematic/control pairs and 10 hard items (negated, quoted, reported,
   policy-descriptive, and inverse cases with mild wording). Results on it are
   development evidence, not validation. Independent annotation is pending.

EVALUATED UNIT (same principle as 16_classifier_v2_abstention.py)
-----------------------------------------------------------------
We score the practice the TRADER'S OWN TEXT performs towards the consumer.
Text that reports, quotes, criticises, negates or describes a practice
(a news item, a regulator, a policy summary) is 'none'.

OUTCOMES ARE COUNTED SEPARATELY, never merged. On problematic items: detection
(any practice named), exact category, wrong category, missed ('none'),
abstention. On plain items: correct 'none', false alarm, abstention.
Abstentions stay in the denominator. For items whose gold_dependency is
'external', abstention is also reported as acceptable handling.

BACKENDS
--------
--backend rules    (default) transparent lexical cue rules, offline, no key.
                   This is the LEXICAL BASELINE: by construction it keys on
                   vocabulary, so it is expected to fail on quoted/negated
                   items and on inverse items with mild wording. It never
                   abstains.
--backend openai   OpenAI API; needs OPENAI_API_KEY and `pip install openai`.
                   Model via --model (e.g. gpt-4o-mini).
--backend ollama   a local model via Ollama (free, offline); e.g.
                   --model llama3.1:8b

USAGE
-----
    python dfa/30_dfa_classifier.py
    python dfa/30_dfa_classifier.py --backend openai --model gpt-4o-mini
    python dfa/30_dfa_classifier.py --backend ollama --model llama3.1:8b
    python dfa/30_dfa_classifier.py --file my_texts.csv --text-col text

With a file that has no gold_category column the script runs in AUDIT MODE:
it labels every row, attaches the DFA provision group, and prints a summary
of flagged areas, without computing accuracy.

Author: Șerban, G. F. (2026). ORCID 0009-0000-2266-3356. Code: MIT.
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE / "dfa_seed_items.csv"

# --------------------------------------------------------------------------
# Taxonomy: category -> definition, DFA provision group (leaked-draft refs),
# and whether a violation can be decided from the text alone.
# --------------------------------------------------------------------------
TAXONOMY = {
    "confirmshaming": {
        "definition": "the option to decline is worded so that refusing is framed "
                      "as foolish, careless or morally wrong.",
        "dfa_area": "Dark patterns (prohibited interface designs)",
        "dfa_ref": "Art. 10 (leaked draft): ban on interfaces that materially "
                   "distort behaviour + blacklist",
        "requires_verification": False,
    },
    "false_urgency_scarcity": {
        "definition": "the text pressures an immediate decision with a countdown, "
                      "low-stock or 'others are looking' claim. Whether the claim "
                      "is false usually cannot be seen in the text itself.",
        "dfa_area": "Dark patterns / misleading urgency",
        "dfa_ref": "Art. 10 (leaked draft) blacklist; cf. UCPD Annex I No 7",
        "requires_verification": True,
    },
    "trial_auto_conversion": {
        "definition": "a free trial or gift turns into a paid contract without the "
                      "consumer actively accepting it at the end of the trial.",
        "dfa_area": "Digital contracts and subscriptions",
        "dfa_ref": "Arts. 20-27 (leaked draft): reminder + active consent before "
                   "trial conversion",
        "requires_verification": False,
    },
    "obstructed_cancellation": {
        "definition": "ending the contract is made harder than entering it: no "
                      "online termination route, written or phone-only "
                      "cancellation, or long notice periods.",
        "dfa_area": "Digital contracts and subscriptions",
        "dfa_ref": "Arts. 20-27 (leaked draft): online termination facility; "
                   "termination at any time on short notice",
        "requires_verification": False,
    },
    "drip_pricing": {
        "definition": "a headline price is shown while mandatory fees are "
                      "deferred to a later step instead of being included up front.",
        "dfa_area": "Pricing practices",
        "dfa_ref": "Drip-pricing provisions (leaked draft; article not yet "
                   "confirmed)",
        "requires_verification": False,
    },
    "undisclosed_influencer_ad": {
        "definition": "a post promotes a product with a discount code, affiliate "
                      "or shop link but does not disclose that it is an ad, a "
                      "paid partnership or sponsored.",
        "dfa_area": "Influencer marketing",
        "dfa_ref": "Art. 11 (leaked draft): prominent disclosure of commercial "
                   "communications",
        "requires_verification": False,
    },
    "virtual_currency_opacity": {
        "definition": "an item or bundle is priced only in a virtual currency "
                      "(gems, coins, crystals) without its real-money value.",
        "dfa_area": "Digital monetisation / virtual currencies",
        "dfa_ref": "Arts. 14-17 (leaked draft): real-money value of virtual "
                   "currencies",
        "requires_verification": False,
    },
    "undisclosed_variable_reward": {
        "definition": "a paid chance mechanism (loot box, mystery chest, lucky "
                      "wheel) is offered without disclosing the probability of "
                      "each reward.",
        "dfa_area": "Digital monetisation / addictive design",
        "dfa_ref": "Arts. 14-17 (leaked draft): disclosure of odds for "
                   "variable-reward mechanisms",
        "requires_verification": False,
    },
    "no_human_contact": {
        "definition": "customer service is limited to an automated system with no "
                      "route to a human interlocutor.",
        "dfa_area": "Digital contracts and customer service",
        "dfa_ref": "Arts. 20-27 (leaked draft): right to a human interlocutor",
        "requires_verification": False,
    },
    "minor_targeted_monetisation": {
        "definition": "the text urges children directly to buy, or sells an "
                      "advantage over others (pay-to-win) to children.",
        "dfa_area": "Protection of minors / digital monetisation",
        "dfa_ref": "Arts. 14-17 (leaked draft): no pay-to-win / pay-to-progress "
                   "for minors; parental approval for paid currencies",
        "requires_verification": False,
    },
}
CATEGORIES = list(TAXONOMY)
NONE = "none"
ABSTAIN = "insufficient_evidence"
OUTPUTS = CATEGORIES + [NONE, ABSTAIN]

# --------------------------------------------------------------------------
# LLM backends (same pattern as 16_classifier_v2_abstention.py)
# --------------------------------------------------------------------------
DEFS = "\n".join(f"- {k}: {v['definition']}" for k, v in TAXONOMY.items()) + (
    "\n- none: the text states its offer, price or terms plainly, or it only "
    "reports, quotes, criticises, negates or describes a practice.\n"
    "- insufficient_evidence: you cannot decide without facts that are not in "
    "the text (for example whether a countdown or stock claim is genuine, or "
    "who the audience is)."
)
SYSTEM = ("You are a careful consumer-protection analyst. You identify a "
          "practice only when the trader's own text performs it towards the "
          "consumer. You never infer intent, and you never treat enthusiastic "
          "or emotional wording as proof on its own. You give a screening "
          "signal, not a legal ruling.")
PROMPT = (
    "Labels and definitions:\n\n{defs}\n\n"
    "Target. Identify only the practice that THIS TEXT ITSELF performs as "
    "trader-to-consumer communication. A practice that is negated (\"we never "
    "auto-renew\"), quoted or reported by a third party, criticised, or "
    "described as a rule is NOT performed: answer 'none'. Name a practice only "
    "if you can point to the fragment that performs it. If the problem would "
    "depend on facts outside the text, answer 'insufficient_evidence' unless "
    "the text itself already shows the problem.\n\n"
    "Text:\n\"{text}\"\n\n"
    "Reply with JSON only, in this form: "
    "{{\"label\": \"<one of: " + ", ".join(OUTPUTS) + ">\", "
    "\"fragment\": \"<the exact words that perform it, or empty>\"}}"
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


def call_with_retry(asker, system, prompt, tries=3):
    for i in range(tries):
        try:
            return asker(system, prompt)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def parse_llm(raw):
    """Return (label, fragment). JSON first; fall back to the longest label name found."""
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
            if lab in OUTPUTS:
                return lab, str(obj.get("fragment", "")).strip()
        except json.JSONDecodeError:
            pass
    t = text.lower().replace(" ", "_").replace("-", "_")
    for lab in sorted(OUTPUTS, key=len, reverse=True):
        if lab in t:
            return lab, ""
    return None, ""


# --------------------------------------------------------------------------
# Rules backend: the transparent lexical baseline
# --------------------------------------------------------------------------
I = re.IGNORECASE
DISCLOSED_AD = re.compile(r"#ad\b|(^|\W)ad\s*[·:|-]|\bsponsored\b|paid partnership|\badvert", I)
PRICE_MONEY = re.compile(r"€|\$|£|\beur\b|\busd\b", I)
ONLINE_ROUTE = re.compile(r"\bonline\b|cancel button|\bsettings\b|in the app|with one click", I)
HUMAN_ROUTE = re.compile(r"\bagent\b|\bperson\b|staff|\bhuman\b|\bemployee\b", I)
ODDS = re.compile(r"\bodds\b|%|probabilit|drop rate", I)

RULES = [
    ("confirmshaming", lambda t: re.search(
        r"no thanks,? i('d| would) rather|i don'?t care about|not recommended for "
        r"(responsible|smart|caring)|i prefer (paying|to pay) full", t, I)),
    ("false_urgency_scarcity", lambda t: re.search(
        r"\bhurry\b|only \d+ (left|rooms?|seats?|items?)|\d+ (people|others) (are )?"
        r"(looking|viewing)|disappears in|don'?t miss out|\bin \d{1,2}:\d{2}\b", t, I)),
    ("trial_auto_conversion", lambda t: re.search(
        r"(continues|renews|converts) automatically|billed after your trial|"
        r"you agree to a (yearly|monthly|annual) plan", t, I)),
    ("obstructed_cancellation", lambda t: re.search(
        r"(cancel|end your plan|terminat)", t, I) and re.search(
        r"\bcall\b|hotline|registered letter|in writing|written request|posted|"
        r"months? before", t, I) and not ONLINE_ROUTE.search(t)),
    ("drip_pricing", lambda t: re.search(
        r"\bfrom €\d+\*|plus taxes|added at (checkout|the final step)|"
        r"shown at checkout|excludes the mandatory|calculated at (payment|checkout)", t, I)),
    ("undisclosed_influencer_ad", lambda t: re.search(
        r"\bcode [A-Z0-9]{3,}|link in bio|link below|affiliate|grab yours", t, I)
        and not DISCLOSED_AD.search(t)),
    ("virtual_currency_opacity", lambda t: re.search(
        r"\d[\d,]*\s+(bonus\s+)?(gems|coins|crystals|diamonds|credits)\b", t, I)
        and not PRICE_MONEY.search(t) and not re.search(r"spin|chest|wheel|loot", t, I)),
    ("undisclosed_variable_reward", lambda t: re.search(
        r"mystery (chest|box)|loot ?box|lucky wheel|\bspin\b", t, I)
        and not ODDS.search(t)),
    ("no_human_contact", lambda t: re.search(
        r"virtual assistant|chatbot|\bbot\b", t, I) and re.search(
        r"\ball\b|\bonly\b|do not offer|no (phone|email)", t, I)
        and not HUMAN_ROUTE.search(t)),
    ("minor_targeted_monetisation", lambda t: re.search(
        r"\bkids\b|ask (mum|mom|dad|your parents)|your class|beat your friends", t, I)),
]


def classify_rules(text):
    for label, test in RULES:
        if test(text):
            return label, ""
    return NONE, ""


# --------------------------------------------------------------------------
# Evaluation (only when gold labels exist)
# --------------------------------------------------------------------------
def pct(a, b):
    return f"{a/b:.3f}  ({a}/{b})" if b else "n/a"


def evaluate(df):
    valid = df[df["pred"].notna()]
    if len(valid) < len(df):
        print(f"({len(df) - len(valid)} unparseable answers counted as missed)")
    pos = df[df["gold_category"] != NONE]
    neg = df[df["gold_category"] == NONE]

    print("\n(1) Items whose text performs a DFA-relevant practice "
          f"(n={len(pos)}):")
    det = pos["pred"].isin(CATEGORIES)
    exact = pos["pred"] == pos["gold_category"]
    abst = pos["pred"] == ABSTAIN
    missed = pos["pred"].isin([NONE]) | pos["pred"].isna()
    print(f"    detection (any practice named):  {pct(int(det.sum()), len(pos))}")
    print(f"    exact category:                  {pct(int(exact.sum()), len(pos))}")
    print(f"    wrong category:                  {pct(int((det & ~exact).sum()), len(pos))}")
    print(f"    missed ('none'):                 {pct(int(missed.sum()), len(pos))}")
    print(f"    abstained:                       {pct(int(abst.sum()), len(pos))}")
    ext = pos[pos["gold_dependency"] == "external"]
    if len(ext):
        ok = ext["pred"].isin([ABSTAIN]) | (ext["pred"] == ext["gold_category"])
        print(f"    of which need external facts (n={len(ext)}): flagged-for-verification "
              f"or abstained {pct(int(ok.sum()), len(ext))}")

    print(f"\n(2) Plain or non-performing items (n={len(neg)}):")
    print(f"    correct 'none':                  {pct(int((neg['pred'] == NONE).sum()), len(neg))}")
    print(f"    false alarm (practice named):    {pct(int(neg['pred'].isin(CATEGORIES).sum()), len(neg))}")
    print(f"    abstained:                       {pct(int((neg['pred'] == ABSTAIN).sum()), len(neg))}")
    hard_neg = neg[neg["member"] == "hard"]
    if len(hard_neg):
        print(f"    hard negatives only (negated/quoted/reported/policy, n={len(hard_neg)}): "
              f"false alarm {pct(int(hard_neg['pred'].isin(CATEGORIES).sum()), len(hard_neg))}")

    pairs = df[df["pair_id"].notna() & (df["pair_id"].astype(str) != "")]
    n_pairs, ok_pairs = 0, 0
    for _, g in pairs.groupby("pair_id"):
        p = g[g["member"] == "problematic"]
        c = g[g["member"] == "control"]
        if len(p) != 1 or len(c) != 1:
            continue
        n_pairs += 1
        p, c = p.iloc[0], c.iloc[0]
        p_ok = p["pred"] == p["gold_category"] or (
            p["gold_dependency"] == "external" and p["pred"] == ABSTAIN)
        if p_ok and c["pred"] == NONE:
            ok_pairs += 1
    print(f"\n(3) Pair accuracy (problematic member correct AND control 'none'): "
          f"{pct(ok_pairs, n_pairs)}")

    print("\n(4) Per-category recall (exact category):")
    for cat in CATEGORIES:
        g = pos[pos["gold_category"] == cat]
        if len(g):
            print(f"      {cat:30s} {pct(int((g['pred'] == cat).sum()), len(g))}")

    errs = df[(df["pred"] != df["gold_category"]) &
              ~((df["gold_dependency"] == "external") & (df["pred"] == ABSTAIN))]
    if len(errs):
        print("\n(5) Errors, for audit:")
        for _, r in errs.iterrows():
            print(f"      {r['item_id']:5s} gold={r['gold_category']:28s} pred={r['pred']}")


def audit_summary(df):
    flagged = df[df["pred"].isin(CATEGORIES)]
    print(f"\nAUDIT MODE: {len(df)} texts, {len(flagged)} flagged, "
          f"{int((df['pred'] == ABSTAIN).sum())} abstained.")
    if len(flagged):
        print("Flagged by DFA area:")
        print(flagged["dfa_area"].value_counts().to_string())
        nver = int(flagged["requires_verification"].sum())
        if nver:
            print(f"{nver} flag(s) depend on facts outside the text and need verification.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--backend", default="rules", choices=["rules", "openai", "ollama"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.file)
    df = pd.read_csv(src, keep_default_na=False)
    if args.text_col not in df.columns:
        raise SystemExit(f"column '{args.text_col}' not found in {src.name}")

    if args.backend == "rules":
        classify = classify_rules
        tag = "rules"
    else:
        asker = ((lambda s, p: ask_openai(s, p, args.model)) if args.backend == "openai"
                 else (lambda s, p: ask_ollama(s, p, args.model)))
        tag = f"{args.backend}_{args.model}".replace(":", "-").replace("/", "-")

        def classify(text):
            raw = call_with_retry(asker, SYSTEM, PROMPT.format(defs=DEFS, text=text))
            return parse_llm(raw)

    print(f"DFA component: backend={args.backend}"
          + ("" if args.backend == "rules" else f", model={args.model}")
          + f", file={src.name}, n={len(df)}")
    preds, frags = [], []
    for i, text in enumerate(df[args.text_col].astype(str)):
        lab, frag = classify(text)
        preds.append(lab)
        frags.append(frag)
        if args.backend != "rules" and (i + 1) % 10 == 0:
            print(f"   {i + 1}/{len(df)} done")

    df["pred"] = preds
    df["fragment"] = frags
    df["dfa_area"] = [TAXONOMY[p]["dfa_area"] if p in TAXONOMY else "" for p in preds]
    df["dfa_ref"] = [TAXONOMY[p]["dfa_ref"] if p in TAXONOMY else "" for p in preds]
    df["requires_verification"] = [TAXONOMY[p]["requires_verification"] if p in TAXONOMY
                                   else False for p in preds]

    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_pred_{tag}.csv")
    df.to_csv(out, index=False)

    print("\n=====================  DFA COMPONENT  =====================")
    print("Screening signal for a human reviewer, NOT a legal determination.")
    print("Article references follow the leaked draft (9 Nov 2026); verify")
    print("against the official Commission text before citing them.")
    if "gold_category" in df.columns:
        evaluate(df)
    else:
        audit_summary(df)
    print(f"\nSaved {out.name} (label, fragment, DFA area and provision group per text).")
    if args.backend == "rules":
        print("\nReading: the rules backend is the LEXICAL BASELINE. Its false alarms on")
        print("hard negatives and its misses on inverse items show how much of the task")
        print("vocabulary alone can and cannot solve; an LLM backend should be judged")
        print("against these numbers, not in isolation.")


if __name__ == "__main__":
    main()
