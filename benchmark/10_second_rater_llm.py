# -*- coding: utf-8 -*-
"""
10_second_rater_llm.py
An INDEPENDENT second rater for the contrastive-pairs benchmark (v0.3, 60 pairs).

Why this exists
---------------
The v0.3 pairs and their manipulative/control labels were written by the author
alone. A reviewer will ask: are those labels just one person's judgment? The
standard answer is a second annotator and an inter-annotator agreement score.
This script provides a second annotator that is NOT a human and NOT the model
under test: a separate large language model that never took part in building the
pairs. It judges every sentence BLIND and independently, and the script reports
how often it agrees with the author's labels, together with Cohen's kappa -- the
same statistic a human second annotator would give.

It does two things:
  (1) PER-MEMBER labelling (primary). All 120 sentences are shuffled so no pair
      is ever seen side by side. The rater labels each one, alone, as
      MANIPULATIVE or PLAIN. We then compute raw agreement and Cohen's kappa
      against the author labels over the 120 items.
  (2) PAIRWISE forced choice (secondary). For each pair, the two members are
      shown in a random order and the rater is asked which is more manipulative.
      We report how often it picks the author's manipulative member, with a
      binomial confidence interval. This mirrors the pair-accuracy metric.

Every raw answer is saved, so the whole judgment is auditable and reproducible.

Backends
--------
--backend ollama   (default) a local model via Ollama. Free, offline, no key.
                   Install Ollama (https://ollama.com), then in a terminal:
                       ollama pull llama3.1:8b
                   and leave Ollama running. Pick any model you like with
                   --model; use a different family from DistilBERT so the rater
                   is genuinely independent.
--backend openai   OpenAI API. Needs the environment variable OPENAI_API_KEY and
                   `pip install openai`. Choose the model with --model
                   (e.g. gpt-4o-mini).

Usage
-----
    python 10_second_rater_llm.py
    python 10_second_rater_llm.py --backend ollama --model llama3.1:8b
    python 10_second_rater_llm.py --backend openai --model gpt-4o-mini

Put this file next to contrastive_pairs_v3.csv. About 120 + 60 short calls; a
local 8B model takes a few minutes on CPU.
"""
import argparse, json, os, random, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PAIRS = Path("contrastive_pairs_v3.csv")
SHUFFLE_SEED = 7           # fixes the presentation order, for reproducibility
MEMBER_OUT = Path("second_rater_member_judgments.csv")
PAIR_OUT = Path("second_rater_pairwise_judgments.csv")

# ---- the rater's instructions (identical wording for every item) ----
MEMBER_SYSTEM = (
    "You are a careful analyst of political language. You judge the RHETORICAL "
    "STRUCTURE of a single sentence, not its topic or whether you agree with it."
)
MEMBER_PROMPT = (
    "Does the sentence below use a MANIPULATIVE rhetorical structure -- for "
    "example presupposing a contested claim as if settled, forcing a false "
    "either/or, deflecting criticism by pointing at the other side "
    "(whataboutism), a loaded rhetorical question that asserts its own answer, "
    "implying causation merely from sequence, or equating two things as "
    "identical -- or does it state its content PLAINLY, as an open, questioned, "
    "or attributed claim without such a device?\n\n"
    "Sentence:\n\"{text}\"\n\n"
    "Answer with exactly one word: MANIPULATIVE or PLAIN."
)
PAIR_SYSTEM = MEMBER_SYSTEM
PAIR_PROMPT = (
    "Two sentences cover the same topic with nearly the same words. One uses a "
    "manipulative rhetorical structure; the other states the content plainly. "
    "Which one is MORE manipulative in structure?\n\n"
    "Sentence 1:\n\"{t1}\"\n\n"
    "Sentence 2:\n\"{t2}\"\n\n"
    "Answer with exactly one character: 1 or 2."
)


# ---------- backends ----------
def ask_ollama(system, prompt, model, host="http://localhost:11434"):
    body = json.dumps({
        "model": model, "system": system, "prompt": prompt,
        "stream": False, "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def ask_openai(system, prompt, model):
    from openai import OpenAI
    client = OpenAI()
    out = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return out.choices[0].message.content


def make_asker(backend, model):
    if backend == "ollama":
        return lambda s, p: ask_ollama(s, p, model)
    if backend == "openai":
        return lambda s, p: ask_openai(s, p, model)
    raise SystemExit(f"unknown backend: {backend}")


def call_with_retry(asker, system, prompt, tries=3):
    for i in range(tries):
        try:
            return asker(system, prompt)
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------- parsing ----------
def parse_member(raw):
    t = (raw or "").strip().lower()
    man = "manipulative" in t
    pln = "plain" in t
    if man and not pln:
        return "manipulative"
    if pln and not man:
        return "control"
    # fall back to the first keyword that appears
    im = t.find("manipulat"); ip = t.find("plain")
    if im == -1 and ip == -1:
        return None
    if ip == -1 or (im != -1 and im < ip):
        return "manipulative"
    return "control"


def parse_pair(raw):
    t = (raw or "").strip()
    for ch in t:
        if ch == "1":
            return 1
        if ch == "2":
            return 2
    return None


# ---------- metrics ----------
def binom_ci(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    se = (p * (1 - p) / n) ** 0.5
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def kappa_band(k):
    if k < 0: return "poor (worse than chance)"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()
    asker = make_asker(args.backend, args.model)
    print(f"Independent second rater: backend={args.backend}, model={args.model}")

    pairs = pd.read_csv(PAIRS)
    rng = random.Random(SHUFFLE_SEED)

    # ---------- (1) per-member, shuffled, independent ----------
    members = pairs[["pair_id", "technique", "member", "text"]].copy()
    members["true_label"] = members["member"]           # manipulative / control
    idx = list(members.index); rng.shuffle(idx)
    members = members.loc[idx].reset_index(drop=True)

    recs = []
    print(f"\n[1/2] Labelling {len(members)} sentences one by one (blind)...")
    for i, row in members.iterrows():
        raw = call_with_retry(asker, MEMBER_SYSTEM, MEMBER_PROMPT.format(text=row.text))
        lab = parse_member(raw)
        recs.append({"pair_id": row.pair_id, "technique": row.technique,
                     "true_label": row.true_label, "rater_label": lab,
                     "raw": (raw or "").strip().replace("\n", " ")[:200]})
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(members)} done")
    md = pd.DataFrame(recs)
    md.to_csv(MEMBER_OUT, index=False)

    valid = md.dropna(subset=["rater_label"])
    n = len(valid); miss = len(md) - n
    agree = (valid["true_label"] == valid["rater_label"]).mean() if n else 0
    kappa = cohen_kappa_score(valid["true_label"], valid["rater_label"]) if n else float("nan")

    # ---------- (2) pairwise forced choice ----------
    wide = pairs.pivot_table(index=["pair_id", "technique"], columns="member",
                             values="text", aggfunc="first").reset_index()
    precs = []
    print(f"\n[2/2] Pairwise forced choice on {len(wide)} pairs (random order)...")
    for i, row in wide.iterrows():
        swap = rng.random() < 0.5
        t1, t2 = (row.control, row.manipulative) if swap else (row.manipulative, row.control)
        raw = call_with_retry(asker, PAIR_SYSTEM, PAIR_PROMPT.format(t1=t1, t2=t2))
        pick = parse_pair(raw)
        manip_pos = 2 if swap else 1
        correct = (pick == manip_pos) if pick in (1, 2) else None
        precs.append({"pair_id": row.pair_id, "technique": row.technique,
                      "manip_shown_as": manip_pos, "rater_pick": pick,
                      "correct": correct, "raw": (raw or "").strip().replace("\n", " ")[:80]})
    pd_ = pd.DataFrame(precs); pd_.to_csv(PAIR_OUT, index=False)
    pv = pd_.dropna(subset=["correct"]); pn = len(pv)
    pk = int(pv["correct"].sum()); pacc = pk / pn if pn else 0
    plo, phi = binom_ci(pk, pn)

    # ---------- report ----------
    print("\n==================  INDEPENDENT SECOND RATER  ==================")
    print(f"Backend / model: {args.backend} / {args.model}")
    print("\n(1) Per-member agreement with the author labels (the inter-annotator number):")
    print(f"    sentences judged: {n}/120" + (f"  ({miss} unparseable, excluded)" if miss else ""))
    print(f"    raw agreement:    {agree:.3f}")
    print(f"    Cohen's kappa:    {kappa:.3f}   ->  {kappa_band(kappa)}")
    print("\n(2) Pairwise forced choice (same metric as the benchmark):")
    print(f"    pairs judged:     {pn}/60")
    print(f"    picks author's manipulative member: {pacc:.3f}  [95% CI {plo:.2f}, {phi:.2f}]")
    print("\nPer technique (per-member agreement):")
    for t, g in valid.groupby("technique"):
        a = (g["true_label"] == g["rater_label"]).mean()
        print(f"    {t:26s} {a:.2f}  ({len(g)} sentences)")
    print("\nSaved:")
    print(f"    {MEMBER_OUT.name}  (every per-member judgment + raw answer)")
    print(f"    {PAIR_OUT.name}  (every pairwise judgment + raw answer)")
    print("\nHow to read it: Cohen's kappa is the number a reviewer wants. Above")
    print("0.6 is substantial agreement, above 0.8 almost perfect. It says an")
    print("independent model, blind to how the pairs were built, recovers the")
    print("author's manipulative/plain distinction -- so the labels are not")
    print("one person's private judgment. Report it honestly as a model rater,")
    print("not a human one.")


if __name__ == "__main__":
    main()
