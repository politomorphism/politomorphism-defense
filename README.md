# Politomorphism — Defensive Political Anthropology Layer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23884579.svg)](https://doi.org/10.5281/zenodo.23884579)

Reproducible code and working papers on detecting political manipulation as a **defensive** practice — detection, measurement, and explanation, never generation. The code only classifies and explains existing text.

> **Where the research stands:** see [`STATUS.md`](STATUS.md) for a dated progress map — what is done, what is reported on author-built data, what is only proposed, and the next step.

## How to cite

> Șerban, G. F. (2026). *A Leakage-Controlled Evaluation of Contextual and Lexical Models for Propaganda Classification.* Zenodo. https://doi.org/10.5281/zenodo.23884579

ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)

## Correction (Version 2, November 2026)

Independent reproduction audits found two problems in the first versions, now corrected:

1. **Data leakage in the official split.** The English PropXplain partitions together contain only ~250 news articles whose paragraphs are scattered across train/validation/test, so every test article also appears in training. The original evaluation was therefore not article-independent. The corrected analysis pools all rows, removes exact-text duplicates, and **re-splits by article** (no article spans two sets). On the article-disjoint test set the baseline's performance is essentially unchanged (propagandistic-class F1 0.525 vs. 0.506), so the leakage did not collapse the result — but because the test sets differ, this is robustness to the protocol change, not proof of no overestimation.
2. **An over-stated error-analysis claim is retracted.** Version 1 claimed the lexical model "systematically misses" structural rather than overt manipulation. A blinded audit did not support this (overt cues were present in most false negatives; structural forms were not enriched; the "9/11" example contains overt loaded vocabulary). The claim is withdrawn and not replaced with a new one.

The corrected paper is `Politomorphism_Defensive_Detection_WorkingPaper_v2_corrected.*`, which **supersedes** Version 1 (kept for history). Version 2 also adds bootstrap confidence intervals, a matched-precision analysis, a comparison across lexical baselines, and explicit construct-validity and ethical-scoping caveats.

## Results (article-disjoint, corrected)

Held-out test set (862 paragraphs; 264 propagandistic). Propagandistic class, 95% article-clustered bootstrap intervals (2,000 resamples):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Trivial (always non-propagandistic) | — | 0.000 | 0.000 |
| TF-IDF unigram + LR | 0.469 | 0.602 | 0.527 |
| TF-IDF char 3–5-gram + LR | 0.506 | 0.614 | 0.555 |
| **TF-IDF uni+bigram + LR (reported)** | **0.464** [.40–.54] | **0.606** [.54–.68] | **0.525** [.47–.58] |

Average precision 0.527; accuracy 0.665 (not the headline — a trivial always-negative classifier scores ~0.69 while detecting nothing). At a validation precision target of 0.50 the threshold applied to test gives precision 0.56 / recall 0.34, showing the baseline cannot hold high precision and useful recall at once.

A classification score is **not** a validated measure of "manipulation density," and a detector trained on corpus labels must not be treated as an arbiter of legitimate political opinion. These are limits on what the Politomorphism Engine can claim on this basis; the framework is not validated by this classifier.

### Transformer vs baseline (Working Paper II, v2)

On the same article-disjoint split, a fine-tuned DistilBERT was compared with the baseline, with per-paragraph scores saved. At default thresholds the transformer is more precise (0.578 vs 0.464) and flags fewer paragraphs (249 vs 345), with lower recall (0.545 vs 0.606); the **F1 difference at default thresholds is not significant** (ΔF1 +0.033, 95% CI [−0.03, +0.09]). But on threshold-independent / equal-cost comparisons the transformer is the better detector: average precision higher (ΔAP +0.074 [+0.00, +0.14]); recall at matched review budget higher (0.670 vs 0.606, Δ +0.068 [+0.01, +0.12]); and significantly more correct classifications (McNemar 148 vs 84, p < 0.001). This **reverses** Working Paper I/II v1's "context buys precision, not recall": at equal cost, context buys recall too. Paired transitions show the two models miss *different* cases, not the same ones.

**Multi-seed robustness:** the transformer was retrained across 5 random seeds (fixed split). AP = 0.605 ± 0.005; in **5/5 seeds** it beats the baseline on average precision, on recall at matched review budget (0.683 ± 0.009 vs 0.606), and on paired correctness (McNemar p ≤ 0.0001 each). The advantage is not a single-run fluke. Default-threshold recall is the most seed-sensitive quantity (0.49–0.62), which is why the threshold-independent and matched-cost measures carry the comparison.

## Files

- `01_download_data.py` → `02_inspect_labels.py` → `03_baseline_detector.py` → `04_transformer_detector.py` — the original pipeline (official split).
- `clean_rerun/` — the correction: `inspect_leakage.py` (quantifies the leakage), `run_baseline_clean.py` (article-disjoint split + baseline + bootstrap + matched precision), `quick_baselines.py` (alternative lexical baselines), `04_transformer_clean.py` (transformer on the clean split, saving scores), `compare_clean.py` (bootstrap CIs, McNemar, matched-budget comparison). Run these from the repository root (they read `data/`, created by `01_download_data.py`).
- Working papers I and II, in PDF and Word; `*_v2_corrected.*` supersede the originals.
- `requirements.txt` — pinned versions.

## Data

English configuration of **PropXplain** (Qatar Computing Research Institute). Not re-hosted here; `01_download_data.py` reproduces it from source. PropXplain retains its own license.

## Author & license

**Șerban Gabriel Florin** — Independent Researcher, Hannover, Germany · ORCID [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356) · Politomorphism Engine (Computational Political Anthropology).
Code: MIT. PropXplain: its own license.
