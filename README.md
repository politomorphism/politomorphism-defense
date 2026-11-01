# Politomorphism — Defensive Political Anthropology Layer

**Reproducible code and working papers on detecting political manipulation as a defensive practice.**

This repository is the reproducible appendix to a two-part study in Computational Political Anthropology. It builds a detector that separates propagandistic from non-propagandistic news text and positions it as the **defensive audit layer** of the Politomorphism Engine. Part I establishes a transparent lexical baseline; Part II tests whether a contextual transformer recovers the structural manipulation the baseline misses.

> Manipulation detection is framed here as a *defensive* practice — detection, measurement, and explanation — not as the inverse of a generative one. The code only classifies and explains existing text; it never generates manipulative content.

## What's in this repository

- `01_download_data.py` — downloads and inspects the PropXplain (English) corpus from Hugging Face and saves local CSV copies.
- `02_inspect_labels.py` — reports row counts and class balance for each split.
- `03_baseline_detector.py` — trains a TF-IDF + logistic-regression baseline (Part I) and evaluates it on the held-out test split.
- `04_transformer_detector.py` — fine-tunes DistilBERT with a class-weighted objective (Part II) and evaluates it with the same protocol.
- `requirements.txt` — the exact package versions used.
- `Politomorphism_Defensive_Detection_WorkingPaper.*` — Working Paper I.
- `Politomorphism_Transformer_WorkingPaper_II.*` — Working Paper II.

## Data

Experiments use the English configuration of **PropXplain** (Qatar Computing Research Institute), a multilingual, explanation-annotated corpus for propaganda detection. The dataset is **not re-hosted** here; `01_download_data.py` reproduces it directly from the source. PropXplain retains its own license, held by its authors.

## Reproduce

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python 01_download_data.py        # fetch + cache the corpus
python 02_inspect_labels.py       # class balance
python 03_baseline_detector.py    # Part I: TF-IDF baseline
python 04_transformer_detector.py # Part II: DistilBERT (slow on CPU)
```

## Results

Held-out test split (n = 922; 72.9% non-propagandistic, 27.1% propagandistic). The propagandistic class is the defensively relevant one.

| Model | Precision | Recall | F1 | False positives | False negatives |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.465 | 0.556 | 0.506 | 160 | 111 |
| Transformer (DistilBERT) | 0.555 | 0.544 | 0.549 | 109 | 114 |

The transformer improves F1 modestly (0.506 → 0.549), but the gain is almost entirely in **precision** — it stops over-flagging, cutting false positives from 160 to 109 — while **recall is flat** and false negatives are essentially unchanged. Context buys precision, not recall; recovering the subtle, structural manipulation the baseline misses remains the open problem. Accuracy (0.706 → 0.758) is reported but is not the headline metric, since a majority-class classifier would already score ~0.729 while detecting no manipulation. Results are single fixed-seed runs; magnitudes should not be over-read.

## How to cite

```
Florin, Ș. G. (2026). Detecting Political Manipulation as a Defensive Practice:
A Transparent Baseline for Explainable Propaganda Detection. Working Paper I,
Politomorphism Engine. ORCID: 0009-0000-2266-3356.

Florin, Ș. G. (2026). Context Buys Precision, Not Recall: A Transformer Successor
to a Lexical Baseline for Propaganda Detection. Working Paper II,
Politomorphism Engine. ORCID: 0009-0000-2266-3356.
```

## Author

**Șerban Gabriel Florin** — Independent Researcher, Hannover, Germany
ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)
Project: Politomorphism Engine (Computational Political Anthropology)

## License

The **code** in this repository is released under the MIT License. The **PropXplain** corpus is governed by its own, separate license.
