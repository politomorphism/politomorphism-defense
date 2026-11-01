# Politomorphism — Defensive Political Anthropology Layer

**Reproducible code for the working paper:** *Detecting Political Manipulation as a Defensive Practice: A Transparent Baseline for Explainable Propaganda Detection.*

This repository is the reproducible appendix to a working paper in Computational Political Anthropology. It trains a transparent baseline classifier that separates propagandistic from non-propagandistic news text, and positions that classifier as the **defensive audit layer** of the Politomorphism Engine.

> Manipulation detection is framed here as a *defensive* practice — detection, measurement, and explanation — not as the inverse of a generative one. The code only classifies and explains existing text; it never generates manipulative content.

## What's in this repository

- `01_download_data.py` — downloads and inspects the PropXplain (English) corpus from Hugging Face and saves local CSV copies.
- `02_inspect_labels.py` — reports row counts and class balance for each split.
- `03_baseline_detector.py` — trains a TF-IDF + logistic-regression baseline and evaluates it on the held-out test split.
- `requirements.txt` — the exact package versions used.

## Data

Experiments use the English configuration of **PropXplain** (Qatar Computing Research Institute), a multilingual, explanation-annotated corpus for propaganda detection. The dataset is **not re-hosted** in this repository; `01_download_data.py` reproduces it directly from the source. PropXplain retains its own license, held by its authors.

## Reproduce

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python 01_download_data.py
python 02_inspect_labels.py
python 03_baseline_detector.py
```

## Baseline results

On the held-out test split (n = 922; 72.9% non-propagandistic, 27.1% propagandistic):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non-propagandistic | 0.822 | 0.762 | 0.791 | 672 |
| propagandistic | 0.465 | 0.556 | 0.506 | 250 |

Overall accuracy is 0.706, but it is deliberately **not** the headline metric: a trivial classifier predicting the majority class everywhere would score 0.729 while detecting no manipulation at all. The precision, recall, and F1 on the *propagandistic* class are the defensively relevant figures.

## How to cite

```
Florin, Ș. G. (2026). Detecting Political Manipulation as a Defensive Practice:
A Transparent Baseline for Explainable Propaganda Detection. Working Paper,
Politomorphism Engine. ORCID: 0009-0000-2266-3356.
```

## Author

**Șerban Gabriel Florin** — Independent Researcher, Hannover, Germany
ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)
Project: Politomorphism Engine (Computational Political Anthropology)

## License

The **code** in this repository is released under the MIT License. The **PropXplain** corpus is governed by its own, separate license.