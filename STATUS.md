# Research status — where the project stands

**Snapshot date:** 2026-11-12 · **Author:** Șerban, G. F. · ORCID [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)

This file is a plain progress map: what exists as working code, what is reported
on author-built data, what is only proposed, and what is still pending. It is an
honest status, not a claim of validation. The code in this repository is the
**defensive detection and measurement layer** of Politomorphism; the seven engine
components (MC, CMG, SRM, EEF, PFM, ISC, LDE) live in the manuscripts, **not** in
this code.

## Status at a glance

| Part | Status |
|---|---|
| Leakage-controlled baseline vs transformer (article-disjoint, multi-seed) | **Done**, reported with bootstrap CIs (see README) |
| Contrastive-pairs benchmark (construct validity: structure vs vocabulary) | **Done** on author-built pairs (v3 dev, v4 held-out) |
| Detector v2: technique classifier + abstention | **Done**, reported on author-built data |
| Negation verifier (attributable, equivalence-only) | **Done**, reported on author-built data |
| Study 2 (controlled structure/negation diagnostic) | **Built & frozen**; not yet run by the author |
| Prevalence / utility and measurement-error correction (Rogan-Gladen) | **Done** as formulation + simulation; not empirically calibrated |
| **First external-label result — false_dilemma vs LOGIC (Jin et al. 2022)** | **Done (2026-11-07)** — F1 **0.895** [95% CI 0.855–0.929], precision 0.922, recall 0.869 on 137 independent-label positives; scope: fallacy-discrimination, not neutral-text false-alarm |
| External-evaluation tooling (inspect + guarded eval) | **Ready**; SemEval corpus is registration-walled, so LOGIC (openly hosted) was used for false_dilemma |
| Target-domain calibration (real r_k, f_k) | **Pending** — not demonstrated |
| Integrated dynamic engine (forecasting equation) | **Proposed** in pre-registration; no validated predictive performance |
| **Pre-registered structural-sensitivity test — false_dilemma on CoCoLoFa** (Yeh et al. 2024; pre-registered 2026-11-11, commit 1bf76c6; one run 2026-11-12) | **Done — verdict INCONCLUSIVE.** Hard-cell balanced accuracy **0.664** [95% CI 0.599–0.719] (rule: supported ≥ 0.75 with lower bound > 0.60; not supported ≤ 0.60 or lower bound ≤ 0.50). Per cell: FD with marker 0.936, FD **without** marker 0.545, non-FD with marker 0.782, non-FD without marker 0.818. Descriptive reading: the marker rarely triggers false alarms, but detection drops sharply when the marker is absent (same pattern as LOGIC, 98% vs 62%). Writer-clustered CI 0.464–0.755; supervised lexical control 0.877. Files: `benchmark/cocolofa_fd_predictions.csv` (no text), `benchmark/cocolofa_fd_report.txt` |
| Romanian-language arm | **Frozen** (2026-11-11) until the CoCoLoFa result is written up |
| **DFA component** (`dfa/`, added 2026-11-11) — consumer-protection practices named by the EU Digital Fairness Act proposal | **Frozen** (2026-11-11). Built; lexical rules baseline on author-built dev and held-out sets (dev 24/24 in-sample, held-out 0/13 — does not generalise); gpt-4o-mini stage 1: dev detection 23/24 but false alarms 18/26, held-out 10/13 and 7/15; article refs follow the leaked draft, not yet checked against the official COM text |

## Implementation map (component → script → input → output → test)

**Detector, stage 1** — `benchmark/16_classifier_v2_abstention.py`: a sentence →
one of six techniques / `none` / `insufficient_evidence`. Tested on
`benchmark/contrastive_pairs_v3.csv` and `_v4_heldout.csv`; metrics via
`benchmark/14_technique_metrics.py`.

**Detector, stage 2** — `benchmark/18_apply_negation_saved.py`: saved predictions
+ text → equivalence flags overturned only where the comparison is denied
(touches nothing outside `manufactured_equivalence`).

**Construct validity** — `benchmark/19_study2_diagnostic.py` on the frozen
`benchmark/study2_mechanism_families.csv` (provenance in
`benchmark/study2_freeze_record.json`) → two-stage log, critical-transition
table, pair / required-change accuracy (invariance reported N/A).

**Measurement → D_{k,t}** — `analysis/prevalence_utility.py` (precision vs
prevalence, break-even), `analysis/measurement_error_correction.py` (Rogan-Gladen:
recover true prevalence π from the flag rate given sensitivity r and
false-positive rate f), `analysis/measurement_error_simulation.py` (Monte Carlo
coverage/bias evidence). Output: D_{k,t} = π_{k,t} as an *uncertain* measured
input, not a point fact.

**External validation (English arm)** — `benchmark/22_inspect_semeval.py`
(data inspection: raw-offset span check, duplicate audit, split provenance) →
`benchmark/21_semeval_eval.py` (config-driven evaluation with guardrails:
justified mapping, fixed unit, verified-vs-uncertain negatives, two-stage logging,
per-category + macro/micro F1 with document-clustered bootstrap). Independent
labels are available for `false_dilemma` and `whataboutism` (clean) and partially
`implied_causation`; **not** for `manufactured_equivalence`, `presupposition`, or
`loaded_rhetorical_question`.

**Engine link** — the reduced equation
`L_{t+1} = α + φL_t + βR_t + γI_t + δ·D_{k,t} + η·D_{k,t}(1−I_t) + c′u_t + ε_t`
with the A/B/C out-of-sample comparison and rejection criteria is **proposed** in
the Section 7 pre-registration. It is not implemented with data.

## The honest boundary

Real code exists for detection, measurement, and the external-evaluation tooling.
What is **not** yet shown: target-domain calibration (so the prevalence correction
stays illustrative / author-label-conditioned), independent human validation of
labels, and any validated predictive performance of the integrated engine.

## Next step

One line only (decided 2026-11-11). Write up the pre-registered CoCoLoFa test as a
short working paper, reporting the INCONCLUSIVE verdict as pre-registered and the
marker asymmetry as a descriptive finding. No re-run or prompt change on the same
sample: that would be tuning on the test. After the write-up, issue the corrections
to earlier published papers once. Everything else (H1–H3, DFA, Romanian arm, engine
components, book) stays frozen until then. A multi-model benchmark on this protocol
would be a new study with its own pre-registration.
