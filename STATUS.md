# Research status — where the project stands

**Snapshot date:** 2026-11-06 · **Author:** Șerban, G. F. · ORCID [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)

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
| SemEval external-evaluation tooling | **Ready**; corpus not yet obtained |
| Target-domain calibration (real r_k, f_k) | **Pending** — not demonstrated |
| Integrated dynamic engine (forecasting equation) | **Proposed** in pre-registration; no validated predictive performance |
| Romanian-language arm | Set aside for now |

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

Obtain a public, independently annotated corpus to evaluate **one** category
(`false_dilemma` vs SemEval `Black-and-White Fallacy`) and decide
GO / CONDITIONAL-GO / NO-GO on real data. SemEval-2020 Task 11 appears to require
registration/request with the organisers; an openly downloadable alternative that
matches the definition is the fallback. Study 2 can run in parallel.
