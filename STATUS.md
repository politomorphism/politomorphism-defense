# Research status — where the project stands

**Snapshot date:** 2026-11-16 · **Author:** Șerban, G. F. · ORCID [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356)

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
| Contrastive-pairs benchmark (pair discrimination under approximate vocabulary matching) | **Done** on author-built pairs (v3 dev, v4 held-out). Does **not** isolate structure from vocabulary: residual lexical cues are not excluded (corrected 2026-11-16, see erratum) |
| Detector v2: technique classifier + abstention | **Done**, reported on author-built data |
| Negation verifier (attributable, equivalence-only) | **Done**, reported on author-built data |
| Study 2 (controlled structure/negation diagnostic) | **Built & frozen**; not yet run by the author |
| Prevalence / utility and measurement-error correction (Rogan-Gladen) | **Done** as formulation + simulation; not empirically calibrated |
| **First external-label result — false_dilemma vs LOGIC (Jin et al. 2022)** | **Done (2026-11-07)** — F1 **0.895** [95% CI 0.855–0.929], precision 0.922, recall 0.869 on 137 independent-label positives; scope: fallacy-discrimination, not neutral-text false-alarm |
| External-evaluation tooling (inspect + guarded eval) | **Ready**; SemEval corpus is registration-walled, so LOGIC (openly hosted) was used for false_dilemma |
| Target-domain calibration (real r_k, f_k) | **Pending** — not demonstrated |
| Integrated dynamic engine (forecasting equation) | **Proposed** in pre-registration; no validated predictive performance |
| **Pre-registered structural-sensitivity test — false_dilemma on CoCoLoFa** (Yeh et al. 2024; pre-registered 2026-11-11, commit 1bf76c6; one run 2026-11-12) | **Done — verdict INCONCLUSIVE.** Hard-cell balanced accuracy **0.664** [95% CI 0.599–0.719] (rule: supported ≥ 0.75 with lower bound > 0.60; not supported ≤ 0.60 or lower bound ≤ 0.50). Per cell: FD with marker 0.936, FD **without** marker 0.545, non-FD with marker 0.782, non-FD without marker 0.818. Descriptive reading: the marker rarely triggers false alarms, but detection drops sharply when the marker is absent (same pattern as LOGIC, 98% vs 62%). Writer-clustered CI 0.464–0.755; supervised lexical control 0.877. Files: `benchmark/cocolofa_fd_predictions.csv` (no text), `benchmark/cocolofa_fd_report.txt` Written up 2026-11-12 as the working paper *Marker Dependence in Zero-Shot False-Dilemma Detection: A Pre-registered Test on Independent Data*: `Marker_Dependence_FalseDilemma_CoCoLoFa_EN.*`, Romanian version `Dependenta_de_Marker_Falsa_Dilema_CoCoLoFa_RO.*` |
| Romanian-language arm | **Frozen** (2026-11-11); stays frozen until the author decides the next direction |
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

## Correction round (issued 2026-11-16)

The single correction round after the CoCoLoFa test is issued. Working Paper III
(English and Romanian) and the overview article are reissued as Version 2 with a
correction note; the claim that the programme has shown a model responding to
structure rather than vocabulary is withdrawn wherever it appears. Everything else
is covered by `Erratum_CoCoLoFa_Correction_Round_EN_RO.*`, with the full
sentence-by-sentence list in `Corecturi_lucrari_dupa_CoCoLoFa.docx`. No score,
sample, prompt, marker or decision rule changed.

## Next step

Decided by the author on 2026-11-18: the multi-model run of the CoCoLoFa protocol.
Its pre-registration is frozen as version 10 in `prereg/`
(`Explicit-Implicit_Gap_Across_Models_Pre-registration_v10_FROZEN.*`), with the SHA-256
of every frozen element in `prereg/FREEZE_MANIFEST.txt`. Primary outcome: writer-equal
hard-cell balanced accuracy; secondary: the writer-equal explicit–implicit gap G (EIADG)
and S2–S10. Arms: gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemma-4-31b-it, all on
the Gemini API free tier with thinking at minimal; gpt-4o-mini is context only.
Launcher `benchmark/33_cocolofa_multimodel_run.py`, analysis
`benchmark/34_cocolofa_multimodel_analysis.py`. Re-analysed with equal weight per
writer, the published gpt-4o-mini predictions give 0.556 (writer interval 0.464–0.652)
and G = 0.534; the published comment-weighted results and verdict stand as published.
The other follow-up, independent human labels for the marker-free cell, stays
deferred (Appendix A of the pre-registration; its annotation manifest is drawn).
H1–H3, DFA, the Romanian arm, the engine components and the book stay frozen.
