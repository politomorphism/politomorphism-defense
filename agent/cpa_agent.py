#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defense Audit Agent for Politomorphism
=======================================
Author of the framework: Șerban, G. F. (ORCID 0009-0000-2266-3356)

A multi-agent verification assistant that runs on the OpenAI Agents API
(OpenAI-hosted sandbox). It works on one direction only: the defensive layer
of Politomorphism (github.com/politomorphism/politomorphism-defense), which
measures how well language models detect implicit political manipulation.
It does three jobs:

  1. reproduces reported numbers from the saved prediction files,
  2. audits pre-registered tests (freeze hashes, decision rule, run record),
  3. checks the claims in a paper against the evidence, and prepares the
     list of corrections to earlier papers.

It never re-runs a detector on a frozen test sample and never changes the
repository; it only reads, recomputes and reports.

Usage (from the folder that contains this file):

    python cpa_agent.py                 start or continue a conversation
    python cpa_agent.py --new           discard the saved session, start fresh
    python cpa_agent.py --dry-run       show what would be sent, call nothing
    python cpa_agent.py --ask "..."     ask one question and exit

Folders created next to this file:

    knowledge/   optional: put papers here that are not in the repository;
                 they are uploaded when a NEW session starts
    outputs/     reports the agent writes (.docx) are downloaded here

Settings (environment variables, all optional except the API key):

    OPENAI_API_KEY          your OpenAI API key (required)
    CPA_MODEL               model name                (default: gpt-6-astra)
    CPA_REASONING_EFFORT    low | medium | high       (default: high)
    CPA_SPEND_LIMIT_CENTS   hard budget per session   (default: 500 = 5 USD)
    CPA_MAX_SUBAGENTS       parallel subagents        (default: 3)
    CPA_CONTAINER_SIZE      small | medium | large    (default: medium)
    CPA_REPOS               comma-separated public git URLs to clone into
                            /workspace/repos
                            (default: the politomorphism-defense repository)

Requires: Python 3.9+ and  pip install --upgrade openai
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
KNOWLEDGE_DIR = HERE / "knowledge"
OUTPUTS_DIR = HERE / "outputs"
SESSION_FILE = HERE / ".cpa_session"

MODEL = os.environ.get("CPA_MODEL", "gpt-6-astra")
REASONING_EFFORT = os.environ.get("CPA_REASONING_EFFORT", "high")
SPEND_LIMIT_CENTS = int(os.environ.get("CPA_SPEND_LIMIT_CENTS", "500"))
MAX_SUBAGENTS = int(os.environ.get("CPA_MAX_SUBAGENTS", "3"))
CONTAINER_SIZE = os.environ.get("CPA_CONTAINER_SIZE", "medium")
DEFAULT_REPO = "https://github.com/politomorphism/politomorphism-defense.git"
REPOS = [r.strip() for r in os.environ.get("CPA_REPOS", DEFAULT_REPO).split(",") if r.strip()]

# Upload limits for the knowledge/ folder (conservative; inline base64 upload).
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {
    ".pdf", ".docx", ".txt", ".md", ".tex", ".bib",
    ".py", ".r", ".ipynb", ".json", ".yaml", ".yml", ".toml",
    ".csv", ".tsv",
}

SANDBOX_KNOWLEDGE = "/workspace/knowledge"
SANDBOX_SKILLS = "/workspace/capabilities/skills"
SANDBOX_OUTPUTS = "/workspace/outputs"

PYTHON_PACKAGES = [
    "numpy", "scipy", "pandas", "statsmodels", "scikit-learn",
    "pypdf", "python-docx", "openpyxl",
]

# --------------------------------------------------------------------------
# Agent instructions
# --------------------------------------------------------------------------

INSTRUCTIONS = """
You are the Defense Audit Agent: a verification and documentation assistant for
the defensive layer of Politomorphism, a Computational Political Anthropology
(CPA) programme by the independent researcher Șerban, G. F.
(ORCID 0009-0000-2266-3356). Always cite the author as "Șerban, G. F.".

THE ONE DIRECTION
The programme follows a single line of work: measuring how well language models
detect political manipulation when it is IMPLICIT, as against explicit - the
gap first reported in "Marker Dependence in Zero-Shot False-Dilemma Detection:
A Pre-registered Test on Independent Data" (Șerban, G. F., 12 November 2026).
Everything you do serves that line. The seven engine components (MC, CMG, SRM,
EEF, PFM, ISC, LDE), the DFA component, the Romanian arm and the book are
frozen: do not work on them and do not propose work on them. If the user asks
about them, answer briefly and say they are frozen by his own decision.

YOUR ROLE
You verify, audit and document work that already exists in the repository
github.com/politomorphism/politomorphism-defense and in the author's papers.
You do not propose new studies, components or directions unless the user
explicitly asks. Your value is that you are stricter than the author would be
with himself.

WHERE THINGS ARE
- /workspace/repos/politomorphism-defense   the cloned public repository
- /workspace/knowledge      papers and files the author uploaded (may be empty)
- /workspace/capabilities/skills   your four skills; read the relevant SKILL.md
                            before starting a task of that kind
- /workspace/outputs        the only place where deliverables go

HARD LIMITS (never cross these)
1. Never run a detector, a model or any tuned procedure on a frozen
   confirmatory sample again, and never change a prompt, marker, sample or
   threshold of a pre-registered test. That would be tuning on the test set.
   You may re-SCORE saved prediction files as often as you like.
2. Never modify, commit to or push to the repository. Work on your clone,
   write only to /workspace/outputs.
3. Corpus texts without a redistribution licence (CoCoLoFa states none) may be
   downloaded from their original source for checking inside the sandbox, but
   must never be copied into /workspace/outputs or quoted at length.
4. This is defensive work: you classify, measure and explain existing text.
   Never generate manipulative or propagandistic text, not even as test items.

EVIDENCE DISCIPLINE
1. Never invent a number, a dataset, a DOI, a commit hash or a result. If it is
   not in the repository, in an uploaded file, or in a source you retrieved and
   can name, say that it is missing and stop there.
2. Every factual statement carries its source: file name plus line, section or
   commit hash.
3. A number is "verified" only if you recomputed it in the sandbox and show the
   command and the output. Otherwise it is "reported by the author".
4. Keep confirmatory and exploratory results apart. A pre-registered test is
   judged only by its registered decision rule; anything examined after the run
   is labelled exploratory.
5. Inconclusive, null and negative results stay exactly as reported. Never
   soften them, reframe them as partial support, or leave them out of a summary.
6. When the text of a paper and the code or data disagree, report both and do
   not choose silently.
7. If you are unsure, say so and say what would settle it.

HOW TO WORK
- Start by looking at what is actually in the repository (git log, STATUS.md,
  README.md, benchmark/) and in /workspace/knowledge. The repository is the
  source of truth; your skills are only a map.
- For a task with several kinds of checking, delegate to subagents with these
  roles and merge their findings yourself:
    * Reproduction Checker      re-scores saved predictions, recomputes metrics
                                and intervals, checks splits for leakage
    * Pre-registration Auditor  checks freeze hashes, commit history, the
                                decision rule and the run record
    * Claims Auditor            tests each claim in a paper against its evidence
  Give every subagent the hard limits and the evidence rules above. Subagents
  report findings; only you write the final answer.
- Use web search only to locate primary sources (dataset papers, codebooks,
  licences, model documentation). Prefer the primary source and give the URL.
- Disagree with the author when the evidence requires it, plainly and
  specifically.

OUTPUT
- Answer in the language the user writes in, in plain words; explain a
  technical term in one sentence the first time you use it. Reports meant for
  publication are written in English unless the user says otherwise.
- Write in connected paragraphs. Use a table only for genuinely tabular
  material.
- Any deliverable longer than a short answer is saved as a .docx file in
  /workspace/outputs (use python-docx), never as .md. Say which file you wrote.
- When you give code, give the complete runnable file, never a fragment.
- End every audit with a short verdict: what is verified, what is only
  reported, what could not be checked and why, what contradicts what.
""".strip()

# --------------------------------------------------------------------------
# Skills (written into the sandbox as SKILL.md files)
# --------------------------------------------------------------------------

SKILL_MAP = """---
name: defense-programme-map
description: Map of the Politomorphism defensive layer - repository layout, corpora, studies done and their recorded results. Read before any task about the programme.
---

# The defensive layer of Politomorphism

Author: Șerban, G. F. Repository: github.com/politomorphism/politomorphism-defense
(code MIT). Cloned at /workspace/repos/politomorphism-defense.

This file is an orientation map written on 15 November 2026 from the
repository as it stood at commit 3ecf5ea and from the published paper. It is
NOT a source. Read STATUS.md, README.md and `git log` first; if they disagree
with this map, the repository wins and you report the discrepancy.

## What the code is

A detector of rhetorical manipulation built as a zero-shot prompt to a language
model. Stage 1 labels a text as one of six techniques (false_dilemma,
whataboutism, implied_causation, manufactured_equivalence, presupposition,
loaded_rhetorical_question), or none, or abstains (insufficient_evidence).
Stage 2 is a negation verifier that only touches manufactured_equivalence.
Scripts are numbered in the repository root (01-04, the PropXplain baseline and
transformer) and in benchmark/ (the detector, probes and external tests);
analysis/ holds prevalence and measurement-error code.

## Recorded results (reported by the author until you recompute them)

- Leakage-controlled baseline vs transformer on PropXplain, article-disjoint
  split, corrected to version 2 after an audit found article-level leakage and
  an over-stated error-analysis claim, which was retracted. Published on Zenodo,
  DOI 10.5281/zenodo.23884579.
- Contrastive-pairs benchmark (Working Paper III): vocabulary-matched pairs
  written by the author; 60 pairs; rated by an LLM, not yet by humans.
- First external-label result, false_dilemma on LOGIC (Jin et al., 2022),
  7 November 2026: F1 0.895, precision 0.922, recall 0.869 on 137 positives.
  A bag-of-words model reaches 0.852 on the same items. Negatives are other
  fallacies, not neutral text.
- LOGIC hard-cell re-analysis (script 26): balanced accuracy 0.704, interval
  0.568-0.830; 98% detection with a marker, 62% without. A re-analysis plan,
  not an independent pre-registration.
- Two author-written 2x2 structural probes failed as instruments because a
  lexical model separated their hard cells (0.786 and 0.893).
- Pre-registered structural-sensitivity test on CoCoLoFa (Yeh, Wan and Huang,
  EMNLP 2024). Pre-registration commit 1bf76c6 (11 November 2026), one run and
  report in commit 3ecf5ea (12 November 2026). Sample: 440 comments, 49
  writers, four cells of 110. Per cell correct: A 103, B 60, C 86, D 90.
  Primary outcome, hard-cell balanced accuracy: 0.664, item-level 95% interval
  0.599-0.719. Rule: SUPPORTED if >= 0.75 and lower bound > 0.60; NOT SUPPORTED
  if <= 0.60 or lower bound <= 0.50; otherwise INCONCLUSIVE. Verdict:
  INCONCLUSIVE. Writer-clustered interval 0.464-0.755. Supervised lexical
  control on hard cells 0.877. Exploratory: one writer produced 53 of the 110
  cell-B comments; detection 41/53 on that writer, 19/57 on the other 21.
  Published as "Marker Dependence in Zero-Shot False-Dilemma Detection: A
  Pre-registered Test on Independent Data" (English and Romanian).
- DFA component (dfa/): frozen; the lexical baseline does not generalise to
  held-out items.

## Standing consequences

- The claim that the detector "tracks structure, not vocabulary" is NOT
  supported in its strong form and is not refuted either. The supported
  statement is narrower: the detector resists the marker as a false cue, while
  its detection of false dilemma depends substantially on the device being
  explicit.
- No re-run or prompt change on the CoCoLoFa sample. A multi-model run or human
  annotation of the marker-free cell would each be a new study with its own
  pre-registration.
- Corrections to earlier published papers are to be issued once.

## Corpora and their constraints

- PropXplain (QCRI): own licence, not re-hosted; reproduced by 01_download_data.py.
- LOGIC: openly hosted; possible exposure of the model in training is unresolved.
- CoCoLoFa: github.com/Crowd-AI-Lab/cocolofa, pinned at commit c39d45f; the
  data files state no licence, so texts are not redistributed - only ids,
  labels and predictions are published. Labels are the fallacy each writer was
  ASSIGNED to produce, not a reader's judgement.
"""

SKILL_REPRO = """---
name: reproduction-check
description: Procedure for reproducing a reported result of the defensive layer from the repository - re-scoring saved predictions, recomputing metrics and intervals, checking splits for leakage. Use when asked to verify, reproduce or sanity-check a number.
---

# Reproduction check

1. Locate the number in the paper or report and the script and input file that
   produced it. Quote file names and lines. Record `git rev-parse HEAD`.
2. Decide what can be recomputed without model calls. Saved prediction files
   (for example benchmark/cocolofa_fd_predictions.csv) allow full re-scoring.
   Anything that needs comment texts needs the corpus from its original source.
   Anything that needs a new model call on a frozen confirmatory sample is
   FORBIDDEN; say so and stop that branch.
3. Re-score from the saved predictions with your own short, complete script in
   /workspace/outputs/verification/, written from the definition in the paper
   and not copied from the author's code. Then run the author's functions on
   the same input (import them, do not retype them) and compare.
4. Metric definitions to check explicitly: what counts as correct; how an
   abstention is scored; whether accuracy is balanced; which class is positive;
   which F1 convention is used.
5. Intervals: reproduce the stated bootstrap exactly (unit of resampling,
   number of replicates, seed). Then say whether the unit of resampling matches
   the dependence in the data. Comments by one writer, sentences from one
   article and items from one source are not independent; if a clustered
   interval is available or computable, report it beside the item-level one and
   say what the decision rule would return under each.
6. Leakage: for any supervised model, check that train and test are disjoint at
   the level of the source document and of the writer, not only of the item.
   Check for duplicates and near-duplicates across the split.
7. Concentration: tabulate items per writer, per article and per source in each
   cell or class. If one unit supplies a large share, report the metric with
   and without it, labelled exploratory.
8. Compare three things: your recomputation, the author's code, the published
   text. State the tolerance.

Verdict vocabulary: VERIFIED, REPRODUCED WITH DIFFERENCES (give them),
NOT REPRODUCIBLE (say what is missing), INCONSISTENT (text versus code or data).
Show the exact commands and their output.
"""

SKILL_PREREG = """---
name: preregistration-audit
description: Procedure for checking that a pre-registered test was run as registered - freeze hashes, commit history, decision rule, run record, deviations. Use for any confirmatory test in the repository and before drafting a new pre-registration.
---

# Pre-registration audit

1. Find the pre-registration document and the commit that froze it. Record the
   commit hash and its date. Find the commit that reported the result.
2. Hashes: recompute SHA-256 for every file listed in the freeze record (for
   example benchmark/cocolofa_freeze.sha256) and compare. Recompute the hash of
   the predictions file and compare with the one printed in the report.
3. History: run `git diff --stat <freeze commit> <result commit>` and
   `git log --follow` on each frozen file. Any change to a frozen script,
   sample, prompt, marker or threshold after the freeze is a deviation,
   whatever its size.
4. Order of events: show from commit dates, file contents and the run record
   that no detector output for the test sample existed before the freeze. If
   this cannot be shown from the repository, say so.
5. Sample: check that the published manifest matches the predictions file row
   for row (ids, cells, labels), and that cell sizes match the registration.
6. Decision rule: copy the rule verbatim from the registration, apply it
   yourself to the recomputed primary outcome and interval, and state the
   verdict. Check boundary cases carefully (>= versus >, which interval).
7. Analyses: list every analysis in the paper and classify it as registered
   primary, registered secondary, or exploratory. An analysis that was not in
   the registration and is not labelled exploratory in the paper is a finding.
8. Run record: one run only. Look for evidence of more than one model run
   (several prediction files, timestamps, retries). Failed launches that
   produced no model output are not runs but must be disclosed.
9. What was known before the freeze: check the registration's own disclosure
   against the repository history.

When asked to help draft a NEW pre-registration, require before any data is
touched: one primary outcome, one decision rule with numeric thresholds, a
fixed sample or sampling rule with seed, the unit of resampling for the
interval justified by the dependence in the data, the list of secondary
analyses, the freeze hashes, and a statement of what each verdict would and
would not show.

Deliverable: a .docx audit with a table (item, registered, observed, match)
and a paragraph on each deviation. Verdict: ADHERED, ADHERED WITH DISCLOSED
DEVIATIONS, or DEVIATED.
"""

SKILL_CLAIMS = """---
name: claims-audit
description: Procedure for auditing the claims of a working paper against its evidence, and for preparing the single round of corrections to earlier papers. Use when asked to review a manuscript or to list what must be corrected.
---

# Claims audit

1. Extract every empirical claim from the text as a numbered list with its
   location. Include the title, the abstract and the conclusion; over-statement
   concentrates there.
2. For each claim identify the evidence offered: table, test, dataset, script.
   A claim with no evidence in the text is UNSUPPORTED.
3. Grade the match:
   - SUPPORTED     the evidence shows what the claim says, at that strength
   - OVERSTATED    the evidence shows something weaker or narrower
   - UNSUPPORTED   no evidence offered
   - CONTRADICTED  evidence in this or another document by the same author
                   says otherwise
4. Look specifically for:
   - "structure, not vocabulary", "reads structure", "structural" or
     "construct validity" language resting on agreement scores that a lexical
     model also reaches
   - author-written test items presented as independent evidence
   - an LLM rater presented as independent human annotation
   - results on other-fallacy negatives presented as a false-alarm rate on
     neutral text
   - exploratory findings in confirmatory wording; an INCONCLUSIVE verdict
     described as support or as refutation
   - item-level intervals where items are clustered by writer or article
   - "validated", "proves", "confirms", "demonstrates" - check each use
   - one model, one run, one language generalised to "language models"
   - an inconclusive or negative result elsewhere in the programme that bears
     on the claim and is not mentioned
5. For each claim that is not SUPPORTED, propose the narrowest rewording the
   evidence does support. Keep the author's voice; change only strength and
   scope.
6. Check the references: each cited work exists, says what it is cited for,
   has complete details (pages, DOI where one exists), and the author is cited
   as "Șerban, G. F.".

## Correction round for earlier papers

When asked for the correction list, go through every paper in the repository
(the .docx files in the root and in benchmark/) and README.md, find each
sentence affected by the CoCoLoFa result or by an earlier retraction, and give
for each: file, location, the sentence as it stands, why it no longer holds,
and the corrected sentence. Do not rewrite anything that is still supported.
The corrections are issued once, so the list must be complete; say explicitly
which files you read in full and which you could not open.

Deliverable: a .docx report with a claim table (number, location, claim,
evidence, grade) and the suggested rewordings. Close with what a sceptical
reviewer would attack first.
"""

SKILLS: Dict[str, str] = {
    "defense-programme-map": SKILL_MAP,
    "reproduction-check": SKILL_REPRO,
    "preregistration-audit": SKILL_PREREG,
    "claims-audit": SKILL_CLAIMS,
}

# --------------------------------------------------------------------------
# Building the request
# --------------------------------------------------------------------------


def b64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def collect_knowledge() -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """Return (file entries for the API, uploaded names, skipped notes)."""
    entries: List[Dict[str, str]] = []
    uploaded: List[str] = []
    skipped: List[str] = []
    total = 0

    if not KNOWLEDGE_DIR.exists():
        return entries, uploaded, skipped

    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = path.relative_to(KNOWLEDGE_DIR).as_posix()
        size = path.stat().st_size
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            skipped.append(f"{rel} (file type not uploaded)")
            continue
        if size == 0:
            skipped.append(f"{rel} (empty)")
            continue
        if size > MAX_FILE_BYTES:
            skipped.append(f"{rel} (larger than {MAX_FILE_BYTES // (1024 * 1024)} MB)")
            continue
        if total + size > MAX_TOTAL_BYTES:
            skipped.append(f"{rel} (total upload limit of {MAX_TOTAL_BYTES // (1024 * 1024)} MB reached)")
            continue
        total += size
        entries.append(
            {
                "type": "inline",
                "path": f"{SANDBOX_KNOWLEDGE}/{rel}",
                "data": b64_file(path),
            }
        )
        uploaded.append(rel)

    return entries, uploaded, skipped


def skill_files() -> List[Dict[str, str]]:
    return [
        {
            "type": "inline",
            "path": f"{SANDBOX_SKILLS}/{name}/SKILL.md",
            "data": b64_text(body),
        }
        for name, body in SKILLS.items()
    ]


def build_agent() -> Dict[str, Any]:
    return {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "reasoning": {"effort": REASONING_EFFORT},
        "tools": [{"type": "web_search"}],
        "multi_agent": {
            "enabled": True,
            "max_concurrent_subagents": MAX_SUBAGENTS,
        },
    }


def build_environment(knowledge_entries: List[Dict[str, str]]) -> Dict[str, Any]:
    setup_commands: List[Dict[str, str]] = [
        {"command": f"mkdir -p {SANDBOX_OUTPUTS} {SANDBOX_KNOWLEDGE} /workspace/repos"},
    ]
    for url in REPOS:
        setup_commands.append(
            {"command": f"git clone {url}", "cwd": "/workspace/repos"}
        )

    packages: Dict[str, List[str]] = {"python": PYTHON_PACKAGES}
    if REPOS:
        packages["system"] = ["git"]

    return {
        "type": "openai_hosted",
        "container_size": CONTAINER_SIZE,
        "network": {"access": "enabled"},
        "packages": packages,
        "files": skill_files() + knowledge_entries,
        "capability_directories": [SANDBOX_SKILLS],
        "setup_commands": setup_commands,
    }


# --------------------------------------------------------------------------
# Streaming and event handling
# --------------------------------------------------------------------------


class TurnPrinter:
    """Prints assistant text as it arrives and keeps the final text."""

    def __init__(self) -> None:
        self.parts: Dict[Tuple[Any, Any], str] = {}
        self.streamed: Set[Tuple[Any, Any]] = set()
        self.status_shown: Set[str] = set()

    def handle(self, event: Any) -> None:
        etype = getattr(event, "type", "")
        if etype == "agent.session.turn.output_text.delta":
            key = (getattr(event, "item_id", None), getattr(event, "content_index", None))
            delta = getattr(event, "delta", "") or ""
            self.parts[key] = self.parts.get(key, "") + delta
            self.streamed.add(key)
            sys.stdout.write(delta)
            sys.stdout.flush()
        elif etype == "agent.session.turn.output_text.done":
            key = (getattr(event, "item_id", None), getattr(event, "content_index", None))
            text = getattr(event, "text", "") or ""
            if key not in self.streamed:
                sys.stdout.write(text)
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.parts[key] = text
        elif etype.startswith("agent.session.environment."):
            label = etype.rsplit(".", 1)[-1]
            if label not in self.status_shown:
                self.status_shown.add(label)
                print(f"[sandbox: {label}]", flush=True)

    @property
    def text(self) -> str:
        return "".join(self.parts.values())


def check_terminal(event: Any) -> bool:
    """Return True when the root agent's turn has completed. Raise on failure."""
    etype = getattr(event, "type", "")
    if etype == "error":
        err = getattr(event, "error", None)
        raise RuntimeError(getattr(err, "message", "unknown stream error"))
    if etype in ("agent.session.failed", "agent.session.environment.failed"):
        raise RuntimeError(f"Agent lifecycle failure: {etype}")
    if etype in (
        "agent.session.turn.completed",
        "agent.session.turn.failed",
        "agent.session.turn.cancelled",
    ):
        turn = getattr(event, "turn", None)
        if getattr(turn, "subagent_id", None) is not None:
            return False  # a subagent finished; the root turn continues
        if etype == "agent.session.turn.completed":
            return True
        if etype == "agent.session.turn.failed":
            err = getattr(turn, "error", None)
            raise RuntimeError(f"The turn failed: {getattr(err, 'message', '') or 'no detail given'}")
        raise RuntimeError("The turn was cancelled")
    return False


# Optional settings the API may not accept on every account. If the API answers
# "Unknown parameter" for one of these, it is removed and the request is sent
# again. Anything not in this list is required and is never removed.
OPTIONAL_PARAMETERS = (
    "spend_control",
    "metadata",
    "agent.reasoning",
    "agent.multi_agent",
    "environment.container_size",
    "environment.capability_directories",
)

_UNKNOWN_PARAM = re.compile(r"Unknown parameter:\s*'([^']+)'")


def _drop_parameter(request: Dict[str, Any], dotted: str) -> bool:
    """Remove a dotted parameter path from the request. Return True if removed."""
    parts = dotted.split(".")
    node: Any = request
    for key in parts[:-1]:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    if isinstance(node, dict) and parts[-1] in node:
        del node[parts[-1]]
        return True
    return False


def _optional_match(name: str) -> Optional[str]:
    """Map the parameter named in an API error to an optional parameter path."""
    for path in OPTIONAL_PARAMETERS:
        if name == path or name.startswith(path + ".") or name.startswith(path + "["):
            return path
    return None


def open_session_stream(client: Any, request: Dict[str, Any]) -> Any:
    """Create the session, removing optional parameters the API rejects."""
    import openai

    while True:
        try:
            return client.beta.agents.sessions.create(**request)
        except openai.BadRequestError as exc:
            found = _UNKNOWN_PARAM.search(str(exc))
            path = _optional_match(found.group(1)) if found else None
            if path is None or not _drop_parameter(request, path):
                raise
            if path == "spend_control":
                print(
                    "[note] Your OpenAI account does not accept a per-session spending "
                    "limit, so none is applied. Set a monthly limit on your account at "
                    "platform.openai.com/settings/organization/limits"
                )
            else:
                print(f"[note] The API does not accept '{path}'; continuing without it.")


def first_turn(client: Any, question: str, knowledge_entries: List[Dict[str, str]]) -> str:
    """Create the session, stream the first turn, return the session id."""
    printer = TurnPrinter()
    request: Dict[str, Any] = {
        "agent": build_agent(),
        "environment": build_environment(knowledge_entries),
        "input": question,
        "spend_control": {"limit": SPEND_LIMIT_CENTS},
        "metadata": {"project": "politomorphism", "agent": "defense-audit-agent"},
        "stream": True,
    }
    stream = open_session_stream(client, request)
    with stream.with_result_collection() as events:
        for event in events:
            printer.handle(event)
            check_terminal(event)
        result = events.get_final_result()
    return result.session_id


def next_turn(client: Any, session_id: str, question: str) -> None:
    """Send a follow-up message to an existing session and stream the reply."""
    printer = TurnPrinter()
    # Open the stream before sending, so early events are not missed.
    with client.beta.agents.sessions.events.stream(session_id) as events:
        client.beta.agents.sessions.events.create(
            session_id,
            idempotency_key=str(uuid.uuid4()),
            events=[
                {
                    "type": "agent.session.input.message",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": question}],
                        }
                    ],
                }
            ],
        )
        for event in events:
            printer.handle(event)
            if check_terminal(event):
                return
    raise RuntimeError("The stream closed before the turn ended.")


# --------------------------------------------------------------------------
# Artifacts (files the agent wrote to /workspace/outputs)
# --------------------------------------------------------------------------


def download_new_artifacts(client: Any, session_id: str, seen: Set[str]) -> List[Path]:
    saved: List[Path] = []
    try:
        page = client.beta.agents.sessions.artifacts.list(session_id, order="asc", limit=100)
        artifacts = list(page)
    except Exception as exc:  # listing must never break the conversation
        print(f"[could not list output files: {exc}]")
        return saved

    for art in artifacts:
        if art.id in seen:
            continue
        seen.add(art.id)
        rel = art.path
        if rel.startswith(SANDBOX_OUTPUTS):
            rel = rel[len(SANDBOX_OUTPUTS):]
        rel = rel.lstrip("/") or art.id
        target = (OUTPUTS_DIR / rel).resolve()
        if OUTPUTS_DIR.resolve() not in target.parents:
            target = OUTPUTS_DIR / Path(rel).name  # never write outside outputs/
        try:
            content = client.beta.agents.sessions.artifacts.content(art.id, session_id=session_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.content)
            saved.append(target)
        except Exception as exc:
            print(f"[could not download {art.path}: {exc}]")
    return saved


def known_artifact_ids(client: Any, session_id: str) -> Set[str]:
    try:
        return {a.id for a in client.beta.agents.sessions.artifacts.list(session_id, limit=100)}
    except Exception:
        return set()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def print_dry_run(question: str) -> None:
    entries, uploaded, skipped = collect_knowledge()
    env = build_environment(entries)
    agent = build_agent()
    print("DRY RUN - nothing is sent to OpenAI\n")
    print(f"model              : {agent['model']}")
    print(f"reasoning effort   : {agent['reasoning']['effort']}")
    print(f"tools              : {[t['type'] for t in agent['tools']]}")
    print(f"subagents          : up to {agent['multi_agent']['max_concurrent_subagents']} in parallel")
    print(f"spend limit        : {SPEND_LIMIT_CENTS} cents per session")
    print(f"container          : {env['container_size']}, network {env['network']['access']}")
    print(f"python packages    : {', '.join(env['packages']['python'])}")
    print(f"repositories       : {', '.join(REPOS) if REPOS else 'none'}")
    print(f"instructions       : {len(INSTRUCTIONS)} characters")
    print(f"skills             : {', '.join(SKILLS)}")
    print(f"knowledge uploaded : {len(uploaded)} file(s)")
    for name in uploaded:
        print(f"    + {name}")
    for note in skipped:
        print(f"    - skipped {note}")
    print(f"first question     : {question!r}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows consoles
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Defense Audit Agent for Politomorphism")
    parser.add_argument("--new", action="store_true", help="start a fresh session")
    parser.add_argument("--dry-run", action="store_true", help="show the request, call nothing")
    parser.add_argument("--ask", metavar="TEXT", help="ask one question and exit")
    args = parser.parse_args()

    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    default_question = (
        "Read STATUS.md, README.md and the git log of the politomorphism-defense "
        "repository and tell me, in plain words, what is done, what is verified "
        "by you, and what the single next step is."
    )

    if args.dry_run:
        print_dry_run(args.ask or default_question)
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it, then run this file again.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("The openai package is missing. Run:  pip install --upgrade openai")
        return 1

    client = OpenAI()
    if not hasattr(client.beta, "agents"):
        print("This openai package is too old for the Agents API. Run:  pip install --upgrade openai")
        return 1

    if args.new and SESSION_FILE.exists():
        SESSION_FILE.unlink()

    session_id: Optional[str] = None
    if SESSION_FILE.exists():
        session_id = SESSION_FILE.read_text(encoding="utf-8").strip() or None

    seen: Set[str] = known_artifact_ids(client, session_id) if session_id else set()

    print("Defense Audit Agent - Politomorphism")
    if session_id:
        print(f"Continuing session {session_id}  (python cpa_agent.py --new starts fresh)")
    else:
        _, uploaded, skipped = collect_knowledge()
        print(f"New session. Uploading {len(uploaded)} file(s) from the knowledge folder.")
        for note in skipped:
            print(f"  skipped {note}")
        print(f"  Repository to clone: {', '.join(REPOS) if REPOS else 'none'}")
    print("Type your question and press Enter. Type exit to stop.\n")

    def run_turn(question: str) -> None:
        nonlocal session_id
        if session_id is None:
            entries, _, _ = collect_knowledge()
            session_id = first_turn(client, question, entries)
            SESSION_FILE.write_text(session_id, encoding="utf-8")
        else:
            next_turn(client, session_id, question)
        for path in download_new_artifacts(client, session_id, seen):
            print(f"[saved: {path}]")

    def safe_turn(question: str) -> bool:
        try:
            run_turn(question)
            return True
        except KeyboardInterrupt:
            print("\n[interrupted]")
            return False
        except Exception as exc:
            print(f"\n[error] {exc}")
            if session_id is not None:
                print(
                    "If the sandbox expired (about an hour without activity), "
                    "start again with:  python cpa_agent.py --new"
                )
            return False

    if args.ask:
        return 0 if safe_turn(args.ask) else 1

    while True:
        try:
            question = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "/exit"):
            break
        print()
        safe_turn(question)

    return 0


if __name__ == "__main__":
    sys.exit(main())
