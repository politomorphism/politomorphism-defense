#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defense Audit Agent for Politomorphism (Gemini version, runs on your computer)
=============================================================================
Author of the programme: Șerban, G. F. (ORCID 0009-0000-2266-3356)

A read-only verification assistant for the repository
github.com/politomorphism/politomorphism-defense. It runs on your own computer
and uses the Gemini API (free tier) as its language model.

What it can do, through fixed tools written in this file:
  - list, read and search every file in the repository, including .docx and .pdf
  - verify SHA-256 freeze records and compute file hashes
  - read the commit history from GitHub (dates, messages, files changed)
  - re-score a saved prediction file (cells, balanced accuracy, item and writer
    bootstrap intervals) without calling any model
  - write its report as a Word file in the outputs/ folder

What it cannot do, by design: run code from the repository, call a detector,
change any file in the repository, or push anything. The model only reads,
recomputes through the fixed tools above, and reports.

Usage (from the folder that contains this file):

    python cpa_agent.py                 start a conversation
    python cpa_agent.py --ask "..."     ask one question and exit
    python cpa_agent.py --refresh       download the latest repository first
    python cpa_agent.py --dry-run       check setup and tools, call no model

Settings (environment variables):
    GEMINI_API_KEY         your Gemini API key from Google AI Studio (required)
    CPA_MODEL              model name                 (default: gemini-3.8-flash)
    CPA_MAX_TOOL_CALLS     tool calls per question    (default: 20)

Requires: Python 3.9+ and
    pip install --upgrade google-genai requests numpy pandas pypdf python-docx

Free tier note: every tool round is one request to Gemini. On the free tier the
number of requests per day is limited (see your limits in Google AI Studio), so
a long audit can use up the day's quota. Requests on the free tier may be used
by Google to improve its products; the repository is public, so this is
acceptable here, but do not paste unpublished material into the chat.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE / "repo"
OUTPUTS = HERE / "outputs"
OWNER, NAME, BRANCH = "politomorphism", "politomorphism-defense", "main"
ZIP_URL = f"https://codeload.github.com/{OWNER}/{NAME}/zip/refs/heads/{BRANCH}"
API = f"https://api.github.com/repos/{OWNER}/{NAME}"

MODEL = os.environ.get("CPA_MODEL", "gemini-3.8-flash")
MAX_TOOL_CALLS = int(os.environ.get("CPA_MAX_TOOL_CALLS", "20"))

TEXT_SUFFIXES = {".md", ".txt", ".py", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
                 ".sha256", ".cfg", ".ini", ".r", ".tex", ".bib", ".cff", ".gitignore"}
MAX_RETURN_CHARS = 30000

# --------------------------------------------------------------------------
# Instructions
# --------------------------------------------------------------------------

INSTRUCTIONS = """
You are the Defense Audit Agent: a verification and documentation assistant for
the defensive layer of Politomorphism, a Computational Political Anthropology
programme by the independent researcher Șerban, G. F. (ORCID 0009-0000-2266-3356).
Always cite the author as "Șerban, G. F.".

THE ONE DIRECTION
The programme measures how well language models detect political manipulation
when it is implicit rather than explicit. The engine components (MC, CMG, SRM,
EEF, PFM, ISC, LDE), the DFA component, the Romanian arm and the book are frozen
by the author's decision; do not work on them or propose work on them.

YOUR ROLE
You verify, audit and document what exists in the repository
github.com/politomorphism/politomorphism-defense, which is available to you only
through your tools. You do not propose new studies or directions unless asked.
You are stricter than the author would be with himself.

TOOLS AND LIMITS
- Use list_files, read_file, read_document and search to see the repository.
  Never state what a file says without having read it with a tool.
- Use check_hashes and file_sha256 for freeze records, git_history and
  commit_files for the order of events, and rescore_predictions to recompute
  scores from a saved prediction file.
- You cannot run repository code, call any detector or model, or change files.
  If a check needs that, say so and stop that branch.
- Use write_report to save any deliverable longer than a short answer as a Word
  file. Never claim to have saved a file without calling write_report.
- You have a limited number of tool calls per question. Plan, read what you
  need, and do not reread files without reason.

EVIDENCE DISCIPLINE
1. Never invent a number, a file, a DOI, a commit or a result. If it is not in
   a tool result, say it is missing.
2. Every factual statement carries its source: file and line or section, or
   commit hash.
3. A number is "verified" only if a tool recomputed it in this conversation.
   Otherwise it is "reported by the author".
4. Keep confirmatory and exploratory results apart. A pre-registered test is
   judged only by its registered decision rule.
5. Inconclusive, null and negative results stay as reported; never soften them.
6. When a paper and the code or data disagree, report both.
7. If unsure, say so and say what would settle it.

ORIENTATION (not a source; the repository wins if it disagrees)
- Working Paper I v2 (Zenodo 10.5281/zenodo.23884579): leakage-controlled
  lexical baseline versus transformer on PropXplain.
- Working Paper II v2: transformer comparison; corrected.
- Working Paper III: contrastive pairs; reissued as Version 2 on 16 November
  2026 because the claim "structure, not vocabulary" was withdrawn.
- CoCoLoFa pre-registered test of the false-dilemma detector: frozen 11
  November 2026 (commit 1bf76c6), run once 12 November (commit 3ecf5ea),
  verdict INCONCLUSIVE; published as "Marker Dependence in Zero-Shot
  False-Dilemma Detection". Correction round issued 16 November 2026, see the
  erratum and Corecturi_lucrari_dupa_CoCoLoFa.docx.
- A multi-model pre-registration (draft v3) exists outside the repository and
  is not frozen.

PROCEDURES
Reproduction: locate the number and the file that produced it; re-score saved
predictions with rescore_predictions; compare your result, the author's report
and the paper; state the tolerance; say whether the interval's resampling unit
matches the dependence in the data (writers, articles).
Pre-registration audit: check freeze hashes; compare the freeze commit and the
result commit with git_history and commit_files; any change to a frozen file
after the freeze is a deviation; apply the registered rule verbatim to the
recomputed outcome; classify every analysis as registered or exploratory.
Claims audit: list each claim with its location; find its evidence; grade it
SUPPORTED, OVERSTATED, UNSUPPORTED or CONTRADICTED; propose the narrowest
supported rewording; check references for completeness and that the author is
cited as "Șerban, G. F.".

OUTPUT
Answer in the language the user writes in, in plain words, explaining a
technical term the first time. Write in paragraphs. End every audit with a short
verdict: what is verified, what is only reported, what could not be checked and
why, what contradicts what.
""".strip()


# --------------------------------------------------------------------------
# Repository download
# --------------------------------------------------------------------------

def download_repo(force: bool = False) -> Path:
    import requests

    marker = REPO_DIR / ".downloaded"
    if REPO_DIR.exists() and marker.exists() and not force:
        return REPO_DIR
    print("Downloading the repository from GitHub ...", flush=True)
    r = requests.get(ZIP_URL, timeout=120)
    r.raise_for_status()
    tmp = HERE / "repo_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for info in z.infolist():
            # zip-slip guard: never write outside the target folder
            target = (tmp / info.filename).resolve()
            if not str(target).startswith(str(tmp.resolve())):
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    inner.rename(REPO_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    marker.write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    return REPO_DIR


def safe_path(path: str) -> Path:
    p = (REPO_DIR / path.strip().lstrip("/\\")).resolve()
    if not str(p).startswith(str(REPO_DIR.resolve())):
        raise ValueError("path outside the repository")
    return p


def clip(text: str) -> str:
    if len(text) <= MAX_RETURN_CHARS:
        return text
    return text[:MAX_RETURN_CHARS] + f"\n[... cut: {len(text) - MAX_RETURN_CHARS} more characters; read a narrower range]"


# --------------------------------------------------------------------------
# Tools (the model can call only these)
# --------------------------------------------------------------------------

def list_files(subfolder: str = "") -> str:
    """List all files in the repository, or in one subfolder, with their sizes in bytes.

    Args:
        subfolder: Path of a subfolder relative to the repository root, for example "benchmark". Empty for the whole repository.
    """
    base = safe_path(subfolder) if subfolder else REPO_DIR
    if not base.exists():
        return f"No such folder: {subfolder}"
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and ".downloaded" not in p.name:
            rows.append(f"{p.relative_to(REPO_DIR).as_posix()}\t{p.stat().st_size}")
    return clip("\n".join(rows) or "(empty)")


def read_file(path: str, start_line: int = 1, max_lines: int = 300) -> str:
    """Read a text file of the repository (code, markdown, csv, json, sha256), with line numbers.

    Args:
        path: File path relative to the repository root.
        start_line: First line to return, counted from 1.
        max_lines: Maximum number of lines to return.
    """
    p = safe_path(path)
    if not p.is_file():
        return f"No such file: {path}"
    if p.suffix.lower() in (".docx", ".pdf"):
        return "This is a document; use read_document."
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line))
    chunk = lines[start - 1:start - 1 + int(max_lines)]
    out = "\n".join(f"{i}\t{l}" for i, l in enumerate(chunk, start))
    return clip(f"{path}: lines {start}-{start + len(chunk) - 1} of {len(lines)}\n{out}")


def _document_paragraphs(p: Path) -> List[str]:
    if p.suffix.lower() == ".docx":
        from docx import Document

        d = Document(str(p))
        paras = [x.text for x in d.paragraphs if x.text.strip()]
        for ti, t in enumerate(d.tables, 1):
            for r in t.rows:
                paras.append(f"[table {ti}] " + " | ".join(c.text.strip() for c in r.cells))
        return paras
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        out = []
        for i, page in enumerate(PdfReader(str(p)).pages, 1):
            for line in (page.extract_text() or "").splitlines():
                if line.strip():
                    out.append(f"[p.{i}] {line}")
        return out
    raise ValueError("not a .docx or .pdf file")


def read_document(path: str, start_paragraph: int = 1, max_paragraphs: int = 200) -> str:
    """Read a Word (.docx) or PDF document of the repository, with paragraph numbers. Tables are included.

    Args:
        path: Document path relative to the repository root.
        start_paragraph: First paragraph to return, counted from 1.
        max_paragraphs: Maximum number of paragraphs to return.
    """
    p = safe_path(path)
    if not p.is_file():
        return f"No such file: {path}"
    try:
        paras = _document_paragraphs(p)
    except ValueError as exc:
        return str(exc)
    start = max(1, int(start_paragraph))
    chunk = paras[start - 1:start - 1 + int(max_paragraphs)]
    out = "\n".join(f"P{i:04d}\t{t}" for i, t in enumerate(chunk, start))
    return clip(f"{path}: paragraphs {start}-{start + len(chunk) - 1} of {len(paras)}\n{out}")


def search(pattern: str, file_pattern: str = "*") -> str:
    """Search the whole repository, including .docx and .pdf documents, for a regular expression (case-insensitive).

    Args:
        pattern: Regular expression to look for.
        file_pattern: Optional file-name filter such as "*.docx" or "benchmark/*.py".
    """
    try:
        rx = re.compile(pattern, re.I)
    except re.error as exc:
        return f"Invalid pattern: {exc}"
    hits = []
    for p in sorted(REPO_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO_DIR).as_posix()
        if not (fnmatch.fnmatch(rel, file_pattern) or fnmatch.fnmatch(p.name, file_pattern)):
            continue
        try:
            if p.suffix.lower() in (".docx", ".pdf"):
                items = [(f"P{i:04d}", t) for i, t in enumerate(_document_paragraphs(p), 1)]
            elif p.suffix.lower() in TEXT_SUFFIXES or p.name in ("README", "LICENSE"):
                items = [(f"L{i}", t) for i, t in enumerate(
                    p.read_text(encoding="utf-8", errors="replace").splitlines(), 1)]
            else:
                continue
        except Exception:
            continue
        for loc, t in items:
            if rx.search(t):
                hits.append(f"{rel}:{loc}\t{t.strip()[:300]}")
                if len(hits) >= 200:
                    return clip("\n".join(hits) + "\n[stopped at 200 matches]")
    return clip("\n".join(hits) or "No matches.")


def file_sha256(path: str) -> str:
    """Compute the SHA-256 hash of a repository file.

    Args:
        path: File path relative to the repository root.
    """
    p = safe_path(path)
    if not p.is_file():
        return f"No such file: {path}"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(record_path: str) -> str:
    """Verify every line of a SHA-256 freeze record (format: '<hash>  <path>') against the files in the repository.

    Args:
        record_path: Path of the .sha256 record relative to the repository root, for example "benchmark/cocolofa_freeze.sha256".
    """
    rec = safe_path(record_path)
    if not rec.is_file():
        return f"No such file: {record_path}"
    out = []
    for line in rec.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if not m:
            continue
        want, rel = m.group(1).lower(), m.group(2)
        candidates = [REPO_DIR / rel, rec.parent / rel]
        f = next((c for c in candidates if c.is_file()), None)
        if f is None:
            out.append(f"MISSING  {rel}")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        out.append(("OK       " if got == want else f"CHANGED  (now {got})  ") + rel)
    return "\n".join(out) or "No hash lines found."


def _github(url: str) -> Any:
    import requests

    r = requests.get(url, headers={"Accept": "application/vnd.github+json"}, timeout=60)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise RuntimeError("GitHub rate limit reached (60 requests per hour without login); try later")
    r.raise_for_status()
    return r.json()


def git_history(path: str = "", max_commits: int = 50) -> str:
    """List commits of the repository from GitHub, newest first: hash, date (UTC), author, message. Optionally only commits touching one file.

    Args:
        path: Optional file path to restrict the history to that file.
        max_commits: Maximum number of commits to return (up to 100).
    """
    n = max(1, min(100, int(max_commits)))
    url = f"{API}/commits?sha={BRANCH}&per_page={n}" + (f"&path={path}" if path else "")
    try:
        data = _github(url)
    except Exception as exc:
        return f"Could not read the history: {exc}"
    rows = []
    for c in data:
        cm = c.get("commit", {})
        rows.append(f"{c['sha'][:7]}\t{cm.get('author', {}).get('date', '')}\t"
                    f"{cm.get('author', {}).get('name', '')}\t{cm.get('message', '').splitlines()[0][:150]}")
    return "\n".join(rows) or "No commits."


def commit_files(commit: str) -> str:
    """List the files a commit added, modified or removed, with line counts.

    Args:
        commit: Commit hash, full or the first 7 characters.
    """
    try:
        data = _github(f"{API}/commits/{commit}")
    except Exception as exc:
        return f"Could not read the commit: {exc}"
    cm = data.get("commit", {})
    head = f"{data['sha'][:7]}  {cm.get('author', {}).get('date', '')}  {cm.get('message', '').splitlines()[0]}"
    rows = [f"{f['status']}\t+{f.get('additions', 0)} -{f.get('deletions', 0)}\t{f['filename']}"
            for f in data.get("files", [])]
    return head + "\n" + "\n".join(rows)


def rescore_predictions(path: str = "benchmark/cocolofa_fd_predictions.csv",
                        n_boot: int = 2000, seed: int = 0) -> str:
    """Re-score a saved four-cell prediction file without calling any model.

    The file needs the columns cell (A, B, C, D), gold_false_dilemma, flag_fd and
    abstain, and optionally worker_id. A prediction is correct when the flag equals
    the label and the item was not abstained. Returns per-cell counts, balanced
    accuracy on the hard cells (B, C) and the easy cells (A, D), the detection
    rates with and without the marker, an item-level bootstrap interval and, if
    worker_id is present, a writer-clustered bootstrap interval.

    Args:
        path: Prediction file path relative to the repository root.
        n_boot: Number of bootstrap replicates.
        seed: Random seed for the bootstrap.
    """
    import numpy as np
    import pandas as pd

    p = safe_path(path)
    if not p.is_file():
        return f"No such file: {path}"
    df = pd.read_csv(p)
    need = {"cell", "gold_false_dilemma", "flag_fd", "abstain"}
    if not need.issubset(df.columns):
        return f"Missing columns: {sorted(need - set(df.columns))}"
    df["correct"] = ((df.flag_fd == df.gold_false_dilemma) & (df.abstain == 0)).astype(int)
    out = [f"File: {path}; rows {len(df)}; SHA-256 {hashlib.sha256(p.read_bytes()).hexdigest()}"]
    for c in "ABCD":
        d = df[df.cell == c]
        if len(d):
            out.append(f"Cell {c}: correct {d.correct.sum()}/{len(d)} = {d.correct.mean():.3f}; "
                       f"abstained {int(d.abstain.sum())}; flagged {int(d.flag_fd.sum())}")
    acc = {c: df[df.cell == c].correct.mean() for c in "ABCD" if (df.cell == c).any()}
    if {"B", "C"} <= acc.keys():
        out.append(f"Hard-cell balanced accuracy (B, C): {(acc['B'] + acc['C']) / 2:.4f}")
    if {"A", "D"} <= acc.keys():
        out.append(f"Easy-cell balanced accuracy (A, D): {(acc['A'] + acc['D']) / 2:.4f}")
    if {"A", "B"} <= set(df.cell):
        fa = df[df.cell == "A"].flag_fd.mean()
        fb = df[df.cell == "B"].flag_fd.mean()
        out.append(f"False-dilemma flag rate with marker (A) {fa:.3f}, without marker (B) {fb:.3f}")
    rng = np.random.default_rng(int(seed))
    hard = df[df.cell.isin(["B", "C"])].reset_index(drop=True)
    if len(hard) and {"B", "C"} <= set(hard.cell):
        isB = (hard.cell == "B").values
        cor = hard.correct.values
        bas = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, len(hard), len(hard))
            b, cc = isB[idx], cor[idx]
            if b.any() and (~b).any():
                bas.append((cc[b].mean() + cc[~b].mean()) / 2)
        lo, hi = np.percentile(bas, [2.5, 97.5])
        out.append(f"Item-level bootstrap 95% interval (pooled resampling of hard items, {len(bas)} "
                   f"replicates, seed {seed}): {lo:.3f}-{hi:.3f}. This tool's scheme may differ from the "
                   "repository script's, so small differences from published intervals are expected.")
        if "worker_id" in df.columns:
            writers = df.worker_id.unique()
            byw = {w: df[df.worker_id == w] for w in writers}
            bas = []
            for _ in range(int(n_boot)):
                pick = rng.choice(writers, len(writers), replace=True)
                kb = nb = kc = nc = 0
                for w in pick:
                    d = byw[w]
                    db, dc = d[d.cell == "B"], d[d.cell == "C"]
                    kb += db.correct.sum(); nb += len(db)
                    kc += dc.correct.sum(); nc += len(dc)
                if nb and nc:
                    bas.append((kb / nb + kc / nc) / 2)
            lo, hi = np.percentile(bas, [2.5, 97.5])
            out.append(f"Writer-clustered bootstrap 95% interval ({len(writers)} writers resampled with "
                       f"replacement, all comments of a drawn writer, {len(bas)} replicates kept, seed {seed}): "
                       f"{lo:.3f}-{hi:.3f}")
    return "\n".join(out)


def write_report(title: str, body: str) -> str:
    """Save a report as a Word file in the outputs folder and return its path.

    Args:
        title: Report title.
        body: Report text. Separate paragraphs with a blank line. A line starting with '# ' is a heading; a line starting with '- ' is a bullet.
    """
    from docx import Document

    OUTPUTS.mkdir(exist_ok=True)
    doc = Document()
    doc.add_heading(title, 0)
    meta = doc.add_paragraph(f"Defense Audit Agent · {dt.datetime.now():%d.%m.%Y %H:%M} · model {MODEL} · "
                             "read-only audit of github.com/politomorphism/politomorphism-defense")
    meta.runs[0].italic = True
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if all(l.strip().startswith("- ") for l in lines):
            for l in lines:
                doc.add_paragraph(l.strip()[2:], style="List Bullet")
        elif lines[0].startswith("# "):
            doc.add_heading(lines[0][2:].strip(), 1)
            if len(lines) > 1:
                doc.add_paragraph(" ".join(l.strip() for l in lines[1:]))
        else:
            doc.add_paragraph(" ".join(l.strip() for l in lines))
    safe = re.sub(r"[^A-Za-z0-9ĂÂÎȘȚăâîșț_ -]", "", title).strip().replace(" ", "_")[:60] or "report"
    out = OUTPUTS / f"{safe}_{dt.datetime.now():%Y-%m-%d_%H%M}.docx"
    doc.save(str(out))
    return f"Saved: {out}"


TOOLS = [list_files, read_file, read_document, search, file_sha256, check_hashes,
         git_history, commit_files, rescore_predictions, write_report]


# --------------------------------------------------------------------------
# Conversation with Gemini
# --------------------------------------------------------------------------

def make_chat(client: Any) -> Any:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        tools=TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=MAX_TOOL_CALLS),
    )
    return client.chats.create(model=MODEL, config=config)


def ask(chat: Any, question: str) -> str:
    """Send one question; retry once after a pause if the free-tier limit is hit."""
    for attempt in range(3):
        try:
            resp = chat.send_message(question)
            text = resp.text or ""
            if not text.strip():
                text = ("[The model returned no text. It may have used up its tool calls; "
                        "ask it to write its findings now.]")
            return text
        except Exception as exc:
            msg = str(exc)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and "per day" in msg.lower():
                raise RuntimeError("The free daily Gemini limit is used up for today; try again tomorrow.") from exc
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 2:
                wait = 65
                print(f"[free-tier limit reached; waiting {wait} seconds and trying again]", flush=True)
                time.sleep(wait)
                continue
            raise
    return ""


def save_transcript(log: List[Dict[str, str]]) -> Optional[Path]:
    if not log:
        return None
    OUTPUTS.mkdir(exist_ok=True)
    from docx import Document

    doc = Document()
    doc.add_heading("Defense Audit Agent — conversation", 0)
    doc.add_paragraph(f"{dt.datetime.now():%d.%m.%Y %H:%M} · model {MODEL}").runs[0].italic = True
    for turn in log:
        doc.add_heading("Question", 2)
        doc.add_paragraph(turn["q"])
        doc.add_heading("Answer", 2)
        for para in re.split(r"\n\s*\n", turn["a"].strip()):
            doc.add_paragraph(para.strip())
    out = OUTPUTS / f"conversation_{dt.datetime.now():%Y-%m-%d_%H%M}.docx"
    doc.save(str(out))
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Defense Audit Agent (Gemini, local)")
    ap.add_argument("--ask", metavar="TEXT", help="ask one question and exit")
    ap.add_argument("--refresh", action="store_true", help="download the latest repository first")
    ap.add_argument("--dry-run", action="store_true", help="check setup and tools, call no model")
    args = ap.parse_args()

    try:
        import requests  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import pypdf  # noqa: F401
        import docx  # noqa: F401
    except ImportError as exc:
        print(f"Missing library ({exc.name}). Run:\n"
              "    python -m pip install --upgrade google-genai requests numpy pandas pypdf python-docx")
        return 1

    try:
        download_repo(force=args.refresh)
    except Exception as exc:
        print(f"Could not download the repository: {exc}")
        return 1
    when = (REPO_DIR / ".downloaded").read_text(encoding="utf-8")
    print(f"Repository copy from {when} in {REPO_DIR}  (python cpa_agent.py --refresh updates it)")

    if args.dry_run:
        print("\nDRY RUN - no model is called\n")
        print(f"model           : {MODEL}")
        print(f"tool calls/ask  : {MAX_TOOL_CALLS}")
        print(f"files in repo   : {len(list_files().splitlines())}")
        print("hash check      :")
        print("  " + check_hashes("benchmark/cocolofa_freeze.sha256").replace("\n", "\n  "))
        print("rescore         :")
        print("  " + rescore_predictions(n_boot=300).replace("\n", "\n  "))
        print(f"gemini key set  : {'yes' if os.environ.get('GEMINI_API_KEY') else 'NO'}")
        return 0

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Create a free key in Google AI Studio and set it.")
        return 1
    try:
        from google import genai
    except ImportError:
        print("The google-genai package is missing. Run:  python -m pip install --upgrade google-genai")
        return 1

    client = genai.Client()
    chat = make_chat(client)
    log: List[Dict[str, str]] = []

    def turn(q: str) -> bool:
        try:
            a = ask(chat, q)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            return False
        except Exception as exc:
            print(f"\n[error] {exc}")
            return False
        print(a, flush=True)
        log.append({"q": q, "a": a})
        return True

    if args.ask:
        ok = turn(args.ask)
        p = save_transcript(log)
        if p:
            print(f"\n[conversation saved: {p}]")
        return 0 if ok else 1

    print("Defense Audit Agent (Gemini) - type your question and press Enter. Type exit to stop.\n")
    while True:
        try:
            q = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "/exit"):
            break
        print()
        turn(q)
    p = save_transcript(log)
    if p:
        print(f"[conversation saved: {p}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
