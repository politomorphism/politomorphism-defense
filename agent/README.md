# Defense Audit Agent

`cpa_agent.py` is a read-only verification agent for this repository. It runs on the
OpenAI Agents API in an OpenAI-hosted sandbox, clones this repository, and does three
jobs: it reproduces reported numbers from the saved prediction files, audits
pre-registered tests (freeze hashes, decision rule, run record), and checks the claims
in a paper against their evidence. It never re-runs a detector on a frozen test sample
and never changes the repository; it only reads, recomputes and reports. Reports are
written as Word files.

The correction round of 16 November 2026 (`Corecturi_lucrari_dupa_CoCoLoFa.docx` in the
repository root) was produced with this agent.

## Run it

Requires Python 3.9+ and `pip install --upgrade openai`, and an OpenAI API key in the
environment variable `OPENAI_API_KEY`. The key is never stored in this repository.

```
python cpa_agent.py --dry-run     # show what would be sent; no cost
python cpa_agent.py --new         # start a fresh session
python cpa_agent.py               # continue the last session
python cpa_agent.py --ask "..."   # one question, then exit
```

Optional settings (environment variables): `CPA_MODEL`, `CPA_REASONING_EFFORT`,
`CPA_MAX_SUBAGENTS`, `CPA_CONTAINER_SIZE`, `CPA_REPOS`. Folders `knowledge/` (optional
extra papers to upload) and `outputs/` (downloaded reports) are created next to the
script and are not committed.

Author: Șerban, G. F. · ORCID 0009-0000-2266-3356 · Code: MIT.
