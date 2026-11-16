# Defense Audit Agent

`cpa_agent.py` is a read-only verification agent for this repository. It runs on your
own computer and uses the Gemini API (free tier) as its language model. It downloads a
copy of this repository from GitHub and works only through fixed tools written in the
file: it lists, reads and searches every file (including .docx and .pdf), verifies
SHA-256 freeze records, reads the commit history, re-scores saved prediction files
(cells, balanced accuracy, item and writer bootstrap intervals) without calling any
model, and writes its reports as Word files. It cannot run repository code, call a
detector, change the repository or push anything.

The correction round of 16 November 2026 (`Corecturi_lucrari_dupa_CoCoLoFa.docx` in the
repository root) was produced with an earlier version of this agent, which ran on the
OpenAI Agents API; that version is in the history of this folder.

## Run it

Requires Python 3.9+ and

```
pip install --upgrade google-genai requests numpy pandas pypdf python-docx
```

and a Gemini API key from Google AI Studio in the environment variable
`GEMINI_API_KEY`. The key is never stored in this repository.

```
python cpa_agent.py --dry-run     # check setup and tools; no model is called
python cpa_agent.py               # start a conversation
python cpa_agent.py --ask "..."   # one question, then exit
python cpa_agent.py --refresh     # download the latest repository first
```

Optional settings: `CPA_MODEL` (default `gemini-3.8-flash`) and `CPA_MAX_TOOL_CALLS`
(default 20). On the free tier every tool round is one request and the number of
requests per day is limited; requests on the free tier may be used by Google to
improve its products, so do not paste unpublished material into the conversation.
Folders `repo/` (the downloaded copy) and `outputs/` (reports) are created next to the
script and are not committed.

Author: Șerban, G. F. · ORCID 0009-0000-2266-3356 · Code: MIT.
