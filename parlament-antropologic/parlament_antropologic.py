#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lectura antropologica a proiectelor de lege aflate in consultare publica la Senat
================================================================================
Autor: Șerban, G. F. (ORCID 0009-0000-2266-3356)

Programul:
  1. citeste lista oficiala a proiectelor aflate in consultare publica la Senat
     (senat.ro/legiproiect.aspx);
  2. descarca expunerea de motive (PDF) a fiecarui proiect si extrage textul;
  3. trimite textul unui model de limbaj care face o lectura antropologica
     descriptiva: comunitatea imaginata, registrul simbolic, sursele de
     legitimare si dispozitivele retorice, fiecare cu citat exact;
  4. verifica automat ca fiecare citat exista in textul expunerii de motive;
     citatele care nu pot fi gasite sunt mutate intr-o anexa, nu in raport;
  5. scrie raportul in Word, in folderul rapoarte/.

Folosire (din folderul in care se afla acest fisier):

    python parlament_antropologic.py              raport pentru cel mult 12 proiecte
    python parlament_antropologic.py --max 25     raport pentru cel mult 25 de proiecte
    python parlament_antropologic.py --dry-run    descarca si citeste, fara model, fara cost
    python parlament_antropologic.py --only B573  doar proiectul B573
    python parlament_antropologic.py --probe      salveaza paginile Senatului pentru diagnostic
    python parlament_antropologic.py --model gemini-3.5-flash-lite   alt model gratuit, daca primul e ocupat

Setari (variabile de mediu, toate optionale in afara de cheie):
    GEMINI_API_KEY          cheia Gemini din Google AI Studio (obligatorie, cu
                            exceptia --dry-run si --probe); nivelul gratuit ajunge
    PA_MODEL                modelul folosit             (implicit: gemini-3.8-flash)
    PA_PAUSE                secunde intre doua lecturi  (implicit: 6)

Necesita: Python 3.9+ si
    pip install --upgrade google-genai requests beautifulsoup4 pypdf python-docx

Nivelul gratuit Gemini are o limita de cereri pe zi (o vezi in Google AI Studio).
Fiecare proiect citit inseamna o cerere; lecturile deja facute sunt pastrate in
cache si nu se mai cer a doua oara. Pe nivelul gratuit, Google poate folosi
cererile pentru a-si imbunatati produsele; expunerile de motive sunt publice.

Limite: lectura este asistata de un model de limbaj si este descriptiva, nu un
verdict politic sau juridic. Detectarea dispozitivelor retorice nu este
validata pe limba romana.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
REPORTS = HERE / "rapoarte"
PROBE = HERE / "probe"

SENATE_BASE = "https://www.senat.ro/"
SENATE_LIST = "https://www.senat.ro/legiproiect.aspx"
PROJECT_URL = "https://www.senat.ro/Legis/Lista.aspx?cod={cod}&pos=0&NR=b{nr}&AN={year}"
EM_FALLBACK = "https://www.senat.ro/Legis/PDF/{year}/{yy}b{nr:03d}EM.pdf"

MODEL = os.environ.get("PA_MODEL", "gemini-3.8-flash")
MODEL_PAUSE = float(os.environ.get("PA_PAUSE", "6"))
PROMPT_VERSION = "2026-11-16.1"

MAX_TEXT_CHARS = 60000      # longer memoranda are cut; the report says so
MIN_TEXT_CHARS = 200        # below this the PDF has no usable text layer
REQUEST_PAUSE = 1.0         # seconds between requests to senat.ro
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "parlament-antropologic (research; contact@politomorphism.org)"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.5",
}

# --------------------------------------------------------------------------
# Model instructions and output schema
# --------------------------------------------------------------------------

INSTRUCTIONS = """
Ești un asistent de cercetare în antropologie politică. Primești expunerea de
motive a unei propuneri legislative depuse în Parlamentul României și faci o
lectură antropologică DESCRIPTIVĂ a textului.

Ce descrii:
1. Comunitatea imaginată: pe cine construiește textul ca „noi” și pe cine ca
   „ei” sau „ceilalți”; ce categorii de persoane creează, numește sau
   delimitează (beneficiari, vinovați, victime, amenințări, străini, copii,
   familia etc.).
2. Registrul simbolic: ce simboluri, memorie istorică, opoziții (pur/impur,
   local/străin, ordine/haos, sacru/profan, suveranitate/dependență) sau
   imagini morale mobilizează textul.
3. Legitimarea: pe ce autoritate se sprijină argumentul (poporul, experții,
   Constituția și legea, UE sau organisme internaționale, tradiția și istoria,
   urgența sau criza, morala, eficiența economică).
4. Dispozitive retorice: falsă dilemă, presupoziție, cauzalitate implicită,
   generalizare, apel la frică, echivalență falsă, întrebare retorică,
   eufemism. Pentru fiecare spui dacă are un marker lexical explicit (de
   exemplu „ori… ori”, „singura soluție”, „fie… fie”) sau dacă este implicit.

Reguli obligatorii:
- Fiecare observație se sprijină pe un CITAT EXACT din text, de cel mult 25 de
  cuvinte, copiat literă cu literă, inclusiv greșelile de scanare, diacriticele
  și punctuația din text. Nu corecta, nu parafraza și nu combina fragmente în
  citat. Dacă nu poți cita exact, nu face observația.
- Ești strict neutru politic. Nu spui dacă legea este bună sau rea, nu evaluezi
  partide, inițiatori sau tabere, nu faci predicții și nu dai sfaturi. Aplici
  aceleași criterii oricărui text, indiferent de orientarea lui.
- Descrii ce face textul, nu intențiile autorilor.
- Dacă textul este tehnic și nu are nimic relevant antropologic, spui asta și
  lași listele goale. Nu umple spațiul.
- „interes_antropologic” arată cât de mult material oferă textul pentru o
  lectură antropologică. Nu este un scor de risc și nu este o judecată despre
  lege.
- Scrii în limba română, clar, fără jargon inutil.
""".strip()

TECHNIQUES = [
    "falsa_dilema", "presupozitie", "cauzalitate_implicita", "generalizare",
    "apel_la_frica", "echivalenta_falsa", "intrebare_retorica", "eufemism", "altele",
]
LEGITIMATION = [
    "popor", "expertiza", "constitutie_lege", "ue_international", "traditie_istorie",
    "urgenta_criza", "morala_valori", "eficienta_economica", "altele",
]

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "rezumat", "interes_antropologic", "motivul_interesului", "comunitate_imaginata",
        "registru_simbolic", "legitimare", "dispozitive_retorice", "ce_nu_se_poate_spune",
    ],
    "properties": {
        "rezumat": {"type": "string", "description": "Ce propune legea, în 1-2 fraze neutre."},
        "interes_antropologic": {"type": "string", "enum": ["ridicat", "mediu", "scazut"]},
        "motivul_interesului": {"type": "string"},
        "comunitate_imaginata": {
            "type": "object",
            "additionalProperties": False,
            "required": ["noi", "ei_sau_ceilalti", "categorii_create"],
            "properties": {
                "noi": {"type": "string"},
                "ei_sau_ceilalti": {"type": "string"},
                "categorii_create": {"type": "array", "items": {"type": "string"}},
            },
        },
        "registru_simbolic": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tema", "descriere", "citat"],
                "properties": {
                    "tema": {"type": "string"},
                    "descriere": {"type": "string"},
                    "citat": {"type": "string"},
                },
            },
        },
        "legitimare": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sursa", "descriere", "citat"],
                "properties": {
                    "sursa": {"type": "string", "enum": LEGITIMATION},
                    "descriere": {"type": "string"},
                    "citat": {"type": "string"},
                },
            },
        },
        "dispozitive_retorice": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tehnica", "citat", "explicatie", "marker_explicit"],
                "properties": {
                    "tehnica": {"type": "string", "enum": TECHNIQUES},
                    "citat": {"type": "string"},
                    "explicatie": {"type": "string"},
                    "marker_explicit": {"type": "boolean"},
                },
            },
        },
        "ce_nu_se_poate_spune": {
            "type": "string",
            "description": "Ce nu se poate concluziona doar din acest text.",
        },
    },
}

TECHNIQUE_LABEL = {
    "falsa_dilema": "Falsă dilemă", "presupozitie": "Presupoziție",
    "cauzalitate_implicita": "Cauzalitate implicită", "generalizare": "Generalizare",
    "apel_la_frica": "Apel la frică", "echivalenta_falsa": "Echivalență falsă",
    "intrebare_retorica": "Întrebare retorică", "eufemism": "Eufemism", "altele": "Altă tehnică",
}
LEGITIMATION_LABEL = {
    "popor": "Poporul", "expertiza": "Expertiza", "constitutie_lege": "Constituția și legea",
    "ue_international": "UE și organisme internaționale", "traditie_istorie": "Tradiția și istoria",
    "urgenta_criza": "Urgența sau criza", "morala_valori": "Morala și valorile",
    "eficienta_economica": "Eficiența economică", "altele": "Altă sursă",
}
INTEREST_ORDER = {"ridicat": 0, "mediu": 1, "scazut": 2}
INTEREST_LABEL = {"ridicat": "ridicat", "mediu": "mediu", "scazut": "scăzut"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def say(msg: str) -> None:
    print(msg, flush=True)


def normalise(text: str) -> str:
    """Normalise text so that a quote can be found despite cedilla/comma
    diacritics, hyphenation at line ends, quotes and spacing."""
    t = unicodedata.normalize("NFC", text)
    t = (t.replace("ş", "ș").replace("Ş", "Ș").replace("ţ", "ț").replace("Ţ", "Ț")
           .replace("­", ""))
    t = re.sub(r"[‘’‚‛`´]", "'", t)
    t = re.sub(r"[“”„‟«»]", '"', t)
    t = re.sub(r"[–—−]", "-", t)
    t = re.sub(r"-\s*\n\s*", "", t)          # word broken across lines
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def match_key(text: str) -> str:
    """Accent-free, punctuation-free form used to match quotes against scanned
    text, where extraction may damage diacritics (e.g. 'act$ionam')."""
    t = unicodedata.normalize("NFKD", normalise(text))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_status(quote: str, source_norm: str, source_key: str) -> str:
    """'exact', 'aproximativ' (scanning noise) or '' (not found)."""
    if quote_found(quote, source_norm):
        return "exact"
    qk = match_key(quote)
    if len(qk) < 8:
        return ""
    if qk in source_key:
        return "aproximativ"
    parts = [match_key(p) for p in re.split(r"\.\.\.|…|\[\.\.\.\]", quote)]
    parts = [p for p in parts if len(p) >= 4]
    if len(parts) > 1:
        pos = 0
        for part in parts:
            i = source_key.find(part, pos)
            if i < 0:
                return ""
            pos = i + len(part)
        return "aproximativ"
    import difflib
    sm = difflib.SequenceMatcher(None, source_key, qk, autojunk=False)
    m = sm.find_longest_match(0, len(source_key), 0, len(qk))
    if m.size < 6:
        return ""
    start = max(0, m.a - m.b - 5)
    window = source_key[start:start + len(qk) + 10]
    ratio = difflib.SequenceMatcher(None, window, qk, autojunk=False).ratio()
    best = 0.0
    for off in range(0, max(1, len(window) - len(qk) + 1)):
        r = difflib.SequenceMatcher(None, window[off:off + len(qk)], qk, autojunk=False).ratio()
        best = max(best, r)
    return "aproximativ" if max(ratio, best) >= 0.92 else ""


def quote_found(quote: str, source_norm: str) -> bool:
    q = normalise(quote).strip(" .,;:\"'")
    if len(q) < 8:
        return False
    if q in source_norm:
        return True
    # tolerate an ellipsis inside the quote: every part must be found in order
    parts = [p.strip(" .,;:\"'") for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q) if p.strip()]
    if len(parts) > 1:
        pos = 0
        for part in parts:
            if len(part) < 4:
                continue
            i = source_norm.find(part, pos)
            if i < 0:
                return False
            pos = i + len(part)
        return True
    return False


def parse_date(s: str) -> Optional[dt.date]:
    m = re.search(r"(\d{2})[-./](\d{2})[-./](\d{4})", s or "")
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Senate: list, project page, explanatory memorandum
# --------------------------------------------------------------------------

def http_get(session: Any, url: str, binary: bool = False) -> Any:
    time.sleep(REQUEST_PAUSE)
    r = session.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    if binary:
        return r.content
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def parse_senate_list(html: str) -> List[Dict[str, Any]]:
    """Parse the public-consultation table. Robust to layout: every table row
    that contains a 'Bnnn / yyyy' number is taken."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    items: Dict[str, Dict[str, Any]] = {}
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        row_text = " | ".join(cells)
        m = re.search(r"\bB\s*(\d{1,4})\s*/\s*(\d{4})\b", row_text)
        if not m:
            continue
        nr, year = int(m.group(1)), int(m.group(2))
        key = f"B{nr}/{year}"
        deadline = parse_date(row_text)
        cod = None
        for a in tr.find_all("a", href=True):
            mc = re.search(r"[?&]cod=(\d+)", a["href"], re.I)
            if mc:
                cod = mc.group(1)
                break
        desc = max(cells, key=len) if cells else ""
        desc = re.sub(r"\s*Opinii trimise\s*\(\d+\)\s*$", "", desc).strip()
        opinions = re.search(r"Opinii trimise\s*\((\d+)\)", row_text)
        items[key] = {
            "id": key, "nr": nr, "year": year, "cod": cod,
            "deadline": deadline.isoformat() if deadline else None,
            "title": desc, "opinions": int(opinions.group(1)) if opinions else None,
        }
    return sorted(items.values(), key=lambda x: (x["deadline"] or "9999", x["nr"]))


def find_em_link(html: str, page_url: str, nr: int) -> Optional[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(" ", strip=True).lower()
        if not re.search(r"\.pdf", href, re.I):
            continue
        full = urljoin(page_url, href)
        name = full.rsplit("/", 1)[-1].lower()
        score = 0
        if re.search(r"em\.pdf", name):
            score += 2
        if "expunere" in label:
            score += 2
        if f"b{nr:03d}" in name or f"b{nr}" in name:
            score += 1
        if score:
            candidates.append((score, full))
    own = [c for c in candidates if re.search(rf"b0*{nr}(?!\d)", c[1].rsplit("/", 1)[-1], re.I)]
    if not own:
        return None          # never take a document that belongs to another project
    own.sort(reverse=True)
    return own[0][1]


def find_initiators(html: str) -> str:
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    m = re.search(r"Ini[țţt]iator[i]?\s*:?\s*\n?(.{3,600}?)(?:\n[A-ZĂÂÎȘŞȚŢ][^\n]{0,40}:|\Z)", text, re.S)
    if not m:
        return ""
    snippet = re.sub(r"\s+", " ", m.group(1)).strip()
    return snippet[:400]


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def fetch_project(session: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in page URL, initiators, memorandum URL and text. Cached on disk."""
    nr, year = item["nr"], item["year"]
    pdf_path = CACHE / "pdf" / f"B{nr}_{year}_EM.pdf"
    txt_path = CACHE / "text" / f"B{nr}_{year}_EM.txt"
    meta_path = CACHE / "meta" / f"B{nr}_{year}.json"
    for p in (pdf_path, txt_path, meta_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    if meta_path.exists() and txt_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        item.update(meta)
        item["text"] = txt_path.read_text(encoding="utf-8")
        return item

    page_url = PROJECT_URL.format(cod=item["cod"], nr=nr, year=year) if item.get("cod") else None
    em_url = None
    item["page_url"] = page_url
    item["initiators"] = ""
    if page_url:
        try:
            html = http_get(session, page_url)
            em_url = find_em_link(html, page_url, nr)
            item["initiators"] = find_initiators(html)
        except Exception as exc:
            item["page_error"] = str(exc)
    if not em_url:
        em_url = EM_FALLBACK.format(year=year, yy=str(year)[2:], nr=nr)
    item["em_url"] = em_url

    try:
        data = http_get(session, em_url, binary=True)
        if not data.startswith(b"%PDF"):
            raise ValueError("fișierul descărcat nu este PDF")
        pdf_path.write_bytes(data)
        text = pdf_text(data)
    except Exception as exc:
        item["em_error"] = str(exc)
        text = ""

    item["text"] = text
    item["text_chars"] = len(text.strip())
    meta = {k: item.get(k) for k in ("page_url", "em_url", "initiators", "page_error", "em_error", "text_chars")}
    if text.strip():
        txt_path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return item


# --------------------------------------------------------------------------
# Model reading
# --------------------------------------------------------------------------

def _gemini_call(client: Any, user: str, with_schema: bool) -> Any:
    from google.genai import types

    if with_schema:
        cfg = types.GenerateContentConfig(system_instruction=INSTRUCTIONS,
                                          response_mime_type="application/json",
                                          response_json_schema=SCHEMA)
        contents = user
    else:
        cfg = types.GenerateContentConfig(system_instruction=INSTRUCTIONS,
                                          response_mime_type="application/json")
        contents = (user + "\n\nRăspunde cu un singur obiect JSON care respectă exact această schemă:\n"
                    + json.dumps(SCHEMA, ensure_ascii=False))
    return client.models.generate_content(model=MODEL, contents=contents, config=cfg)


def read_with_model(client: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    text = item["text"]
    truncated = len(text) > MAX_TEXT_CHARS
    body = text[:MAX_TEXT_CHARS]
    key = hashlib.sha256(f"{PROMPT_VERSION}|{MODEL}|{body}".encode("utf-8")).hexdigest()[:16]
    cache_path = CACHE / "lecturi" / f"B{item['nr']}_{item['year']}_{key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    user = (
        f"Proiect: B{item['nr']}/{item['year']}\n"
        f"Titlul din lista Senatului: {item.get('title', '')}\n\n"
        "Textul expunerii de motive (extras automat din PDF, poate conține greșeli de scanare; "
        "citează-le exact așa cum apar):\n\n" + body
    )
    time.sleep(MODEL_PAUSE)
    with_schema = True
    resp = None
    for attempt in range(4):
        try:
            resp = _gemini_call(client, user, with_schema)
            break
        except Exception as exc:
            msg = str(exc)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 3:
                if "per day" in msg.lower() or "perday" in msg.lower().replace(" ", ""):
                    raise RuntimeError("limita zilnică gratuită Gemini a fost atinsă; reia mâine "
                                       "(lecturile făcute deja rămân în cache)") from exc
                say("    limita pe minut a fost atinsă; aștept 65 de secunde")
                time.sleep(65)
                continue
            if any(k in msg for k in ("503", "UNAVAILABLE", "504", "DEADLINE", "timed out", "Timeout",
                                      "ReadTimeout", "ConnectTimeout")) and attempt < 3:
                say(f"    modelul este ocupat la Google; aștept {20 * (attempt + 1)} de secunde")
                time.sleep(20 * (attempt + 1))
                continue
            if with_schema and ("400" in msg or "INVALID_ARGUMENT" in msg) and "schema" in msg.lower():
                with_schema = False      # older models: schema in the prompt instead
                continue
            raise
    raw = (resp.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    reading = json.loads(raw)
    for k in ("registru_simbolic", "legitimare", "dispozitive_retorice"):
        reading.setdefault(k, [])
    reading.setdefault("comunitate_imaginata", {"noi": "", "ei_sau_ceilalti": "", "categorii_create": []})
    usage = getattr(resp, "usage_metadata", None)
    reading["_meta"] = {
        "model": MODEL, "prompt_version": PROMPT_VERSION, "schema_enforced": with_schema,
        "truncated": truncated, "date": dt.datetime.now().isoformat(timespec="seconds"),
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }
    cache_path.write_text(json.dumps(reading, ensure_ascii=False, indent=1), encoding="utf-8")
    return reading


def verify_quotes(reading: Dict[str, Any], text: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Keep only observations whose quote is found in the source text."""
    src = normalise(text)
    key = match_key(text)
    rejected: List[Dict[str, str]] = []
    for field in ("registru_simbolic", "legitimare", "dispozitive_retorice"):
        kept = []
        for obs in reading.get(field, []):
            status = quote_status(obs.get("citat", ""), src, key)
            if status:
                obs["_potrivire"] = status
                kept.append(obs)
            else:
                rejected.append({"sectiune": field, "citat": obs.get("citat", "")})
        reading[field] = kept
    return reading, rejected


# --------------------------------------------------------------------------
# Word report
# --------------------------------------------------------------------------

def write_report(items: List[Dict[str, Any]], skipped: List[Dict[str, Any]], dry_run: bool) -> Path:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)

    today = dt.date.today()
    title = doc.add_heading("Lectură antropologică a proiectelor de lege în consultare publică la Senat", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = doc.add_paragraph()
    r = p.add_run(f"Șerban, G. F. · {today.strftime('%d.%m.%Y')} · sursa: senat.ro, lista proiectelor "
                  "aflate în consultare publică și expunerile de motive")
    r.italic = True

    doc.add_heading("Cum se citește acest raport", 1)
    doc.add_paragraph(
        "Pentru fiecare proiect, expunerea de motive a fost descărcată de pe senat.ro și citită de un model "
        f"de limbaj ({MODEL}), cu instrucțiunea de a face o lectură antropologică descriptivă: pe cine "
        "construiește textul ca „noi” și ca „ei”, ce simboluri și ce memorie mobilizează, pe ce autoritate "
        "își sprijină argumentul și ce dispozitive retorice folosește. Fiecare observație are un citat exact. "
        "Programul a verificat automat că fiecare citat există în textul expunerii de motive; observațiile "
        "al căror citat nu a putut fi găsit au fost scoase din raport și sunt enumerate în anexă. Unde "
        "textul scanat are diacritice sau litere stricate, citatul este acceptat dacă se potrivește aproape "
        "literă cu literă și este marcat ca „potrivire aproximativă”.")
    doc.add_paragraph(
        "Lectura este asistată de AI și este descriptivă. Nu spune dacă o lege este bună sau rea și nu "
        "evaluează inițiatorii sau partidele. „Interesul antropologic” arată cât material oferă textul "
        "pentru o astfel de lectură, nu un risc. Recunoașterea dispozitivelor retorice de către modele de "
        "limbaj nu este validată pe limba română și depinde mult de prezența unor marcatori expliciți; "
        "o tehnică nesemnalată poate fi totuși prezentă.")

    if dry_run:
        doc.add_paragraph("Rulare de probă: expunerile de motive au fost descărcate, dar nu au fost citite "
                          "de model.").runs[0].bold = True

    analysed = [i for i in items if i.get("reading")]
    analysed.sort(key=lambda i: (i.get("deadline") or "9999",
                                 INTEREST_ORDER.get(i["reading"].get("interes_antropologic"), 3)))

    doc.add_heading("Pe scurt", 1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for c, t in zip(hdr, ["Proiect", "Ce propune", "Termen consultare", "Interes antropologic"]):
        c.text = t
    for it in (analysed if not dry_run else items):
        row = table.add_row().cells
        row[0].text = it["id"]
        row[1].text = (it["reading"]["rezumat"] if it.get("reading") else it.get("title", ""))[:300]
        d = it.get("deadline")
        row[2].text = dt.date.fromisoformat(d).strftime("%d.%m.%Y") if d else "—"
        row[3].text = INTEREST_LABEL.get(it["reading"]["interes_antropologic"], "—") if it.get("reading") else "—"

    for it in analysed:
        rd = it["reading"]
        doc.add_heading(f"{it['id']} — {it.get('title', '')[:160]}", 1)
        meta = doc.add_paragraph()
        d = it.get("deadline")
        bits = [f"Termen consultare: {dt.date.fromisoformat(d).strftime('%d.%m.%Y')}" if d else None,
                f"Inițiatori (din fișa Senatului): {it['initiators']}" if it.get("initiators") else None,
                f"Expunerea de motive: {it.get('em_url', '')}",
                f"Fișa proiectului: {it.get('page_url', '')}" if it.get("page_url") else None]
        meta.add_run("\n".join(b for b in bits if b)).font.size = Pt(9)

        doc.add_paragraph().add_run("Ce propune. ").bold = True
        doc.paragraphs[-1].add_run(rd["rezumat"])
        p = doc.add_paragraph()
        p.add_run(f"Interes antropologic: {INTEREST_LABEL.get(rd['interes_antropologic'], '')}. ").bold = True
        p.add_run(rd.get("motivul_interesului", ""))

        ci = rd.get("comunitate_imaginata", {})
        p = doc.add_paragraph()
        p.add_run("Comunitatea imaginată. ").bold = True
        parts = []
        if ci.get("noi"):
            parts.append(f"„Noi”: {ci['noi']}")
        if ci.get("ei_sau_ceilalti"):
            parts.append(f"„Ei” sau ceilalți: {ci['ei_sau_ceilalti']}")
        if ci.get("categorii_create"):
            parts.append("Categorii create sau delimitate: " + "; ".join(ci["categorii_create"]))
        p.add_run(". ".join(x.rstrip(".") for x in parts) + "." if parts else "Textul nu construiește explicit un „noi” și un „ei”.")

        def observations(head: str, obs: List[Dict[str, Any]], label_fn) -> None:
            if not obs:
                return
            doc.add_paragraph().add_run(head).bold = True
            for o in obs:
                q = doc.add_paragraph(style="List Bullet")
                q.add_run(label_fn(o) + ". ").bold = True
                q.add_run(o.get("descriere") or o.get("explicatie") or "")
                q.add_run(f" Citat: „{o['citat']}”").italic = True
                if o.get("_potrivire") == "aproximativ":
                    q.add_run(" (potrivire aproximativă cu textul scanat)").font.size = Pt(8)

        observations("Registrul simbolic", rd.get("registru_simbolic", []),
                     lambda o: o.get("tema", "").capitalize() or "Temă")
        observations("Legitimarea", rd.get("legitimare", []),
                     lambda o: LEGITIMATION_LABEL.get(o.get("sursa"), "Sursă"))
        observations("Dispozitive retorice", rd.get("dispozitive_retorice", []),
                     lambda o: TECHNIQUE_LABEL.get(o.get("tehnica"), "Tehnică")
                     + (" (marker explicit)" if o.get("marker_explicit") else " (implicit)"))

        p = doc.add_paragraph()
        p.add_run("Ce nu se poate spune din acest text. ").bold = True
        p.add_run(rd.get("ce_nu_se_poate_spune", ""))
        if rd.get("_meta", {}).get("truncated"):
            doc.add_paragraph(f"Atenție: expunerea de motive a fost tăiată la {MAX_TEXT_CHARS} de caractere "
                              "înainte de lectură.").runs[0].italic = True

    if skipped:
        doc.add_heading("Proiecte fără lectură", 1)
        doc.add_paragraph("Pentru aceste proiecte expunerea de motive nu a putut fi descărcată sau nu are "
                          "un strat de text care să poată fi citit (de exemplu, un document scanat ca imagine).")
        for it in skipped:
            doc.add_paragraph(f"{it['id']} — {it.get('title', '')[:160]}. Motiv: {it.get('skip_reason', '')}",
                              style="List Bullet")

    rejected_all = [(it["id"], r) for it in analysed for r in it.get("rejected", [])]
    doc.add_heading("Anexă: citate care nu au putut fi verificate", 1)
    if rejected_all:
        doc.add_paragraph("Modelul a propus aceste citate, dar ele nu apar exact în textul expunerii de motive. "
                          "Observațiile corespunzătoare au fost scoase din raport.")
        for pid, r in rejected_all:
            sect = {"registru_simbolic": "registrul simbolic", "legitimare": "legitimare",
                    "dispozitive_retorice": "dispozitive retorice"}.get(r["sectiune"], r["sectiune"])
            doc.add_paragraph(f"{pid} · {sect}: „{r['citat']}”", style="List Bullet")
    else:
        doc.add_paragraph("Toate citatele propuse de model au fost găsite în textele sursă.")

    tok_in = sum((it["reading"].get("_meta", {}).get("input_tokens") or 0) for it in analysed)
    tok_out = sum((it["reading"].get("_meta", {}).get("output_tokens") or 0) for it in analysed)
    foot = doc.add_paragraph()
    foot.add_run(f"Model: {MODEL} (Gemini API), versiunea instrucțiunilor {PROMPT_VERSION}. Tokeni: "
                 f"{tok_in} la intrare, {tok_out} la ieșire (lecturile luate din cache nu se mai cer "
                 "a doua oară).").font.size = Pt(8)
    foot.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    REPORTS.mkdir(exist_ok=True)
    suffix = "_proba" if dry_run else ""
    out = REPORTS / f"Lectura_antropologica_Senat_{today.isoformat()}{suffix}.docx"
    n = 2
    while out.exists():
        out = REPORTS / f"Lectura_antropologica_Senat_{today.isoformat()}{suffix}_{n}.docx"
        n += 1
    doc.save(out)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Lectură antropologică a proiectelor în consultare la Senat")
    ap.add_argument("--max", type=int, default=12, help="câte proiecte să citească modelul (implicit 12)")
    ap.add_argument("--dry-run", action="store_true", help="descarcă și citește PDF-urile, fără model")
    ap.add_argument("--only", help="un singur proiect, de exemplu B573")
    ap.add_argument("--probe", action="store_true", help="salvează paginile Senatului pentru diagnostic")
    ap.add_argument("--model", help="alt model Gemini gratuit, de exemplu gemini-3.5-flash-lite")
    args = ap.parse_args()
    global MODEL
    if args.model:
        MODEL = args.model

    try:
        import requests
        import bs4  # noqa: F401
        import pypdf  # noqa: F401
        import docx  # noqa: F401
    except ImportError as exc:
        say(f"Lipsește o bibliotecă ({exc.name}). Rulează:\n"
            "    python -m pip install --upgrade google-genai requests beautifulsoup4 pypdf python-docx")
        return 1

    session = requests.Session()

    say("Citesc lista proiectelor în consultare publică de pe senat.ro ...")
    try:
        list_html = http_get(session, SENATE_LIST)
    except Exception as exc:
        say(f"Nu am putut deschide {SENATE_LIST}: {exc}")
        return 1

    if args.probe:
        PROBE.mkdir(exist_ok=True)
        (PROBE / "senat_lista.html").write_text(list_html, encoding="utf-8")
        items = parse_senate_list(list_html)
        say(f"Am găsit {len(items)} proiecte în listă.")
        if items and items[0].get("cod"):
            url = PROJECT_URL.format(cod=items[0]["cod"], nr=items[0]["nr"], year=items[0]["year"])
            (PROBE / "senat_proiect.html").write_text(http_get(session, url), encoding="utf-8")
            say(f"Am salvat și fișa proiectului {items[0]['id']}.")
        say(f"Fișierele de diagnostic sunt în {PROBE}")
        return 0

    items = parse_senate_list(list_html)
    if not items:
        say("Nu am găsit niciun proiect în listă. Pagina Senatului s-a schimbat probabil; rulează cu --probe.")
        return 1
    today = dt.date.today().isoformat()
    open_items = [i for i in items if not i["deadline"] or i["deadline"] >= today]
    say(f"Am găsit {len(items)} proiecte în consultare, dintre care {len(open_items)} cu termenul încă deschis.")

    if args.only:
        want = args.only.upper().replace(" ", "").split("/")[0]
        open_items = [i for i in items if f"B{i['nr']}" == want]
        if not open_items:
            say(f"Proiectul {args.only} nu este în lista de consultare publică.")
            return 1
    selected = open_items[: max(1, args.max)]
    say(f"Citesc {len(selected)} proiecte, în ordinea termenelor.\n")

    client = None
    if not args.dry_run:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            say("GEMINI_API_KEY nu este setată. Fă o cheie gratuită în Google AI Studio, sau rulează cu --dry-run.")
            return 1
        try:
            from google import genai
        except ImportError:
            say("Lipsește biblioteca google-genai. Rulează:  python -m pip install --upgrade google-genai")
            return 1
        from google.genai import types

        client = genai.Client(http_options=types.HttpOptions(
            timeout=120_000,                                    # două minute pe cerere
            retry_options=types.HttpRetryOptions(attempts=2)))

    done, skipped = [], []
    for n, it in enumerate(selected, 1):
        say(f"[{n}/{len(selected)}] {it['id']} — {it['title'][:80]}")
        fetch_project(session, it)
        if len(it.get("text", "").strip()) < MIN_TEXT_CHARS:
            it["skip_reason"] = it.get("em_error") or "PDF fără text care să poată fi citit"
            say(f"    sărit: {it['skip_reason']}")
            skipped.append(it)
            continue
        say(f"    expunerea de motive: {len(it['text'])} caractere")
        if args.dry_run:
            done.append(it)
            continue
        try:
            reading = read_with_model(client, it)
        except Exception as exc:
            it["skip_reason"] = f"eroare la model: {exc}"
            say(f"    {it['skip_reason']}")
            skipped.append(it)
            continue
        reading, rejected = verify_quotes(reading, it["text"])
        it["reading"], it["rejected"] = reading, rejected
        kept = sum(len(reading.get(k, [])) for k in ("registru_simbolic", "legitimare", "dispozitive_retorice"))
        say(f"    interes {reading.get('interes_antropologic')}; {kept} observații cu citat verificat, "
            f"{len(rejected)} scoase")
        done.append(it)

    out = write_report(done, skipped, args.dry_run)
    say(f"\nRaportul este gata: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
