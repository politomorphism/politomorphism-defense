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
    python parlament_antropologic.py --model gemini-3.5-flash-lite   doar modelul ales, fara trecere automata

Daca modelul principal ramane aglomerat (eroarea 503), programul trece singur pe
gemini-3.5-flash-lite pentru proiectul respectiv si noteaza asta in raport.

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
PROMPT_VERSION = "2026-11-16.2"
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"   # folosit automat când modelul principal e aglomerat
AUTO_FALLBACK = True                       # oprit când utilizatorul alege singur --model

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
lectură antropologică DESCRIPTIVĂ a textului, pe variabilele de mai jos.

Variabilele lecturii:
1. Comunitatea imaginată: pe cine construiește textul ca „noi” și pe cine ca
   „ei” sau „ceilalți”; ce categorii de persoane creează, numește sau delimitează.
2. Granițe și apartenență: după ce criterii include sau exclude textul oameni
   (cetățenie, teritoriu, etnie sau limbă, vârstă, gen sau familie, ocupație,
   statut moral, venit sau proprietate).
3. Relația stat–cetățean: ce rol îi dă textul statului (protector, controlor,
   furnizor, arbitru, partener, administrator) și ce rol cetățeanului
   (beneficiar, contribuabil, suspect, victimă, consumator, participant,
   subiect disciplinat).
4. Distribuția puterii: cine câștigă și cine pierde putere, competențe sau
   control prin această lege (instituții, niveluri de guvernare, grupuri).
5. Economia morală: ce prezintă textul ca datorie, merit, dreptate, abuz sau
   privilegiu, reciprocitate, rușine sau onoare; cine datorează ce cui.
6. Temporalitatea: cum construiește textul timpul politic (criză și urgență,
   decădere și restaurare, progres și modernizare, tradiție și continuitate,
   normalitate tehnică).
7. Memoria și istoria: ce trecut invocă (regimuri, evenimente, figuri, traume).
8. Spațiul și teritoriul: la ce scară gândește comunitatea (local, regional,
   național, european, global) și ce opoziții spațiale folosește (centru și
   periferie, rural și urban, intern și extern).
9. Corpul și biopolitica: ce corpuri, comportamente, vârste, consumuri sau
   sănătate reglementează textul.
10. Registrul simbolic: simboluri și opoziții (pur/impur, local/străin,
    ordine/haos, sacru/profan, suveranitate/dependență).
11. Resemnificarea: un termen sau simbol politic căruia textul îi dă un sens nou
    ori diferit de cel obișnuit (de exemplu „suveranitate”, „familie”,
    „transparență”, „siguranță”).
12. Legitimarea: pe ce autoritate se sprijină argumentul.
13. Cunoașterea și dovezile: ce tip de cunoaștere invocă textul și dacă aduce
    date empirice verificabile sau doar afirmații.
14. Registrul afectiv dominant: frică sau amenințare, indignare, compasiune,
    mândrie, speranță sau ton neutru-tehnic.
15. Performativitatea: legea este preponderent instrumentală (schimbă reguli
    și resurse), mixtă sau preponderent simbolică și declarativă (afirmă
    valori, comemorează, semnalează).
16. Dispozitive retorice: falsă dilemă, presupoziție, cauzalitate implicită,
    generalizare, apel la frică, echivalență falsă, întrebare retorică,
    eufemism; pentru fiecare spui dacă are un marker lexical explicit sau dacă
    este implicit.
17. Tăcerile textului: categorii de oameni, grupuri sau consecințe pe care regula
    le atinge prin chiar formularea ei, dar pe care textul nu le numește. Aici
    nu există citat, pentru că e vorba de ce lipsește; explici din ce formulare
    a textului reiese tăcerea și de ce contează antropologic.

Reguli obligatorii:
- Fiecare observație care cere citat se sprijină pe un CITAT EXACT din text, de
  cel mult 25 de cuvinte, copiat literă cu literă, inclusiv greșelile de
  scanare, diacriticele și punctuația. Nu corecta, nu parafraza, nu combina
  fragmente. Dacă nu poți cita exact, nu face observația.
- Cel mult 3 observații pe fiecare listă. Listele pot rămâne goale; nu umple
  spațiul. Câmpurile care nu sunt liste se completează mereu; dacă textul nu
  oferă material, alegi varianta tehnică sau neutră și lași citatul gol.
- Ești strict neutru politic. Nu spui dacă legea este bună sau rea, nu evaluezi
  partide, inițiatori sau tabere, nu faci predicții și nu dai sfaturi. Aplici
  aceleași criterii oricărui text, indiferent de orientarea lui.
- Descrii ce face textul, nu intențiile autorilor. La tăceri descrii efectul
  formulării, nu motivul pentru care a fost aleasă, și nu numești partide decât
  dacă textul le numește.
- Un text tehnic poate avea totuși interes antropologic prin ce normalizează,
  ce universalizează sau ce trece sub tăcere; ține cont de asta la
  „interes_antropologic”, care arată materialul pentru lectură, nu un risc.
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
BOUNDARY = ["cetatenie", "teritoriu", "etnie_limba", "varsta", "gen_familie", "ocupatie_statut",
            "statut_moral", "venit_proprietate", "altele"]
STATE_ROLE = ["protector", "controlor", "furnizor", "arbitru", "partener", "administrator", "altele"]
CITIZEN_ROLE = ["beneficiar", "contribuabil", "suspect", "victima", "consumator", "participant",
                "subiect_disciplinat", "altele"]
MORAL = ["datorie", "merit", "dreptate_echitate", "abuz_privilegiu", "reciprocitate", "rusine_onoare",
         "altele"]
TIME = ["criza_urgenta", "decadere_restaurare", "progres_modernizare", "traditie_continuitate",
        "normalitate_tehnica", "altele"]
SCALE = ["local", "regional", "national", "european", "global", "centru_periferie", "rural_urban",
         "intern_extern"]
AFFECT = ["frica_amenintare", "indignare", "compasiune", "mandrie", "speranta", "neutru_tehnic", "altele"]
PERFORMATIVE = ["preponderent_instrumental", "mixt", "preponderent_simbolic_declarativ"]
KNOWLEDGE = ["statistici_studii", "comparatie_internationala", "jurisprudenta", "experienta_practica",
             "bun_simt_evidenta", "niciuna"]


def _s(desc: str = "") -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": "string"}
    if desc:
        d["description"] = desc
    return d


def _enum(values: List[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": values}


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _list(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "array", "items": _obj(props)}


QUOTE = _s("Citat exact din text, cel mult 25 de cuvinte; gol dacă nu există.")

SCHEMA: Dict[str, Any] = _obj({
    "rezumat": _s("Ce propune legea, în 1-2 fraze neutre."),
    "interes_antropologic": _enum(["ridicat", "mediu", "scazut"]),
    "motivul_interesului": _s(),
    "comunitate_imaginata": _obj({"noi": _s(), "ei_sau_ceilalti": _s(),
                                  "categorii_create": {"type": "array", "items": _s()}}),
    "granite_apartenenta": _list({"criteriu": _enum(BOUNDARY), "descriere": _s(), "citat": QUOTE}),
    "relatia_stat_cetatean": _obj({"statul_ca": _enum(STATE_ROLE), "cetateanul_ca": _enum(CITIZEN_ROLE),
                                   "descriere": _s(), "citat": QUOTE}),
    "distributia_puterii": _list({"castiga": _s(), "pierde": _s(), "descriere": _s(), "citat": QUOTE}),
    "economie_morala": _list({"tema": _enum(MORAL), "descriere": _s(), "citat": QUOTE}),
    "temporalitate": _obj({"tip": _enum(TIME), "descriere": _s(), "citat": QUOTE}),
    "memorie_istorie": _list({"descriere": _s(), "citat": QUOTE}),
    "spatiu_teritoriu": _list({"scara": _enum(SCALE), "descriere": _s(), "citat": QUOTE}),
    "corp_biopolitica": _list({"descriere": _s(), "citat": QUOTE}),
    "registru_simbolic": _list({"tema": _s(), "descriere": _s(), "citat": QUOTE}),
    "resemnificare": _list({"termen": _s(), "sens_nou": _s(), "citat": QUOTE}),
    "legitimare": _list({"sursa": _enum(LEGITIMATION), "descriere": _s(), "citat": QUOTE}),
    "cunoastere_dovezi": _obj({"date_empirice_invocate": {"type": "boolean"}, "tip": _enum(KNOWLEDGE),
                               "descriere": _s()}),
    "registru_afectiv": _obj({"dominant": _enum(AFFECT), "descriere": _s(), "citat": QUOTE}),
    "performativitate": _obj({"caracter": _enum(PERFORMATIVE), "justificare": _s()}),
    "dispozitive_retorice": _list({"tehnica": _enum(TECHNIQUES), "citat": QUOTE, "explicatie": _s(),
                                   "marker_explicit": {"type": "boolean"}}),
    "taceri": _list({"ce_lipseste": _s(), "de_unde_reiese": _s(), "de_ce_conteaza": _s()}),
    "ce_nu_se_poate_spune": _s("Ce nu se poate concluziona doar din acest text."),
})

# Fields whose observations carry a quote that must be verified.
QUOTED_LISTS = ["granite_apartenenta", "distributia_puterii", "economie_morala", "memorie_istorie",
                "spatiu_teritoriu", "corp_biopolitica", "registru_simbolic", "resemnificare",
                "legitimare", "dispozitive_retorice"]
QUOTED_OBJECTS = ["relatia_stat_cetatean", "temporalitate", "registru_afectiv"]

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
LABEL = {
    # boundaries
    "cetatenie": "Cetățenie", "teritoriu": "Teritoriu", "etnie_limba": "Etnie sau limbă", "varsta": "Vârstă",
    "gen_familie": "Gen sau familie", "ocupatie_statut": "Ocupație sau statut", "statut_moral": "Statut moral",
    "venit_proprietate": "Venit sau proprietate",
    # state and citizen
    "protector": "protector", "controlor": "controlor", "furnizor": "furnizor", "arbitru": "arbitru",
    "partener": "partener", "administrator": "administrator", "beneficiar": "beneficiar",
    "contribuabil": "contribuabil", "suspect": "suspect", "victima": "victimă", "consumator": "consumator",
    "participant": "participant", "subiect_disciplinat": "subiect disciplinat",
    # moral economy
    "datorie": "Datorie", "merit": "Merit", "dreptate_echitate": "Dreptate și echitate",
    "abuz_privilegiu": "Abuz sau privilegiu", "reciprocitate": "Reciprocitate", "rusine_onoare": "Rușine sau onoare",
    # time
    "criza_urgenta": "criză și urgență", "decadere_restaurare": "decădere și restaurare",
    "progres_modernizare": "progres și modernizare", "traditie_continuitate": "tradiție și continuitate",
    "normalitate_tehnica": "normalitate tehnică",
    # space
    "local": "Local", "regional": "Regional", "national": "Național", "european": "European", "global": "Global",
    "centru_periferie": "Centru și periferie", "rural_urban": "Rural și urban", "intern_extern": "Intern și extern",
    # affect
    "frica_amenintare": "frică sau amenințare", "indignare": "indignare", "compasiune": "compasiune",
    "mandrie": "mândrie", "speranta": "speranță", "neutru_tehnic": "neutru-tehnic",
    # performativity
    "preponderent_instrumental": "preponderent instrumentală", "mixt": "mixtă",
    "preponderent_simbolic_declarativ": "preponderent simbolică și declarativă",
    # knowledge
    "statistici_studii": "statistici sau studii", "comparatie_internationala": "comparații internaționale",
    "jurisprudenta": "jurisprudență", "experienta_practica": "experiență practică",
    "bun_simt_evidenta": "bun-simț sau „evidență”", "niciuna": "niciuna",
    "altele": "altele",
}
INTEREST_ORDER = {"ridicat": 0, "mediu": 1, "scazut": 2}
INTEREST_LABEL = {"ridicat": "ridicat", "mediu": "mediu", "scazut": "scăzut"}


def lab(v: Any) -> str:
    return LABEL.get(str(v), str(v).replace("_", " "))


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
    form_words = ("Indiferent", "Cuvinte în titlu", "Prima Cameră", "Fără paginare", "Nume inițiator")
    if any(w in snippet for w in form_words) or len(snippet) < 5:
        return ""
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

def _gemini_call(client: Any, model: str, user: str, with_schema: bool) -> Any:
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
    return client.models.generate_content(model=model, contents=contents, config=cfg)


class ModelBusy(RuntimeError):
    """The model stayed overloaded (503) or timed out after all retries."""


class DailyLimit(RuntimeError):
    """The free daily quota of this model is used up (Google counts it separately for each model)."""


EXHAUSTED: set = set()          # models whose daily quota ran out during this run


def _cache_path(item: Dict[str, Any], model: str, body: str) -> Path:
    key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{body}".encode("utf-8")).hexdigest()[:16]
    path = CACHE / "lecturi" / f"B{item['nr']}_{item['year']}_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_once(client: Any, model: str, item: Dict[str, Any], body: str, truncated: bool) -> Dict[str, Any]:
    cache_path = _cache_path(item, model, body)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    user = (
        f"Proiect: B{item['nr']}/{item['year']}\n"
        f"Titlul din lista Senatului: {item.get('title', '')}\n\n"
        "Textul expunerii de motive (extras automat din PDF, poate conține greșeli de scanare; "
        "citează-le exact așa cum apar):\n\n" + body
    )
    say(f"    citesc cu {model} ...")
    time.sleep(MODEL_PAUSE)
    with_schema = True
    resp = None
    busy_tries = 0
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = _gemini_call(client, model, user, with_schema)
            break
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if "per day" in msg.lower() or "perday" in msg.lower().replace(" ", ""):
                    EXHAUSTED.add(model)
                    raise DailyLimit(f"limita zilnică gratuită pentru {model} a fost atinsă") from exc
                if attempt <= 3:
                    say("    limita pe minut a fost atinsă; aștept 65 de secunde")
                    time.sleep(65)
                    continue
                raise
            if any(k in msg for k in ("503", "UNAVAILABLE", "504", "DEADLINE", "timed out", "Timeout",
                                      "ReadTimeout", "ConnectTimeout")):
                busy_tries += 1
                if busy_tries <= 3:
                    say(f"    {model} este aglomerat la Google; aștept {20 * busy_tries} de secunde")
                    time.sleep(20 * busy_tries)
                    continue
                raise ModelBusy(f"{model} a rămas aglomerat după trei încercări") from exc
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
        "model": model, "prompt_version": PROMPT_VERSION, "schema_enforced": with_schema,
        "truncated": truncated, "date": dt.datetime.now().isoformat(timespec="seconds"),
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }
    cache_path.write_text(json.dumps(reading, ensure_ascii=False, indent=1), encoding="utf-8")
    return reading


def read_with_model(client: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    """Read with the chosen model; if it stays overloaded, switch automatically to the lighter one."""
    text = item["text"]
    truncated = len(text) > MAX_TEXT_CHARS
    body = text[:MAX_TEXT_CHARS]
    can_fall_back = AUTO_FALLBACK and MODEL != FALLBACK_MODEL
    if MODEL not in EXHAUSTED or not can_fall_back:
        try:
            return _read_once(client, MODEL, item, body, truncated)
        except ModelBusy:
            if not can_fall_back:
                raise
            say(f"    trec automat pe modelul mai ușor {FALLBACK_MODEL}")
        except DailyLimit:
            if not can_fall_back:
                raise
            say(f"    limita zilnică pentru {MODEL} s-a terminat; trec pe {FALLBACK_MODEL}, "
                "care are limita lui separată")
    if FALLBACK_MODEL in EXHAUSTED:
        raise DailyLimit("limita zilnică gratuită a fost atinsă la ambele modele")
    return _read_once(client, FALLBACK_MODEL, item, body, truncated)


def verify_quotes(reading: Dict[str, Any], text: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Keep only quoted observations whose quote is found in the source text.
    For single objects the classification is kept but an unverified quote is removed."""
    src = normalise(text)
    key = match_key(text)
    rejected: List[Dict[str, str]] = []
    for field in QUOTED_LISTS:
        kept = []
        for obs in reading.get(field, []) or []:
            status = quote_status(obs.get("citat", ""), src, key)
            if status:
                obs["_potrivire"] = status
                kept.append(obs)
            else:
                rejected.append({"sectiune": field, "citat": obs.get("citat", "")})
        reading[field] = kept[:3]
    for field in QUOTED_OBJECTS:
        obj = reading.get(field) or {}
        q = (obj.get("citat") or "").strip()
        if q:
            status = quote_status(q, src, key)
            if status:
                obj["_potrivire"] = status
            else:
                rejected.append({"sectiune": field, "citat": q})
                obj["citat"] = ""
        reading[field] = obj
    reading["taceri"] = (reading.get("taceri") or [])[:3]
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

    def bold_lead(lead: str, text: str) -> None:
        p = doc.add_paragraph()
        p.add_run(lead).bold = True
        p.add_run(text)

    def quote_run(p: Any, o: Dict[str, Any]) -> None:
        if o.get("citat"):
            p.add_run(f" Citat: „{o['citat']}”").italic = True
            if o.get("_potrivire") == "aproximativ":
                p.add_run(" (potrivire aproximativă cu textul scanat)").font.size = Pt(8)

    def bullets(head: str, obs: List[Dict[str, Any]], label_fn, text_fn) -> None:
        if not obs:
            return
        doc.add_paragraph().add_run(head).bold = True
        for o in obs:
            q = doc.add_paragraph(style="List Bullet")
            lead = label_fn(o)
            if lead:
                q.add_run(lead + ". ").bold = True
            q.add_run(text_fn(o))
            quote_run(q, o)

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
        f"de limbaj Gemini (modelul folosit e trecut la fiecare proiect), cu instrucțiunea de a face o lectură antropologică descriptivă pe șaptesprezece "
        "variabile: comunitatea imaginată, granițele apartenenței, relația dintre stat și cetățean, distribuția "
        "puterii, economia morală, temporalitatea, memoria, spațiul, corpul, registrul simbolic, resemnificarea "
        "termenilor, legitimarea, cunoașterea invocată, registrul afectiv, performativitatea, dispozitivele "
        "retorice și tăcerile textului. Observațiile au citat exact. Programul a verificat automat că fiecare "
        "citat există în expunerea de motive; observațiile al căror citat nu a putut fi găsit au fost scoase și "
        "sunt enumerate în anexă. Unde textul scanat are diacritice stricate, citatul e acceptat dacă se "
        "potrivește aproape literă cu literă și e marcat ca „potrivire aproximativă”. Tăcerile nu au citat, "
        "pentru că descriu ce lipsește; sunt observații de context, nu constatări textuale.")
    doc.add_paragraph(
        "Lectura este asistată de AI și este descriptivă. Nu spune dacă o lege este bună sau rea și nu "
        "evaluează inițiatorii sau partidele. „Interesul antropologic” arată cât material oferă textul pentru o "
        "astfel de lectură, nu un risc. Recunoașterea dispozitivelor retorice de către modele de limbaj nu este "
        "validată pe limba română și depinde mult de prezența unor marcatori expliciți; o tehnică nesemnalată "
        "poate fi totuși prezentă.")
    light = [i["id"] for i in items
             if i.get("reading") and i["reading"].get("_meta", {}).get("model", MODEL) != DEFAULT_MODEL]
    if light:
        used = sorted({i["reading"]["_meta"].get("model", MODEL) for i in items
                       if i.get("reading") and i["id"] in light})
        warn = doc.add_paragraph()
        warn.add_run("Atenție la model. ").bold = True
        subj = (f"Proiectul {light[0]} a fost citit" if len(light) == 1
                else f"Proiectele {', '.join(light)} au fost citite")
        warn.add_run(f"{subj} cu {', '.join(used)}, nu cu modelul principal "
                     f"{DEFAULT_MODEL}, de obicei pentru că acesta era aglomerat la Google. Modelele mai ușoare "
                     "tind să vadă mai puțin din ce e implicit, așa că lectura pot fi mai superficiale. "
                     "La o rulare ulterioară programul încearcă din nou modelul principal; lecturile noi se fac "
                     "separat și nu le înlocuiesc pe cele vechi din cache.")

    if dry_run:
        doc.add_paragraph("Rulare de probă: expunerile de motive au fost descărcate, dar nu au fost citite "
                          "de model.").runs[0].bold = True

    analysed = [i for i in items if i.get("reading")]
    analysed.sort(key=lambda i: (i.get("deadline") or "9999",
                                 INTEREST_ORDER.get(i["reading"].get("interes_antropologic"), 3)))

    def fmt_date(d: Optional[str]) -> str:
        return dt.date.fromisoformat(d).strftime("%d.%m.%Y") if d else "—"

    doc.add_heading("Pe scurt", 1)
    if not analysed and not dry_run:
        doc.add_paragraph("În această rulare niciun proiect nu a putut fi citit de model. Motivele sunt mai jos, "
                          "la „Proiecte fără lectură”. Rulează programul din nou mai târziu.").runs[0].bold = True
    table = doc.add_table(rows=1, cols=4) if (analysed or dry_run) else None
    if table is not None:
        table.style = "Light Grid Accent 1"
        for c, t in zip(table.rows[0].cells, ["Proiect", "Ce propune", "Termen consultare",
                                              "Interes antropologic"]):
            c.text = t
    for it in ((analysed if not dry_run else items) if table is not None else []):
        row = table.add_row().cells  # type: ignore[union-attr]
        row[0].text = it["id"]
        row[1].text = (it["reading"]["rezumat"] if it.get("reading") else it.get("title", ""))[:300]
        row[2].text = fmt_date(it.get("deadline"))
        row[3].text = INTEREST_LABEL.get(it["reading"]["interes_antropologic"], "—") if it.get("reading") else "—"

    if analysed:
        doc.add_heading("Matricea comparativă", 1)
        doc.add_paragraph("Clasificările de mai jos permit compararea proiectelor între ele. Fiecare este explicată, "
                          "cu citat, în secțiunea proiectului.")
        mt = doc.add_table(rows=1, cols=7)
        mt.style = "Light Grid Accent 1"
        heads = ["Proiect", "Statul ca", "Cetățeanul ca", "Timpul politic", "Afect dominant",
                 "Caracterul legii", "Date empirice"]
        for c, t in zip(mt.rows[0].cells, heads):
            c.text = t
        for it in analysed:
            rd = it["reading"]
            sc = rd.get("relatia_stat_cetatean", {}) or {}
            cells = mt.add_row().cells
            vals = [it["id"], lab(sc.get("statul_ca", "")), lab(sc.get("cetateanul_ca", "")),
                    lab((rd.get("temporalitate") or {}).get("tip", "")),
                    lab((rd.get("registru_afectiv") or {}).get("dominant", "")),
                    lab((rd.get("performativitate") or {}).get("caracter", "")),
                    "da" if (rd.get("cunoastere_dovezi") or {}).get("date_empirice_invocate") else "nu"]
            for c, v in zip(cells, vals):
                c.text = v
                for para in c.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9)
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        for row in mt.rows:                      # keep each row on one page
            trPr = row._tr.get_or_add_trPr()
            el = OxmlElement("w:cantSplit")
            el.set(qn("w:val"), "true")
            trPr.append(el)

    for it in analysed:
        rd = it["reading"]
        doc.add_heading(f"{it['id']} — {it.get('title', '')[:160]}", 1)
        meta = doc.add_paragraph()
        bits = [f"Termen consultare: {fmt_date(it.get('deadline'))}",
                f"Inițiatori (din fișa Senatului): {it['initiators']}" if it.get("initiators") else None,
                f"Expunerea de motive: {it.get('em_url', '')}",
                f"Citit cu: {rd.get('_meta', {}).get('model', MODEL)}",
                f"Fișa proiectului: {it.get('page_url', '')}" if it.get("page_url") else None]
        meta.add_run("\n".join(b for b in bits if b)).font.size = Pt(9)

        bold_lead("Ce propune. ", rd.get("rezumat", ""))
        bold_lead(f"Interes antropologic: {INTEREST_LABEL.get(rd.get('interes_antropologic'), '')}. ",
                  rd.get("motivul_interesului", ""))

        doc.add_heading("Comunitate și apartenență", 2)
        ci = rd.get("comunitate_imaginata", {}) or {}
        parts = []
        if ci.get("noi"):
            parts.append(f"„Noi”: {ci['noi']}")
        if ci.get("ei_sau_ceilalti"):
            parts.append(f"„Ei” sau ceilalți: {ci['ei_sau_ceilalti']}")
        if ci.get("categorii_create"):
            parts.append("Categorii create sau delimitate: " + "; ".join(ci["categorii_create"]))
        bold_lead("Comunitatea imaginată. ", ". ".join(x.rstrip(".") for x in parts) + "." if parts
                  else "Textul nu construiește explicit un „noi” și un „ei”.")
        bullets("Granițele apartenenței", rd.get("granite_apartenenta", []),
                lambda o: lab(o.get("criteriu")), lambda o: o.get("descriere", ""))
        if rd.get("taceri"):
            doc.add_paragraph().add_run("Ce lasă textul nenumit").bold = True
            for t in rd["taceri"]:
                q = doc.add_paragraph(style="List Bullet")
                q.add_run(t.get("ce_lipseste", "").rstrip(".") + ". ").bold = True
                q.add_run(f"Reiese din: {t.get('de_unde_reiese', '').rstrip('.')}. "
                          f"De ce contează: {t.get('de_ce_conteaza', '')}")
                q.add_run(" (observație de context, fără citat)").font.size = Pt(8)

        doc.add_heading("Stat, putere și morală", 2)
        sc = rd.get("relatia_stat_cetatean", {}) or {}
        p = doc.add_paragraph()
        p.add_run("Relația stat–cetățean. ").bold = True
        p.add_run(f"Statul apare ca {lab(sc.get('statul_ca', ''))}, cetățeanul ca "
                  f"{lab(sc.get('cetateanul_ca', ''))}. {sc.get('descriere', '')}")
        quote_run(p, sc)
        bullets("Distribuția puterii", rd.get("distributia_puterii", []),
                lambda o: "",
                lambda o: f"Câștigă: {o.get('castiga', '—')}; pierde: {o.get('pierde', '—')}. {o.get('descriere', '')}")
        bullets("Economia morală", rd.get("economie_morala", []),
                lambda o: lab(o.get("tema")), lambda o: o.get("descriere", ""))

        doc.add_heading("Timp, memorie, spațiu și corp", 2)
        tp = rd.get("temporalitate", {}) or {}
        p = doc.add_paragraph()
        p.add_run("Timpul politic. ").bold = True
        p.add_run(f"{lab(tp.get('tip', '')).capitalize()}. {tp.get('descriere', '')}")
        quote_run(p, tp)
        bullets("Memoria și istoria", rd.get("memorie_istorie", []), lambda o: "", lambda o: o.get("descriere", ""))
        bullets("Spațiul și teritoriul", rd.get("spatiu_teritoriu", []),
                lambda o: lab(o.get("scara")), lambda o: o.get("descriere", ""))
        bullets("Corpul și biopolitica", rd.get("corp_biopolitica", []), lambda o: "", lambda o: o.get("descriere", ""))

        doc.add_heading("Simboluri, legitimare și cunoaștere", 2)
        bullets("Registrul simbolic", rd.get("registru_simbolic", []),
                lambda o: (o.get("tema", "") or "Temă").capitalize(), lambda o: o.get("descriere", ""))
        bullets("Resemnificarea termenilor", rd.get("resemnificare", []),
                lambda o: f"„{o.get('termen', '')}”", lambda o: o.get("sens_nou", ""))
        bullets("Legitimarea", rd.get("legitimare", []),
                lambda o: LEGITIMATION_LABEL.get(o.get("sursa"), "Sursă"), lambda o: o.get("descriere", ""))
        kd = rd.get("cunoastere_dovezi", {}) or {}
        bold_lead("Cunoașterea invocată. ",
                  f"Date empirice verificabile: {'da' if kd.get('date_empirice_invocate') else 'nu'}; tipul "
                  f"principal: {lab(kd.get('tip', ''))}. {kd.get('descriere', '')}")

        doc.add_heading("Afect, performativitate și retorică", 2)
        af = rd.get("registru_afectiv", {}) or {}
        p = doc.add_paragraph()
        p.add_run("Registrul afectiv. ").bold = True
        p.add_run(f"Dominant: {lab(af.get('dominant', ''))}. {af.get('descriere', '')}")
        quote_run(p, af)
        pf = rd.get("performativitate", {}) or {}
        bold_lead("Caracterul legii. ", f"{lab(pf.get('caracter', '')).capitalize()}. {pf.get('justificare', '')}")
        bullets("Dispozitive retorice", rd.get("dispozitive_retorice", []),
                lambda o: TECHNIQUE_LABEL.get(o.get("tehnica"), "Tehnică")
                + (" (marker explicit)" if o.get("marker_explicit") else " (implicit)"),
                lambda o: o.get("explicatie", ""))

        bold_lead("Ce nu se poate spune din acest text. ", rd.get("ce_nu_se_poate_spune", ""))
        if rd.get("_meta", {}).get("truncated"):
            doc.add_paragraph(f"Atenție: expunerea de motive a fost tăiată la {MAX_TEXT_CHARS} de caractere "
                              "înainte de lectură.").runs[0].italic = True

    if skipped:
        doc.add_heading("Proiecte fără lectură", 1)
        doc.add_paragraph("Pentru aceste proiecte lectura nu a putut fi făcută: expunerea de motive nu a putut "
                          "fi descărcată, nu are text care să poată fi citit, sau modelul nu a răspuns.")
        for it in skipped:
            doc.add_paragraph(f"{it['id']} — {it.get('title', '')[:160]}. Motiv: {it.get('skip_reason', '')}",
                              style="List Bullet")

    rejected_all = [(it["id"], r) for it in analysed for r in it.get("rejected", [])]
    if analysed:
        doc.add_heading("Anexă: citate care nu au putut fi verificate", 1)
    if not analysed:
        pass
    elif rejected_all:
        doc.add_paragraph("Modelul a propus aceste citate, dar ele nu apar în textul expunerii de motive. "
                          "Observațiile corespunzătoare au fost scoase din raport (la clasificările unice s-a "
                          "păstrat doar clasificarea, fără citat).")
        for pid, r in rejected_all:
            doc.add_paragraph(f"{pid} · {r['sectiune'].replace('_', ' ')}: „{r['citat']}”", style="List Bullet")
    else:
        doc.add_paragraph("Toate citatele propuse de model au fost găsite în textele sursă.")

    tok_in = sum((it["reading"].get("_meta", {}).get("input_tokens") or 0) for it in analysed)
    tok_out = sum((it["reading"].get("_meta", {}).get("output_tokens") or 0) for it in analysed)
    foot = doc.add_paragraph()
    models_used = sorted({it["reading"].get("_meta", {}).get("model", MODEL) for it in analysed}) or [MODEL]
    foot.add_run(f"Model: {', '.join(models_used)} (Gemini API), versiunea instrucțiunilor {PROMPT_VERSION}. Tokeni: "
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
    global MODEL, AUTO_FALLBACK
    if args.model:
        MODEL = args.model
        AUTO_FALLBACK = False

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
        except DailyLimit as exc:
            it["skip_reason"] = (f"{exc}; rulează din nou mâine (lecturile făcute deja rămân în cache "
                                 "și nu se mai cer a doua oară)")
            say(f"    {it['skip_reason']}")
            skipped.append(it)
            continue
        except ModelBusy:
            it["skip_reason"] = ("modelele Gemini gratuite au fost aglomerate la Google în acest moment; "
                                 "rulează din nou mai târziu")
            say(f"    {it['skip_reason']}")
            skipped.append(it)
            continue
        except Exception as exc:
            it["skip_reason"] = f"eroare la model: {str(exc)[:200]}"
            say(f"    {it['skip_reason']}")
            skipped.append(it)
            continue
        reading, rejected = verify_quotes(reading, it["text"])
        it["reading"], it["rejected"] = reading, rejected
        kept = sum(len(reading.get(k, []) or []) for k in QUOTED_LISTS)
        say(f"    interes {reading.get('interes_antropologic')}; {kept} observații cu citat verificat, "
            f"{len(rejected)} scoase")
        done.append(it)

    out = write_report(done, skipped, args.dry_run)
    say(f"\nRaportul este gata: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
