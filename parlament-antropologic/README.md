# Lectură antropologică a proiectelor de lege (Senatul României)

`parlament_antropologic.py` citește lista oficială a proiectelor aflate în consultare
publică la Senat (senat.ro), descarcă expunerea de motive a fiecăruia, o trimite unui
model Gemini (nivel gratuit) pentru o lectură antropologică descriptivă pe șaptesprezece
variabile — comunitatea imaginată, granițele apartenenței, relația stat–cetățean,
distribuția puterii, economia morală, temporalitatea, memoria, spațiul, corpul, registrul
simbolic, resemnificarea termenilor, legitimarea, cunoașterea invocată, registrul afectiv,
performativitatea, dispozitivele retorice și tăcerile textului — cu citat exact acolo unde
e cazul. Programul verifică automat că fiecare citat există în textul sursă și scrie un
raport Word în folderul `rapoarte/`, cu o matrice care compară proiectele între ele.

Lectura este asistată de AI și descriptivă: nu spune dacă o lege este bună sau rea și nu
evaluează inițiatori sau partide. Recunoașterea dispozitivelor retorice de către modele
de limbaj nu este validată pe limba română.

## Rulare

Necesită Python 3.9+ și

```
pip install --upgrade google-genai requests beautifulsoup4 pypdf python-docx
```

și o cheie Gemini din Google AI Studio în variabila de mediu `GEMINI_API_KEY`
(cheia nu se păstrează niciodată în acest repository).

```
python parlament_antropologic.py --dry-run    # descarcă și citește PDF-urile, fără model
python parlament_antropologic.py              # raport pentru cel mult 12 proiecte
python parlament_antropologic.py --only B573  # un singur proiect
python parlament_antropologic.py --model gemini-3.5-flash-lite   # alt model gratuit
python parlament_antropologic.py --probe      # salvează paginile Senatului pentru diagnostic
```

Folderele `cache/`, `rapoarte/` și `probe/` se creează lângă script și nu se publică.
Deocamdată programul acoperă doar Senatul; Camera Deputaților nu este inclusă.

Autor: Șerban, G. F. · ORCID 0009-0000-2266-3356 · Cod: MIT.
