// make_dfa_note.js — builds Nota_Componenta_DFA_RO_EN.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const FONT = "Calibri";
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 140, line: 288 },
  alignment: opts.align || AlignmentType.JUSTIFIED,
  children: [new TextRun({ text, font: FONT, size: 22, italics: !!opts.italics })],
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });

const W = [3000, 2400, 1800, 1800]; // sums to 9000
const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
const cell = (t, i, head) => new TableCell({
  width: { size: W[i], type: WidthType.DXA },
  borders: { top: border, bottom: border, left: border, right: border },
  shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF4" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 20, bold: !!head })] })],
});
const table = (rows) => new Table({
  width: { size: 9000, type: WidthType.DXA },
  columnWidths: W,
  rows: rows.map((r, k) => new TableRow({ children: r.map((t, i) => cell(t, i, k === 0)) })),
});

const RO_HEAD = ["Măsură (backend rules)", "Set", "Valoare", "n"];
const EN_HEAD = ["Measure (rules backend)", "Set", "Value", "n"];
const rowsFor = (dev, held) => [
  ["Detecție pe itemi problematici", dev, "1.000", "24/24"],
  ["Detecție pe itemi problematici", held, "0.000", "0/13"],
  ["Alarme false pe negative dificile", dev, "0.333", "2/6"],
  ["Alarme false pe negative dificile", held, "0.000", "0/5"],
  ["Acuratețe pe perechi", dev, "1.000", "20/20"],
  ["Acuratețe pe perechi", held, "0.000", "0/10"],
];
const roRows = [RO_HEAD, ...rowsFor("dezvoltare (in-sample)", "held-out")];
const enRows = [EN_HEAD, ...rowsFor("development (in-sample)", "held-out").map((r) => [
  r[0].replace("Detecție pe itemi problematici", "Detection on problematic items")
      .replace("Alarme false pe negative dificile", "False alarms on hard negatives")
      .replace("Acuratețe pe perechi", "Pair accuracy"),
  r[1], r[2], r[3]])];

const ro = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Componenta DFA a detectorului defensiv")] }),
  P("Notă tehnică bilingvă · Șerban, G. F. · 11 noiembrie 2026 · ORCID 0009-0000-2266-3356", { align: AlignmentType.LEFT, italics: true }),
  H1("1. Ce adaugă componenta"),
  P("În ziua în care Comisia Europeană a prezentat propunerea de Digital Fairness Act, stratul defensiv al Politomorphism primește o componentă nouă, dfa/30_dfa_classifier.py. Ea citește text adresat consumatorului (microcopy de interfață, bannere de ofertă, condiții de abonament, postări de influenceri, etichete din magazinele de jocuri, mesaje de customer service) și numește practica relevantă pentru DFA pe care o performează textul comerciantului, împreună cu grupul de prevederi din propunere sub care cade. Taxonomia are zece categorii: confirmshaming, urgență sau raritate falsă, conversia automată a perioadei de probă, anularea obstrucționată, drip pricing, reclamă nedeclarată la influenceri, opacitatea monedei virtuale, recompense variabile fără probabilități, lipsa contactului uman și monetizarea țintită spre minori, plus etichetele none și insufficient_evidence."),
  P("Componenta preia neschimbat principiul din clasificatorul v2: se evaluează doar practica pe care textul însuși o performează față de consumator. Un text care raportează, citează, critică, neagă sau descrie o practică (o știre, un regulator, un rezumat de politică publică) primește none. Rezultatele se numără separat (detecție, categorie exactă, categorie greșită, ratare, abținere; respectiv none corect, alarmă falsă, abținere), iar abținerile rămân în numitor."),
  H1("2. Limitele asumate"),
  P("DFA este drept al consumatorului (B2C) și nu reglementează discursul politic; publicitatea politică plătită ține de Regulamentul (UE) 2024/900. Componenta auditează deci text comercial. Legătura cu programul Politomorphism este metodologică și trece prin brațul de evaluare a sistemelor AI: aceeași componentă poate măsura dacă un sistem AI generează microcopy sau răspunsuri de customer service care reproduc o practică vizată de DFA."),
  P("Trimiterile la articole urmează draftul scurs pe 9 noiembrie 2026 și sunt grupuri de prevederi, nu citări juridice; trebuie verificate pe textul oficial COM înainte de orice publicare. O etichetă este un semnal de triere pentru un evaluator uman, nu o constatare că o practică e ilegală. Urgența și raritatea falsă depind de fapte din afara textului (dacă cronometrul e real, dacă stocul chiar e mic), așa că taxonomia le marchează requires_verification, iar abținerea este acceptată ca răspuns corect pentru ele."),
  H1("3. Primul rezultat: baseline-ul lexical nu generalizează"),
  P("Backend-ul implicit, rules, este un set transparent de reguli pe cuvinte-cheie, offline, fără cheie API. Pe setul de dezvoltare (20 de perechi potrivite lexical și 10 itemi dificili) a obținut detecție completă și acuratețe completă pe perechi. Acest rezultat este în-eșantion și nu dovedește nimic: regulile au fost scrise văzând exact acei itemi. Le-am înghețat (hash SHA-256 în dfa/rules_freeze.sha256) și am scris un set held-out separat, cu 10 perechi și 8 itemi dificili în formulări diferite, rulat o singură dată, fără nicio modificare ulterioară a codului."),
  table(roRows),
  P(""),
  P("Pe held-out, regulile ratează toți cei 13 itemi problematici. Faptul că nu produc alarme false acolo nu e un merit, ci consecința faptului că nu declanșează nimic. Concluzia este aceeași ca în Working Paper III: un verificator DFA bazat pe vocabular atinge scoruri perfecte pe propriul set și se prăbușește la parafrază. Rezervă importantă: setul held-out a fost scris de același autor care cunoștea vocabularul regulilor, deci este un test de stres la parafrază, nu un eșantion din text real, iar căderea la zero nu estimează rata reală de detecție."),
  H1("4. Pașii următori"),
  P("Primul pas este rularea backend-urilor LLM (openai sau ollama) pe ambele seturi, judecate față de cifrele de mai sus. Al doilea este adnotarea independentă a celor două seturi. Al treilea este un eșantion de text real (pagini de abonament, magazine din jocuri, postări de influenceri), evaluat în modul audit. Al patrulea este alinierea trimiterilor la textul oficial al propunerii, odată verificat."),
  new Paragraph({ children: [new PageBreak()] }),
];

const en = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("The DFA Component of the Defensive Detector")] }),
  P("Bilingual technical note · Șerban, G. F. · 11 November 2026 · ORCID 0009-0000-2266-3356", { align: AlignmentType.LEFT, italics: true }),
  H1("1. What the component adds"),
  P("On the day the European Commission presented its Digital Fairness Act proposal, the Politomorphism defensive layer gains a new component, dfa/30_dfa_classifier.py. It reads consumer-facing text (interface microcopy, offer banners, subscription terms, influencer posts, in-game store labels, customer-service notices) and names which DFA-relevant practice the trader's own text performs, together with the provision group of the proposal it falls under. The taxonomy has ten categories: confirmshaming, false urgency or scarcity, automatic trial conversion, obstructed cancellation, drip pricing, undisclosed influencer advertising, virtual-currency opacity, undisclosed variable rewards, no human contact and minor-targeted monetisation, plus the labels none and insufficient_evidence."),
  P("The component carries over, unchanged, the principle of the v2 classifier: only the practice the text itself performs towards the consumer is scored. Text that reports, quotes, criticises, negates or describes a practice (a news item, a regulator, a policy summary) is none. Outcomes are counted separately (detection, exact category, wrong category, missed, abstained; and correct none, false alarm, abstained), and abstentions stay in the denominator."),
  H1("2. Stated limits"),
  P("The DFA is consumer (B2C) law and does not regulate political speech; paid political advertising falls under Regulation (EU) 2024/900. The component therefore audits commercial text. Its link to the Politomorphism programme is methodological and runs through the AI-evaluation arm: the same component can measure whether an AI system generates interface copy or customer-service answers that reproduce a practice the DFA targets."),
  P("Article references follow the draft leaked on 9 November 2026 and are provision groups, not legal citations; they must be checked against the official COM text before any publication. A label is a screening signal for a human reviewer, not a finding that a practice is unlawful. False urgency and scarcity depend on facts outside the text (whether a timer is real, whether stock is really low), so the taxonomy marks them requires_verification and accepts abstention as correct handling."),
  H1("3. First result: the lexical baseline does not generalise"),
  P("The default backend, rules, is a transparent keyword rule set that runs offline without an API key. On the development set (20 vocabulary-matched pairs and 10 hard items) it reached full detection and full pair accuracy. That result is in-sample and proves nothing: the rules were written while looking at those very items. They were then frozen (SHA-256 hash in dfa/rules_freeze.sha256), and a separate held-out set of 10 pairs and 8 hard items in different wording was written and run once, with no later change to the code."),
  table(enRows),
  P(""),
  P("On the held-out set the rules miss all 13 problematic items. Their zero false-alarm rate there is not a merit; it follows from their firing on nothing. The conclusion matches Working Paper III: a vocabulary-based DFA checker scores perfectly on its own set and collapses under paraphrase. An important caveat is that the held-out set was written by the same author, who knew the rule vocabulary, so it is a paraphrase stress test rather than a sample of real text, and the drop to zero does not estimate a real-world detection rate."),
  H1("4. Next steps"),
  P("First, run the LLM backends (openai or ollama) on both sets and judge them against the figures above. Second, obtain independent annotation of both sets. Third, evaluate a sample of real text (subscription pages, in-game stores, influencer posts) in audit mode. Fourth, align the article references to the official text of the proposal once verified."),
];

const doc = new Document({
  creator: "Șerban, G. F.",
  title: "Componenta DFA / The DFA Component",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 36, bold: true, color: "1F3A5F" },
        paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "1F3A5F" },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1300, bottom: 1300, left: 1450, right: 1450 } } },
    children: [...ro, ...en],
  }],
});

const out = path.join(__dirname, "Nota_Componenta_DFA_RO_EN.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
