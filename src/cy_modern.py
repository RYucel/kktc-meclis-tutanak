"""Extract, repair, parse, and build modern & 1977-1979 Republic of Cyprus (parliament.cy) plenary minutes.

Handles three distinct PDF generations from `parliament.cy`:
1. 1977-1979 (`KC:3`, Γ΄ Περίοδος): Polytonic Greek with embedded FineReader OCR (`HiddenHorzOCR`)
   plus Tesseract `ell` OCR fallback for scanned-only sittings.
2. 2002-2004 (`KC:8`, Η΄ Περίοδος): Clean born-digital Modern Greek PDFs.
3. 2015-2017 (`KC:10` Ι΄ Περίοδος & `KC:11` ΙΑ΄ Περίοδος): Word-2007 TrueType `Identity-H` PDFs with
   corrupted `/ToUnicode` CMaps — repaired losslessly in memory from embedded `/FontFile2` `cmap` tables.

Outputs KKTC-compatible `documents.jsonl`, `speeches.jsonl`, `rosters.json`, `aliases.json`, and `summary.json`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from cy_parse import script_of, shape, strip_accents
from cy_translate import load_cache, lookup_translation

pymupdf.TOOLS.mupdf_display_errors(False)

FOLDER = "Kıbrıs Cumhuriyeti Temsilciler Meclisi"
LANG_LABEL = {"el": "Yunanca", "tr": "Türkçe", "mixed": "Yunanca/Türkçe"}
TEXT_LABEL = "Tutanak Metni"

# Explicit canonical overrides so well-known Greek Cypriot leaders & returning 1960-64 MPs
# match standard Turkish historiography AND link across 1960-64 / 1977-79 / 2002-04 / 2015-17 careers.
CANONICAL_NAME_OVERRIDES: dict[str, str] = {
    # Returning 1960-64 MPs & 1977-79 leaders (exact match to cy_parse.GREEK_MEMBERS where applicable)
    "ΚΛΗΡΙΔΗΣ ΓΛΑΥΚΟΣ": "Glafkos Klerides",
    "ΚΥΠΡΙΑΝΟΥ ΣΠΥΡΟΣ": "Spiros Kiprianu",
    "ΛΥΣΣΑΡΙΔΗΣ ΒΑΣΟΣ": "Vasos Lisaridis",
    "ΠΑΠΑΪΩΑΝΝΟΥ ΕΖΕΚΙΑΣ": "Ezekias Papayoannu",
    "ΖΙΑΡΤΙΔΗΣ ΑΝΔΡΕΑΣ": "Andreas Ziartidis",
    "ΓΙΑΓΚΟΥ ΑΝΔΡΕΑΣ": "Andreas Yangu",
    "ΠΑΡΑΣΚΕΥΑΪΔΗΣ ΦΕΙΔΙΑΣ": "Fidias Paraskevaidis",
    "ΠΑΡΑΣΚΕΥΑΙΔΗΣ ΦΕΙΔΙΑΣ": "Fidias Paraskevaidis",
    "ΠΟΣΚΩΤΗΣ ΣΤΑΥΡΟΣ": "Stavros Poskotis",
    "ΠΟΤΑΜΙΤΗΣ ΓΙΑΓΚΟΣ": "Con Potamitis",
    "ΠΟΤΑΜΙΤΗΣ ΙΩΑΝΝΗΣ": "Con Potamitis",
    "ΜΙΧΑΗΛΙΔΗΣ ΧΑΜΠΗΣ": "Haralambos Mihailidis",
    "ΜΙΧΑΗΛΙΔΗΣ ΧΑΡΑΛΑΜΠΟΣ": "Haralambos Mihailidis",
    "ΜΙΧΑΗΛΙΔΗΣ ΑΛΕΚΟΣ": "Alekos Mihailidis",
    "ΠΑΠΑΔΟΠΟΥΛΟΣ ΤΑΣΣΟΣ": "Tasos Papadopulos",
    "ΠΑΠΑΔΟΠΟΥΛΛΟΣ ΤΑΣΣΟΣ": "Tasos Papadopulos",
    "ΓΑΛΑΝΟΣ ΑΛΕΞΗΣ": "Aleksis Galanos",
    "ΛΑΔΑΣ ΓΕΩΡΓΙΟΣ": "Yorgos Ladas",
    "ΛΑΔΑΣ ΓΙΩΡΓΟΣ": "Yorgos Ladas",
    "ΧΑΤΖΗΔΗΜΗΤΡΙΟΥ ΤΑΚΗΣ": "Takis Hacıdimitriu",
    "ΧΡΙΣΤΟΦΙΔΗΣ ΧΡΙΣΤΟΦΟΡΟΣ": "Hristoforos Hristofidis",
    "ΑΜΠΙΖΑΣ ΣΤΑΥΡΟΣ": "Stavros Ambizas",
    "ΦΑΝΤΗΣ ΑΝΔΡΕΑΣ": "Andreas Fandis",
    "ΤΟΥΜΑΖΗΣ ΑΝΤΩΝΗΣ": "Antonis Tumazis",
    "ΑΣΤΖΙΑΝ ΑΝΤΡΑΝΙΚ": "Antranik Aşciyan",
    "ΠΑΥΛΑΚΗΣ ΠΑΥΛΟΣ": "Pavlos Pavlakis",
    "ΝΕΜΙΤΣΑΣ ΤΑΚΗΣ": "Takis Nemitsas",
    "ΔΗΜΗΤΡΙΟΥ ΜΙΣΙΑΟΥΛΗ ΣΤΕΛΛΑ": "Stella Dimitriu Misiauli",
    "ΚΟΥΚΟΥΜΑ ΚΟΥΤΡΑ ΣΚΕΥΗ": "Skevi Kukuma Kutra",
    # 2001-2006 & 2011-2021 party leaders, presidents, speakers, prominent MPs
    "ΧΡΙΣΤΟΦΙΑΣ ΔΗΜΗΤΡΗΣ": "Dimitris Hristofyas",
    "ΑΝΑΣΤΑΣΙΑΔΗΣ ΝΙΚΟΣ": "Nikos Anastasiadis",
    "ΚΥΠΡΙΑΝΟΥ ΑΝΤΡΟΣ": "Andros Kiprianu",
    "ΚΥΠΡΙΑΝΟΥ ΑΝΔΡΕΑΣ": "Andreas Kiprianu",
    "ΚΥΠΡΙΑΝΟΥ ΜΑΡΚΟΣ": "Markos Kiprianu",
    "ΝΕΟΦΥΤΟΥ ΑΒΕΡΩΦ": "Averof Neofitu",
    "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΣ": "Nikolas Papadopulos",
    "ΣΥΛΛΟΥΡΗΣ ΔΗΜΗΤΡΗΣ": "Dimitris Silluris",
    "ΟΜΗΡΟΥ ΓΙΑΝΝΑΚΗΣ": "Yanakis Omiru",
    "ΚΑΡΟΓΙΑΝ ΜΑΡΙΟΣ": "Marios Karoyan",
    "ΛΙΛΛΗΚΑΣ ΓΙΩΡΓΟΣ": "Yorgos Lillikas",
    "ΣΙΖΟΠΟΥΛΟΣ ΜΑΡΙΝΟΣ": "Marinos Sizopulos",
    "ΠΕΡΔΙΚΗΣ ΓΙΩΡΓΟΣ": "Yorgos Perdikis",
    "ΚΑΤΣΟΥΡΙΔΗΣ ΝΙΚΟΣ": "Nikos Katsuridis",
    "ΤΟΡΝΑΡΙΤΗΣ ΝΙΚΟΣ": "Nikos Tornaritis",
    "ΠΡΟΔΡΟΜΟΥ ΠΡΟΔΡΟΜΟΣ": "Prodromos Prodromu",
    "ΜΑΥΡΟΥ ΕΛΕΝΗ": "Eleni Mavru",
    "ΒΑΣΙΛΕΙΟΥ ΑΝΔΡΟΥΛΑ": "Andrula Vasiliu",
    "ΘΕΟΧΑΡΟΥΣ ΕΛΕΝΗ": "Eleni Teoharus",
    "ΔΗΜΗΤΡΙΟΥ ΑΝΝΙΤΑ": "Annita Dimitriu",
    "ΔΗΜΗΤΡΙΟΥ ΔΗΜΗΤΡΗΣ": "Dimitris Dimitriu",
    "ΔΗΜΗΤΡΙΟΥ ΠΑΝΑΓΙΩΤΗΣ": "Panayotis Dimitriu",
    "ΧΑΡΑΛΑΜΠΙΔΟΥ ΕΙΡΗΝΗ": "İrini Haralambidu",
    "ΕΡΩΤΟΚΡΙΤΟΥ ΧΡΙΣΤΙΑΝΑ": "Hristiana Erotokritu",
    "ΕΡΩΤΟΚΡΙΤΟΥ ΡΙΚΚΟΣ": "Rikkos Erotokritu",
    "ΣΤΕΦΑΝΟΥ ΣΤΕΦΑΝΟΣ": "Stefanos Stefanu",
    "ΔΑΜΙΑΝΟΥ ΑΡΙΣΤΟΣ": "Aristos Damianu",
    "ΛΟΥΚΑΪΔΗΣ ΓΙΩΡΓΟΣ": "Yorgos Lukaidis",
    "ΛΟΥΚΑΙΔΗΣ ΓΙΩΡΓΟΣ": "Yorgos Lukaidis",
    "ΚΟΥΛΙΑΣ ΖΑΧΑΡΙΑΣ": "Zaharias Kulias",
    "ΑΔΑΜΟΥ ΑΔΑΜΟΣ": "Adamos Adamu",
    "ΜΑΤΣΑΚΗΣ ΜΑΡΙΟΣ": "Marios Matsakis",
    "ΠΟΥΡΓΟΥΡΙΔΗΣ ΧΡΗΣΤΟΣ": "Hristos Purguridis",
    "ΠΟΥΡΓΟΥΡΙΔΗΣ ΧΡΙΣΤΟΣ": "Hristos Purguridis",
    "ΚΛΗΡΙΔΗΣ ΧΡΙΣΤΟΣ": "Hristos Klerides",
    "ΚΛΗΡΙΔΟΥ ΚΑΙΤΗ": "Keti Kleridu",
    "ΧΡΙΣΤΟΥ ΧΡΙΣΤΟΣ": "Hristos Hristu",
    "ΧΡΙΣΤΟΥ ΑΝΔΡΕΑΣ": "Andreas Hristu",
    "ΘΕΟΠΕΜΠΤΟΥ ΧΑΡΑΛΑΜΠΟΣ": "Haralambos Teopemptu",
    "ΜΥΛΩΝΑΣ ΠΑΥΛΟΣ": "Pavlos Milonas",
    "ΘΕΟΛΟΓΟΥ ΑΝΝΑ": "Anna Teologu",
    "ΧΡΥΣΟΣΤΟΜΟΥ ΑΡΙΣΤΟΣ": "Aristos Hrisostomu",
    "ΦΥΤΤΗΣ ΣΟΦΟΚΛΗΣ": "Sofoklis Fittis",
    "ΜΑΧΤΕΣΙΑΝ ΒΑΡΤΚΕΣ": "Vartkes Mahtesyan",
    "ΚΑΛΑΪΤΖΙΑΝ ΠΕΤΡΟΣ": "Petros Kalayciyan",
    "ΚΑΛΑΙΤΖΙΑΝ ΠΕΤΡΟΣ": "Petros Kalayciyan",
    "ΑΤΤΑΛΙΔΟΥ ΑΛΕΞΑΝΔΡΑ": "Aleksandra Attalidu",
    "ΓΕΩΡΓΙΑΔΗΣ ΧΑΡΗΣ": "Haris Yeoryiadis",
    "ΤΣΙΡΙΔΟΥ ΦΩΤΕΙΝΗ": "Fotini Tsiridu",
    "ΤΡΥΦΩΝΙΔΗΣ ΑΛΕΚΟΣ": "Alekos Trifonidis",
    "ΟΡΦΑΝΙΔΟΥ ΣΑΒΙΑ": "Savia Orfanidu",
    "ΔΙΠΛΑΡΟΣ ΕΥΘΥΜΙΟΣ": "Efthimios Diplaros",
    "ΠΑΠΑΔΟΥΡΗΣ ΣΤΑΥΡΟΣ": "Stavros Papaduris",
    "ΑΛΑΜΠΡΙΤΗΣ ΠΡΟΔΡΟΜΟΣ": "Prodromos Alambritis",
}

# English/international spellings for 0ms autocomplete discoverability in speakers_index.json
INTERNATIONAL_ALIASES: dict[str, list[str]] = {
    "Glafkos Klerides": ["glafcos clerides", "glafkos kliridis"],
    "Spiros Kiprianu": ["spyros kyprianou", "spiros kyprianou"],
    "Tasos Papadopulos": ["tassos papadopoulos", "tasos papadopoulos"],
    "Dimitris Hristofyas": ["demetris christofias", "dimitris christofias", "hristofias"],
    "Nikos Anastasiadis": ["nicos anastasiades", "nikos anastasiades"],
    "Andros Kiprianu": ["andros kyprianou"],
    "Averof Neofitu": ["averof neophytou", "averof neofytou"],
    "Nikolas Papadopulos": ["nicholas papadopoulos", "nikolas papadopoulos"],
    "Dimitris Silluris": ["demetris syllouris", "dimitris syllouris"],
    "Yanakis Omiru": ["yiannakis omirou", "giannakis omirou"],
    "Vasos Lisaridis": ["vassos lyssarides", "vasos lyssarides"],
    "Aleksis Galanos": ["alexis galanos"],
    "Alekos Mihailidis": ["alecos michaelides", "alekos michaelides"],
    "Markos Kiprianu": ["marcos kyprianou", "markos kyprianou"],
    "Yorgos Lillikas": ["giorgos lillikas", "george lillikas"],
    "Marinos Sizopulos": ["marinos sizopoulos"],
    "Yorgos Perdikis": ["giorgos perdikis", "george perdikis"],
    "Annita Dimitriu": ["annita demetriou", "annita dimitriou"],
    "İrini Haralambidu": ["irene charalambides", "eirini charalambidou"],
    "Hristiana Erotokritu": ["christiana erotokritou"],
    "Eleni Mavru": ["eleni mavrou"],
    "Andrula Vasiliu": ["androulla vassiliou"],
    "Eleni Teoharus": ["eleni theocharous"],
    "Stefanos Stefanu": ["stefanos stefanou"],
    "Nikos Katsuridis": ["nicos katsourides", "nikos katsouridis"],
    "Nikos Tornaritis": ["nicos tornaritis"],
    "Prodromos Prodromu": ["prodromos prodromou"],
    "Takis Hacıdimitriu": ["takis hadjidemetriou"],
    "Con Potamitis": ["yangos potamitis", "yiangos potamitis", "yorgos potamitis"],
}

# Known non-speaker structural headings in modern & 1977-79 minutes
NON_SPEAKER_PREFIXES = (
    "ΗΜΕΡΗΣΙΑ ΔΙΑΤΑΞ",
    "ΠΕΡΙΕΧΟΜΕΝΑ",
    "ΠΑΡΟΝΤΕΣ",
    "ΑΠΟΝΤΕΣ",
    "ΚΕΦΑΛΑΙΟ",
    "ΕΚΘΕΣΗ",
    "ΕΚΘΕΣΙΣ",
    "ΜΗ ΜΕΛΗ",
    "ΑΝΤΙΠΡΟΣΩΠΟΙ ΘΡΗΣΚΕΥΤΙΚΩΝ",
    "ΘΕΜΑ",
    "ΕΡΩΤΗΣΗ",
    "ΕΡΩΤΗΣΙΣ",
    "ΑΠΑΝΤΗΣΗ",
    "ΑΠΑΝΤΗΣΙΣ",
    "ΣΗΜΕΙΩΣΗ",
    "ΣΗΜΕΙΩΣΙΣ",
    "ΠΙΝΑΚΑΣ",
    "ΠΑΡΑΡΤΗΜΑ",
    "ΝΟΜΟΣΧΕΔΙ",
    "ΠΡΟΤΑΣΗ ΝΟΜΟΥ",
    "ΠΡΟΤΑΣΙΣ ΝΟΜΟΥ",
    "ΚΑΤΑΘΕΣΗ",
    "ΚΑΤΑΘΕΣΙΣ",
    "ΣΥΝΟΛΟ",
    "ΤΑΞΕΩΣ",
    "ΙΚΕΦΑΛΑΙΟΝ",
    "ΦΥΣΙΚΩΝ ΠΟΡΩΝ",
)

# Garbage words captured in OCR attendance scans that are NOT person names
_ROSTER_BAD_SUBSTR = (
    "ΒΟΥΛΕΥΤ", "ΠΑΡΟΝΤ", "ΑΠΟΝΤ", "ΑΔΕΙΑ", "ΕΙΔΙΚ", "ΑΡΡΩΣΤ", "ΑΣΘΕΝ",
    "ΔΙΚΑΙΟΛΟΓ", "ΕΞΩΤΕΡΙΚ", "ΥΠΗΡΕΣ", "ΑΝΤΙΠΡΟΣΩΠ", "ΟΜΑΔ", "ΚΑΤΑΘΕΣ",
    "ΝΟΜΟΣΧΕΔ", "ΕΠΙΤΡΟΠ", "ΦΥΣΙΚ", "ΡΥΘΜΙΣ", "ΠΕΡΙΕΧΟΜ", "ΣΥΝΕΔΡ",
    "ΚΕΦΑΛΑΙ", "ΕΚΘΕΣ", "ΥΠΟΥΡΓ", "ΠΡΟΕΔΡ", "ΓΡΑΜΜΑΤ", "ΤΜΗΜ", "ΔΗΜΟΚΡΑΤ",
    "ΤΥΠΟΓΡΑΦ", "ΛΕΥΚΩΣ", "ΑΜΜΟΧΩΣΤ", "ΛΕΜΕΣ", "ΛΑΡΝΑΚ", "ΚΕΡΥΝ", "ΠΑΦΟ",
    "ΔΥΝΑΜΕΙΣ", "ΣΥΜΜΑΧΙΑ", "ΕΚΠΑΙΔΕΥΤ", "ΔΗΜΟΣΙΑΣ", "ΣΥΣΤΗΜΑΤ", "ΥΓΕΙΑΣ",
    "ΕΘΝΙΚ", "ΚΟΙΝΩΝΙΚ", "ΑΡΙΣΤΕΡ", "ΣΥΝΕΡΓΑΣ", "ΚΙΝΗΜΑ", "ΠΑΡΑΤΑΞ",
    "ΚΟΜΜΑ", "ΣΟΣΙΑΛΙΣΤ", "ΠΟΛΙΤΩΝ", "ΑΝΕΞΑΡΤΗΤ", "ΛΑΙΚ",
)

_VOICELESS = set("πτκφθχσςξψΠΤΚΦΘΧΣΞΨ")
_FRONT_VOWELS = set("εέιίηήυύϊΐϋΰΕΈΙΊΗΉΥΎΪΫ")


def clean_greek_token(s: str) -> str:
    """Normalize OCR-corrupted Greek word to clean uppercase Modern Greek letters."""
    # Fix common 1977 FineReader polytonic/Latin look-alike glitches inside Greek words
    s = (
        s.replace("ϊ", "Ι").replace("ϋ", "Υ").replace("ΐ", "Ι").replace("ΰ", "Υ")
        .replace("!Ξ", "Ξ").replace("ί:", "Σ").replace("c:", "ς").replace("r", "τ")
    )
    out = []
    for ch in strip_accents(s):
        u = ch.upper()
        if "Α" <= u <= "Ω":
            out.append(u)
        elif ch in (" ", "-"):
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def transliterate_greek_word(word: str) -> str:
    """Deterministic Modern Greek -> Turkish phonetic transliteration for a single capitalized word."""
    w = strip_accents(word.strip()).upper()
    w = re.sub(r"[^Α-Ω]", "", w)
    if not w:
        return ""
    # Prefix Χατζη- -> Hacı- (matches 1960-64 Hacıharos convention)
    out: list[str] = []
    i = 0
    if w.startswith("ΧΑΤΖΗ") and len(w) > 6:
        out.append("hacı")
        i = 5
    n = len(w)
    while i < n:
        rem = w[i:]
        nxt = w[i + 2] if i + 2 < n else ""
        if rem.startswith("ΟΥ"):
            out.append("u")
            i += 2
            continue
        if rem.startswith("ΑΙ"):
            out.append("e")
            i += 2
            continue
        if rem.startswith(("ΕΙ", "ΟΙ")):
            out.append("i")
            i += 2
            continue
        if rem.startswith("ΑΥ"):
            out.append("af" if (not nxt or nxt in _VOICELESS) else "av")
            i += 2
            continue
        if rem.startswith("ΕΥ"):
            out.append("ef" if (not nxt or nxt in _VOICELESS) else "ev")
            i += 2
            continue
        if rem.startswith("ΜΠ"):
            out.append("b" if i == 0 else "mb")
            i += 2
            continue
        if rem.startswith("ΝΤ"):
            out.append("d" if i == 0 else "nd")
            i += 2
            continue
        if rem.startswith(("ΓΚ", "ΓΓ")):
            out.append("g" if i == 0 else "ng")
            i += 2
            continue
        if rem.startswith("ΤΣ"):
            out.append("ts")
            i += 2
            continue
        if rem.startswith("ΤΖ"):
            out.append("c")
            i += 2
            continue
        if rem.startswith(("ΓΙ", "ΓΥ")) and nxt and nxt in "ΑΕΗΙΟΥΩ":
            out.append("y")
            i += 2
            continue
        ch = w[i]
        nxt1 = w[i + 1] if i + 1 < n else ""
        # Deduplicate geminate consonants except λ/ν/ρ where double l/n is standard (Lillikas, Silluris, Yannakis, Annita)
        if nxt1 == ch and ch in "ΚΜΠΣΤΦΧ":
            i += 1
            continue
        if ch == "Γ":
            out.append("y" if nxt1 in "ΕΗΙΥ" else "g")
        elif ch == "Θ":
            out.append("t")
        elif ch == "Δ":
            out.append("d")
        elif ch == "Β":
            out.append("v")
        elif ch == "Χ":
            out.append("h")
        elif ch == "Φ":
            out.append("f")
        elif ch == "Ξ":
            out.append("ks")
        elif ch == "Ψ":
            out.append("ps")
        elif ch == "Ζ":
            out.append("z")
        elif ch == "Σ":
            out.append("s")
        elif ch == "Κ":
            out.append("k")
        elif ch == "Λ":
            out.append("l")
        elif ch == "Μ":
            out.append("m")
        elif ch == "Ν":
            out.append("n")
        elif ch == "Π":
            out.append("p")
        elif ch == "Ρ":
            out.append("r")
        elif ch == "Τ":
            out.append("t")
        elif ch == "Α":
            out.append("a")
        elif ch == "Ε":
            out.append("e")
        elif ch in ("Η", "Ι", "Υ"):
            out.append("i")
        elif ch in ("Ο", "Ω"):
            out.append("o")
        i += 1
    raw = "".join(out)
    if not raw:
        return ""
    first = "İ" if raw[0] == "i" else raw[0].upper()
    return first + raw[1:]


def greek_fullname_to_turkish(surname_first_el: str) -> str:
    """Convert Greek 'Surname FirstName' (as printed in attendance rolls) into canonical 'FirstName Surname'."""
    cleaned = clean_greek_token(surname_first_el)
    if cleaned in CANONICAL_NAME_OVERRIDES:
        return CANONICAL_NAME_OVERRIDES[cleaned]
    parts = [p for p in cleaned.split() if len(p) >= 2]
    if len(parts) < 2:
        return transliterate_greek_word(cleaned)
    # Attendance lists print: Surname [SecondSurname] FirstName
    first_el = parts[-1]
    sur_el = " ".join(parts[:-1])
    rev_key = f"{sur_el} {first_el}"
    if rev_key in CANONICAL_NAME_OVERRIDES:
        return CANONICAL_NAME_OVERRIDES[rev_key]
    # Check if already FirstName Surname order
    fwd_key = f"{parts[0]} {' '.join(parts[1:])}"
    if fwd_key in CANONICAL_NAME_OVERRIDES:
        return CANONICAL_NAME_OVERRIDES[fwd_key]
    first_tr = transliterate_greek_word(first_el)
    sur_tr = " ".join(transliterate_greek_word(p) for p in parts[:-1])
    return f"{first_tr} {sur_tr}".strip()


def needs_tounicode_repair(sample_text: str) -> bool:
    if "\u03a2" in sample_text:
        return True
    return any(
        tok in sample_text
        for tok in ("ΒΟΤΛ", "Πξαθ", "Πραθηηθ", "ΠΡΟΒΑΡ", "ΠΡΟΔΓΡ", "ΓΡΏΜΜ", "΢ΤΝΟ", "Αληηπξνζ")
    )


def fix_broken_tounicode(doc: pymupdf.Document) -> int:
    """Rebuild broken Word-2007 Type0 `/ToUnicode` streams in memory from embedded `/FontFile2` TTF `cmap`."""
    fixed = 0
    seen_tu: set[int] = set()
    for xref in range(1, doc.xref_length()):
        obj = doc.xref_object(xref)
        if "/Subtype /Type0" not in obj or "/ToUnicode" not in obj:
            continue
        m_tu = re.search(r"/ToUnicode\s+(\d+)\s+0\s+R", obj)
        m_df = re.search(r"/DescendantFonts\s+(\d+)\s+0\s+R", obj)
        if not m_tu or not m_df:
            continue
        tu_xref = int(m_tu.group(1))
        if tu_xref in seen_tu:
            continue
        df_arr = doc.xref_object(int(m_df.group(1)))
        m_cid = re.search(r"(\d+)\s+0\s+R", df_arr)
        if not m_cid:
            continue
        cid_obj = doc.xref_object(int(m_cid.group(1)))
        m_fd = re.search(r"/FontDescriptor\s+(\d+)\s+0\s+R", cid_obj)
        if not m_fd:
            continue
        fd_obj = doc.xref_object(int(m_fd.group(1)))
        m_ff = re.search(r"/FontFile2\s+(\d+)\s+0\s+R", fd_obj)
        if not m_ff:
            continue
        ttf_bytes = doc.xref_stream(int(m_ff.group(1)))
        fnt = pymupdf.Font(fontbuffer=ttf_bytes)
        gid_to_uni: dict[int, int] = {}
        for cp in range(0x0020, 0x2200):
            if 0x0370 <= cp <= 0x03FF or cp < 0x0370 or cp >= 0x1F00:
                g = fnt.has_glyph(cp)
                if g and g not in gid_to_uni:
                    gid_to_uni[g] = cp
        # Ensure standard Modern Greek block wins over technical symbols (Δ vs ∆, μ vs µ, Ω vs Ω)
        for cp in range(0x0384, 0x03CF):
            g = fnt.has_glyph(cp)
            if g:
                gid_to_uni[g] = cp
        if not gid_to_uni:
            continue
        lines = [
            "/CIDInit /ProcSet findresource begin",
            "12 dict begin",
            "begincmap",
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
            "/CMapName /Adobe-Identity-UCS def",
            "/CMapType 2 def",
            "1 begincodespacerange",
            "<0000> <FFFF>",
            end_cs := "endcodespacerange",
        ]
        _ = end_cs
        items = sorted(gid_to_uni.items())
        for idx in range(0, len(items), 100):
            chunk = items[idx : idx + 100]
            lines.append(f"{len(chunk)} beginbfchar")
            for gid, cp in chunk:
                lines.append(f"<{gid:04X}> <{cp:04X}>")
            lines.append("endbfchar")
        lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
        doc.update_stream(tu_xref, "\n".join(lines).encode("ascii"))
        seen_tu.add(tu_xref)
        fixed += 1
    return fixed


def extract_sitting_pages(pdf_path: Path, ocr_root: Path | None = None) -> tuple[list[str], str]:
    """Return `(pages_text, text_source)` for one sitting PDF."""
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)
    sample = "\n".join(doc[i].get_text("text") for i in range(min(4, n_pages)))
    if len(sample.strip()) < 100:
        doc.close()
        if ocr_root is not None:
            pdir = ocr_root / pdf_path.stem
            if pdir.exists():
                pages = []
                for pno in range(1, n_pages + 1):
                    pf = pdir / f"p{pno:04d}.txt"
                    pages.append(pf.read_text(encoding="utf-8") if pf.exists() else "")
                if sum(len(x.strip()) for x in pages) >= 100:
                    return pages, "pdf_tesseract_ell"
        return ["" for _ in range(n_pages)], "pdf_empty"

    if needs_tounicode_repair(sample):
        nfix = fix_broken_tounicode(doc)
        if nfix > 0:
            raw_bytes = doc.tobytes()
            doc.close()
            doc = pymupdf.open("pdf", raw_bytes)
            pages = [doc[i].get_text("text") for i in range(len(doc))]
            doc.close()
            return pages, "pdf_tounicode_repaired"

    # Distinguish 1977-79 FineReader OCR layer from native digital Unicode PDF
    fonts0 = doc.get_page_fonts(0) if n_pages else []
    is_hidden_ocr = any("HiddenHorzOCR" in str(f) for f in fonts0)
    pages = [doc[i].get_text("text") for i in range(n_pages)]
    doc.close()
    return pages, ("pdf_embedded_ocr" if is_hidden_ocr else "pdf_digital")


def determine_donem(period_folder: str, iso_date: str) -> int:
    if period_folder == "denktas_makarios_1977_1979":
        return 3
    if period_folder == "annan_2002_2004":
        return 8
    # May 22, 2016 legislative election -> 11th House (ΙΑ΄ Περίοδος) convened June 2, 2016
    return 10 if iso_date < "2016-06-01" else 11


_SYNODOS_MAP = {"Α": 1, "A": 1, "Β": 2, "B": 2, "Γ": 3, "Δ": 4, "Ε": 5, "E": 5, "ΣΤ": 6}


def extract_header_meta(first_page_text: str, file_title: str) -> tuple[int | None, int | None]:
    """Extract `(yasama_yili, birlesim_no)` from page 1 header or original filename."""
    head = "\n".join(first_page_text.splitlines()[:20])
    yy = None
    m_syn = re.search(r"ΣΥΝΟΔΟΣ\s+([ΑΒΓΔΕAΒBΕEΣΤ]{1,2})[΄'’\b\s]", strip_accents(head).upper())
    if m_syn:
        yy = _SYNODOS_MAP.get(m_syn.group(1))
    elif "ΕΚΤΑΚΤ" in strip_accents(head).upper() or "EKTAKTH" in head.upper():
        yy = 6
    birl = None
    m_ar = re.search(r"(?:Αρ\.|Άρ\.|AR\.|[ΙI][Α-ΩA-Z]?[΄'’])\s*(\d{1,3})\b", head + "\n" + file_title, re.I)
    if m_ar:
        birl = int(m_ar.group(1))
    return yy, birl


_END_SITTING_RE = re.compile(r"[\(\[]?\s*['’΄]?Ώρα\s+λή[ξζ!Ι][εη]\w*", re.I)
_TOC_RE = re.compile(r"^\s*ΠΕΡΙΕΧΟΜΕΝΑ\s*$")


def harvest_attendance_names(pages: list[str]) -> list[str]:
    """Extract raw Greek 'Surname FirstName' pairs from the post-sitting attendance roll."""
    tail = "\n".join(pages[max(0, len(pages) - 8) :])
    m_end = _END_SITTING_RE.search(tail)
    if not m_end:
        return []
    att = tail[m_end.end() :]
    m_toc = re.search(r"\bΠΕΡΙΕΧΟΜΕΝΑ\b", att)
    if m_toc:
        att = att[: m_toc.start()]
    found: list[str] = []
    # Match Greek Title-Case 2-word (or 3-word compound surname) sequences
    pat = re.compile(
        r"\b([Α-ΩΆ-ΏἈ-῿][α-ωά-ώἀ-ῷϊϋΐΰ]{2,20}\s+[Α-ΩΆ-ΏἈ-῿][α-ωά-ώἀ-ῷϊϋΐΰ]{2,20}(?:\s+(?:Στέλλα|Σκεύη))?)\b"
    )
    for ln in att.splitlines():
        s = re.sub(r"\s+", " ", ln.strip())
        for m in pat.finditer(s):
            raw_pair = m.group(1)
            up = clean_greek_token(raw_pair)
            if len(up.split()) not in (2, 3):
                continue
            if any(bad in up for bad in _ROSTER_BAD_SUBSTR):
                continue
            found.append(up)
    return found


class PeriodSpeakerResolver:
    """Resolves Greek floor labels (`Γ. ΠΕΡΔΙΚΗΣ`, `ΑΝΤΡΟΣ ΚΥΠΡΙΑΝΟΥ`, `ΠΡΟΕΔΡΟΣ`) per parliamentary period."""

    def __init__(self, donem: int, roster_el_pairs: list[str]):
        self.donem = donem
        self.members_canon: list[str] = []
        self.by_exact_el: dict[str, str] = {}
        # (surname_shape, first_initial_shape, full_first_shape, canonical_tr)
        self.shapes: list[tuple[str, str, str, str]] = []

        # Seed with overrides active in all/specific periods + harvested attendance pairs
        seed_pairs = list(roster_el_pairs)
        if donem == 3:
            seed_pairs.extend([
                "ΚΥΠΡΙΑΝΟΥ ΣΠΥΡΟΣ", "ΜΙΧΑΗΛΙΔΗΣ ΑΛΕΚΟΣ", "ΛΥΣΣΑΡΙΔΗΣ ΒΑΣΟΣ",
                "ΠΑΠΑΔΟΠΟΥΛΟΣ ΤΑΣΣΟΣ", "ΓΑΛΑΝΟΣ ΑΛΕΞΗΣ", "ΠΟΤΑΜΙΤΗΣ ΓΙΑΓΚΟΣ",
                "ΧΑΤΖΗΔΗΜΗΤΡΙΟΥ ΤΑΚΗΣ", "ΠΑΠΑΪΩΑΝΝΟΥ ΕΖΕΚΙΑΣ", "ΖΙΑΡΤΙΔΗΣ ΑΝΔΡΕΑΣ",
                "ΓΙΑΓΚΟΥ ΑΝΔΡΕΑΣ", "ΠΑΡΑΣΚΕΥΑΪΔΗΣ ΦΕΙΔΙΑΣ", "ΠΟΣΚΩΤΗΣ ΣΤΑΥΡΟΣ",
                "ΑΜΠΙΖΑΣ ΣΤΑΥΡΟΣ", "ΧΡΙΣΤΟΦΙΔΗΣ ΧΡΙΣΤΟΦΟΡΟΣ", "ΜΙΧΑΗΛΙΔΗΣ ΧΑΜΠΗΣ",
                "ΛΑΔΑΣ ΓΕΩΡΓΙΟΣ", "ΦΑΝΤΗΣ ΑΝΔΡΕΑΣ", "ΤΟΥΜΑΖΗΣ ΑΝΤΩΝΗΣ",
                "ΑΣΤΖΙΑΝ ΑΝΤΡΑΝΙΚ", "ΠΑΥΛΑΚΗΣ ΠΑΥΛΟΣ", "ΝΕΜΙΤΣΑΣ ΤΑΚΗΣ",
            ])
        elif donem == 8:
            seed_pairs.extend(["ΧΡΙΣΤΟΦΙΑΣ ΔΗΜΗΤΡΗΣ", "ΠΑΠΑΔΟΠΟΥΛΟΣ ΤΑΣΣΟΣ", "ΑΝΑΣΤΑΣΙΑΔΗΣ ΝΙΚΟΣ", "ΛΥΣΣΑΡΙΔΗΣ ΒΑΣΟΣ"])
        elif donem == 10:
            seed_pairs.extend(["ΟΜΗΡΟΥ ΓΙΑΝΝΑΚΗΣ", "ΝΕΟΦΥΤΟΥ ΑΒΕΡΩΦ", "ΚΥΠΡΙΑΝΟΥ ΑΝΤΡΟΣ", "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΣ", "ΣΥΛΛΟΥΡΗΣ ΔΗΜΗΤΡΗΣ"])
        elif donem == 11:
            seed_pairs.extend(["ΣΥΛΛΟΥΡΗΣ ΔΗΜΗΤΡΗΣ", "ΝΕΟΦΥΤΟΥ ΑΒΕΡΩΦ", "ΚΥΠΡΙΑΝΟΥ ΑΝΤΡΟΣ", "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΣ", "ΔΗΜΗΤΡΙΟΥ ΑΝΝΙΤΑ"])
        elif donem == 12:
            seed_pairs.extend([
                "ΔΗΜΗΤΡΙΟΥ ΑΝΝΙΤΑ", "ΚΟΥΛΙΑΣ ΖΑΧΑΡΙΑΣ", "ΝΕΟΦΥΤΟΥ ΑΒΕΡΩΦ", "ΣΤΕΦΑΝΟΥ ΣΤΕΦΑΝΟΣ",
                "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΛΑΣ", "ΧΡΙΣΤΟΥ ΧΡΙΣΤΟΣ", "ΣΙΖΟΠΟΥΛΟΣ ΜΑΡΙΝΟΣ", "ΚΑΡΟΓΙΑΝ ΜΑΡΙΟΣ",
                "ΘΕΟΠΕΜΠΤΟΥ ΧΑΡΑΛΑΜΠΟΣ", "ΑΤΤΑΛΙΔΟΥ ΑΛΕΞΑΝΔΡΑ", "ΛΟΥΚΑΪΔΗΣ ΓΙΩΡΓΟΣ", "ΔΑΜΙΑΝΟΥ ΑΡΙΣΤΟΣ",
                "ΧΑΡΑΛΑΜΠΙΔΟΥ ΕΙΡΗΝΗ", "ΕΡΩΤΟΚΡΙΤΟΥ ΧΡΙΣΤΙΑΝΑ", "ΓΕΩΡΓΙΑΔΗΣ ΧΑΡΗΣ", "ΤΟΡΝΑΡΙΤΗΣ ΝΙΚΟΣ",
                "ΔΙΠΛΑΡΟΣ ΕΥΘΥΜΙΟΣ", "ΤΣΙΡΙΔΟΥ ΦΩΤΕΙΝΗ", "ΤΡΥΦΩΝΙΔΗΣ ΑΛΕΚΟΣ",
            ])

        seen_canon: set[str] = set()
        for pair in seed_pairs:
            up = clean_greek_token(pair)
            toks = up.split()
            if len(toks) < 2:
                continue
            sur_el, first_el = " ".join(toks[:-1]), toks[-1]
            canon = greek_fullname_to_turkish(up)
            if not canon or len(canon.split()) < 2:
                continue
            self.by_exact_el[f"{first_el} {sur_el}"] = canon
            self.by_exact_el[f"{sur_el} {first_el}"] = canon
            self.shapes.append((shape(sur_el), shape(first_el[:1]), shape(first_el), canon))
            if canon not in seen_canon:
                seen_canon.add(canon)
                self.members_canon.append(canon)

    def default_president(self, iso_date: str) -> str:
        if self.donem == 3:
            # Spiros Kiprianu became acting President of the Republic Aug 3, 1977; Alekos Mihailidis elected Speaker Sep 1977
            return "Spiros Kiprianu" if iso_date < "1977-09-01" else "Alekos Mihailidis"
        if self.donem == 8:
            return "Dimitris Hristofyas"
        if self.donem == 10:
            return "Yanakis Omiru"
        if self.donem == 11:
            return "Dimitris Silluris"
        if self.donem == 12:
            return "Annita Dimitriu"
        return "Meclis Başkanı"

    def resolve_person_token(self, raw_label: str) -> tuple[str, str]:
        """Resolve an MP/Chair name expression like `Γ. ΠΕΡΔΙΚΗΣ` or `ΑΝΤΡΟΣ ΚΥΠΡΙΑΝΟΥ` -> `(canon_name, method)`."""
        up = clean_greek_token(raw_label)
        if up in self.by_exact_el:
            return self.by_exact_el[up], "roster_exact"
        # Rescue unambiguous 1977 left-margin OCR truncations
        if self.donem == 3:
            if up in ("ΑΜΙΤΗΣ", "ΠΟΤΑΜΙΤΗΣ"):
                return "Con Potamitis", "ocr_trunc"
            if up in ("ΙΖΑΣ", "ΠΙΖΑΣ", "ΑΜΠΙΖΑΣ"):
                return "Stavros Ambizas", "ocr_trunc"
            if up in ("ΩΑΝΝΟΥ", "ΙΩΑΝΝΟΥ", "ΠΑΠΑΙΩΑΝΝΟΥ"):
                return "Ezekias Papayoannu", "ocr_trunc"
        sh_toks = shape(raw_label).split()
        if not sh_toks:
            return raw_label.strip(), "raw"
        if len(sh_toks) >= 3 and len(sh_toks[0]) <= 2:
            # Join OCR-split surname fragments after short initial (`ΣΤ. ΑΜΠ ΙΖΑΣ` -> `ΑΜΠΙΖΑΣ`, `Ε. ΠΑΠΑ ' Ι'ΩΑΝΝΟΥ` -> `ΠΑΠΑΙΩΑΝΝΟΥ`)
            first_sh = sh_toks[0]
            sur_sh = "".join(sh_toks[1:])
        else:
            sur_sh = sh_toks[-1]
            first_sh = sh_toks[0] if len(sh_toks) > 1 else ""
        init_sh = first_sh[:1] if first_sh else ""

        # Disambiguate initial-only `Α. ΚΥΠΡΙΑΝΟΥ` in KC:10 / KC:11 (where Andros Kiprianu is printed `ΑΝΤΡΟΣ ΚΥΠΡΙΑΝΟΥ`)
        if self.donem in (10, 11) and len(first_sh) == 1 and init_sh == "A" and sur_sh == shape("ΚΥΠΡΙΑΝΟΥ"):
            for _gs_s, _gs_i, _gs_f, canon in self.shapes:
                if canon == "Andreas Kiprianu":
                    return "Andreas Kiprianu", "roster_disambig"

        scored: list[tuple[float, str]] = []
        for gs_sur, gs_init, gs_first, canon in self.shapes:
            r_sur = 1.0 if sur_sh == gs_sur else difflib.SequenceMatcher(None, sur_sh, gs_sur).ratio()
            if r_sur < 0.82:
                continue
            bonus = 0.0
            if len(first_sh) >= 3:
                r_first = difflib.SequenceMatcher(None, first_sh, gs_first).ratio()
                if r_first >= 0.80:
                    bonus = 0.12
                elif init_sh != gs_init:
                    bonus = -0.15
            elif init_sh:
                bonus = 0.06 if init_sh == gs_init else -0.10
            scored.append((r_sur + bonus, canon))
        if scored:
            scored.sort(key=lambda x: -x[0])
            if scored[0][0] >= 0.84:
                return scored[0][1], "roster_match"

        # Fallback: transliterate Greek tokens directly if clean Greek name
        el_toks = [t for t in re.split(r"[\s.]+", up) if t]
        if len(el_toks) >= 2 and len(el_toks[-1]) >= 4:
            sur_tr = transliterate_greek_word(el_toks[-1])
            if len(el_toks[0]) >= 3:
                first_tr = transliterate_greek_word(el_toks[0])
                return f"{first_tr} {sur_tr}", "translit_full"
            init_tr = transliterate_greek_word(el_toks[0])[:1]
            if init_tr:
                return f"{init_tr}. {sur_tr}", "translit_init"
        return re.sub(r"\s+", " ", raw_label).strip(" :;.-—()"), "raw"


_SPEAKER_LINE_RE = re.compile(r"^([A-ZΑ-ΩΆ-ΏἈ-῿a-zα-ωά-ώἀ-ῷϊϋΐΰ .,'’\-]{3,68}?)\s*[:;]\s*$")
_PAREN_PERSON_RE = re.compile(r"^\(\s*([Α-ΩΆ-ΏA-Z .,'’\-]{4,42})\s*\)[\s.:;]*$")


def classify_speaker_line(
    line: str,
    next_nonempty_line: str,
    resolver: PeriodSpeakerResolver,
    iso_date: str,
    active_chair: str | None,
) -> tuple[str, str, str | None, str, bool, str | None] | None:
    """Inspect `line` (and `next_nonempty_line` for `(Δ. ΧΡΙΣΤΟΦΙΑΣ)`).

    Returns `(speaker_name, speaker_role, speaker_title, method, consume_next_line, new_active_chair)` or `None`.
    """
    s = line.strip()
    if not s:
        return None
    m = _SPEAKER_LINE_RE.match(s)
    if not m:
        return None
    raw_lab = m.group(1).strip()
    # Must be predominantly uppercase on speaker header lines (avoids matching normal prose ending in `:` like `Παρόντες:` or `είπε τα εξής:`)
    letters = [c for c in raw_lab if c.isalpha()]
    if len(letters) < 3 or len(raw_lab.split()) > 8:
        return None
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    up_clean = clean_greek_token(raw_lab)

    # Non-spoken structural sections (`Παρόντες:`, `ΚΕΦΑΛΑΙΟ ΠΡΩΤΟ:`, etc.)
    if any(up_clean.startswith(pfx) for pfx in NON_SPEAKER_PREFIXES):
        return TEXT_LABEL, "metin", None, "section", False, active_chair

    if upper_ratio < 0.72:
        return None

    sh = shape(raw_lab)
    sh_nospace = sh.replace(" ", "")

    # Check optional parenthetical person on the next line, e.g. `(Δ. ΧΡΙΣΤΟΦΙΑΣ)`
    paren_person = None
    m_p = _PAREN_PERSON_RE.match(next_nonempty_line.strip()) if next_nonempty_line else None
    if m_p:
        cand_in = m_p.group(1).strip()
        cand_letters = [c for c in cand_in if c.isalpha()]
        if cand_letters and (sum(1 for c in cand_letters if c.isupper()) / len(cand_letters)) >= 0.85:
            if "." in cand_in or len(cand_in.split()) >= 2:
                paren_person = cand_in

    # Clerk (`ΓΡΑΜΜΑΤΕΑΣ` / `ΓΡΑΜΜΑΤΕΥΣ` / OCR `ΤΕΥΣ`)
    if sh_nospace in ("TPAMMATEAZ", "TPAMMATEYZ", "TEYZ", "TPAMMATEIA") or difflib.SequenceMatcher(None, sh_nospace, "TPAMMATEAZ").ratio() >= 0.78 or difflib.SequenceMatcher(None, sh_nospace, "TPAMMATEYZ").ratio() >= 0.78:
        return "Katip", "katip", "Katip", "role", bool(paren_person), active_chair

    # President of the House (`ΠΡΟΕΔΡΟΣ` + 1977 Latin-OCR `nPOEb.POL`)
    if sh_nospace == "HPOEAPOZ" or difflib.SequenceMatcher(None, sh_nospace, "HPOEAPOZ").ratio() >= 0.82 or raw_lab.upper().startswith("NPOE"):
        if paren_person:
            nm, _ = resolver.resolve_person_token(paren_person)
            return nm, "baskan", "Meclis Başkanı", "chair_named", True, nm
        pres = resolver.default_president(iso_date)
        return pres, "baskan", "Meclis Başkanı", "chair_default", False, pres

    # Acting / Deputy Chair (`ΠΡΟΕΔΡΕΥΩΝ` / `ΠΡΟΕΔΡΕΥΟΥΣΑ` / OCR `Ρ ΕΥΩΝ`)
    if sh_nospace.startswith("HPOEAPEY") or sh_nospace in ("PEYON", "EYON") or difflib.SequenceMatcher(None, sh_nospace, "HPOEAPEYON").ratio() >= 0.78:
        if paren_person:
            nm, _ = resolver.resolve_person_token(paren_person)
            return nm, "gecici_baskan", "Geçici Başkan", "deputy_chair_named", True, nm
        if active_chair:
            return active_chair, "gecici_baskan", "Geçici Başkan", "deputy_chair_carried", False, active_chair
        return "Geçici Başkan", "gecici_baskan", "Geçici Başkan", "role", False, active_chair

    # Unidentified MP(s) (`ΒΟΥΛΕΥΤΗΣ` / `ΒΟΥΛΕΥΤΡΙΑ` / `ΒΟΥΛΕΥΤΕΣ` / `ΒΟΥΛΕΥΤΑΙ`)
    if sh_nospace in ("BOYAEΥTHZ", "BOYAEYTHZ", "BOYAEYTPIA"):
        return "Bir Milletvekili", "milletvekili_bakan", None, "role", False, active_chair
    if sh_nospace in ("BOYAEYTEZ", "BOYAEYTAI"):
        return "Milletvekilleri", "milletvekili_bakan", None, "role", False, active_chair

    # Ministers (`ΥΠΟΥΡΓΟΣ ...`)
    if up_clean.startswith("ΥΠΟΥΡΓΟΣ"):
        if paren_person:
            nm, mth = resolver.resolve_person_token(paren_person)
            return nm, "milletvekili_bakan", "Bakan", f"minister_{mth}", True, active_chair
        return "Bakan", "milletvekili_bakan", "Bakan", "role", False, active_chair

    # Regular MP (`Γ. ΠΕΡΔΙΚΗΣ:`, `ΑΝΤΡΟΣ ΚΥΠΡΙΑΝΟΥ:`)
    nm, mth = resolver.resolve_person_token(raw_lab)
    return nm, "milletvekili_bakan", None, mth, False, active_chair


def parse_sitting(
    pages: list[str],
    text_source: str,
    donem: int,
    iso_date: str,
    seq_in_day: int,
    file_title: str,
    source_url: str,
    resolver: PeriodSpeakerResolver,
    trans_cache: dict[str, str] | None = None,
    page_offset: int = 0,
    forced_yy: int | None = None,
) -> tuple[dict, list[dict]]:
    """Parse one sitting's extracted pages into `(doc_row, speech_rows)`."""
    trans_cache = trans_cache or {}
    yy_hdr, birlesim_hdr = extract_header_meta(pages[0] if pages else "", file_title)
    yy = forced_yy if forced_yy is not None else yy_hdr
    birlesim = birlesim_hdr or seq_in_day
    suffix = f" ({birlesim}. birleşim)" if birlesim_hdr else (f" (#{seq_in_day})" if seq_in_day > 1 else "")
    if seq_in_day > 1 and birlesim_hdr:
        suffix = f" ({birlesim}. birleşim #{seq_in_day})"
    doc_id = f"{FOLDER}/Dönem {donem} - {iso_date}{suffix}"
    dslug = f"KC_D{donem}_{iso_date.replace('-', '')}_{seq_in_day:02d}"

    # Flatten `(page_no, line)` stopping cleanly at `(Ώρα λήξεως / λήξης ...)` or `ΠΕΡΙΕΧΟΜΕΝΑ`
    flat: list[tuple[int, str]] = []
    ended = False
    for p_idx, ptxt in enumerate(pages, start=1):
        if ended:
            break
        for raw_ln in ptxt.splitlines():
            s = raw_ln.rstrip()
            if _TOC_RE.match(s) and p_idx > max(1, len(pages) - 6):
                ended = True
                break
            if _END_SITTING_RE.search(s) and p_idx >= max(1, len(pages) - 8):
                flat.append((p_idx, s))
                ended = True
                break
            if re.fullmatch(r"\s*\d{1,5}\s*", s):
                continue
            flat.append((p_idx, s))

    turns_raw: list[dict] = []
    cur_raw = None
    cur_name = None
    cur_role = None
    cur_title = None
    cur_method = None
    cur_page = 1
    active_chair: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        txt = "\n".join(body)
        txt = re.sub(r"(\w)[-‐­]\n(\w)", r"\1\2", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        if cur_name and txt:
            turns_raw.append({
                "page": cur_page,
                "speaker_raw": cur_raw,
                "speaker_name": cur_name,
                "speaker_role": cur_role,
                "speaker_title": cur_title,
                "canon_method": f"cy_{cur_method}",
                "text": txt,
                "lang": script_of(txt),
            })
        body = []

    idx = 0
    n_lines = len(flat)
    while idx < n_lines:
        pno, ln = flat[idx]
        # Peek next non-empty line on same/next page for `(Δ. ΧΡΙΣΤΟΦΙΑΣ)`
        nxt_ln = ""
        nxt_idx = idx + 1
        while nxt_idx < min(n_lines, idx + 4):
            if flat[nxt_idx][1].strip():
                nxt_ln = flat[nxt_idx][1]
                break
            nxt_idx += 1

        cls = classify_speaker_line(ln, nxt_ln, resolver, iso_date, active_chair)
        if cls is not None:
            flush()
            nm, role, title, mth, consume_nxt, active_chair = cls
            cur_raw = ln.strip().rstrip(":;").strip()
            if consume_nxt and nxt_ln:
                cur_raw = f"{cur_raw} {nxt_ln.strip()}"
                idx = nxt_idx + 1
            else:
                idx += 1
            cur_name, cur_role, cur_title, cur_method, cur_page = nm, role, title, mth, pno
            continue

        if cur_name is not None:
            body.append(ln)
        idx += 1
    flush()

    # Fallback if a very short procedural sitting had no speaker colon matched
    if not turns_raw:
        all_txt = "\n\n".join(p.strip() for p in pages if p.strip()).strip()
        if all_txt:
            turns_raw.append({
                "page": 1,
                "speaker_raw": TEXT_LABEL,
                "speaker_name": TEXT_LABEL,
                "speaker_role": "metin",
                "speaker_title": None,
                "canon_method": "cy_fallback_doc",
                "text": all_txt,
                "lang": script_of(all_txt),
            })

    speeches: list[dict] = []
    doc_blocks: list[str] = []
    for t_i, t in enumerate(turns_raw):
        page = (t["page"] or 1) + page_offset
        tr = lookup_translation(t["text"], trans_cache) if trans_cache else None
        surl = f"{source_url}#page={page}" if page else source_url
        speeches.append({
            "doc_id": doc_id,
            "meclis": "KC",
            "donem": donem,
            "yasama_yili": yy,
            "birlesim": birlesim,
            "oturum": 1,
            "tarih": iso_date,
            "speaker_raw": t["speaker_raw"],
            "speaker_name": t["speaker_name"],
            "speaker_role": t["speaker_role"],
            "speaker_title": t["speaker_title"],
            "constituency": None,
            "modifiers": LANG_LABEL.get(t["lang"] or "", "Yunanca"),
            "text_source": text_source,
            "source_file": file_title,
            "source_url": surl,
            "text": t["text"],
            "text_tr": tr,
            "char_count": len(t["text"]),
            "word_count": len(t["text"].split()),
            "turn_index": t_i,
            "speech_id": f"{dslug}#o1#t{t_i:04d}",
            "speaker_name_parsed": t["speaker_raw"],
            "canon_method": t["canon_method"],
        })
        if tr:
            doc_blocks.append(f"{t['speaker_raw']}:\n{t['text']}\n[Türkçe Çeviri]:\n{tr}")
        else:
            doc_blocks.append(f"{t['speaker_raw']}:\n{t['text']}")

    full_doc_text = "\n\n".join(doc_blocks)
    people = sorted({
        s["speaker_name"] for s in speeches
        if s["speaker_role"] == "milletvekili_bakan"
        and s["speaker_name"] not in ("Bir Milletvekili", "Milletvekilleri", "Bakan", TEXT_LABEL)
    })
    chairs = sorted({s["speaker_name"] for s in speeches if s["speaker_role"] in ("baskan", "gecici_baskan")})
    doc_row = {
        "doc_id": doc_id,
        "meclis": "KC",
        "donem": donem,
        "yasama_yili": yy,
        "birlesim": birlesim,
        "tarih": iso_date,
        "oturum_sayisi": 1,
        "konusma_sayisi": len(speeches),
        "konusmacilar": people,
        "baskanlar": chairs,
        "text": full_doc_text,
        "char_count": len(full_doc_text),
        "page_count": len(pages),
        "text_source": text_source,
        "source_file": file_title,
        "source_url": f"{source_url}#page={page_offset + 1}",
        "sha256": None,
    }
    return doc_row, speeches


_PARTY_HEADER_MAP = [
    ("ΔΗΣΥ", "DİSİ"),
    ("ΑΚΕΛ", "AKEL"),
    ("ΔΗΚΟ", "DİKO"),
    ("ΕΛΑΜ", "ELAM"),
    ("ΕΔΕΚ", "EDEK"),
    ("ΔΗΠΑ", "DİPA"),
    ("ΚΙΝΗΜΑ ΟΙΚΟΛΟΓΩΝ", "Ekologlar"),
    ("ΟΙΚΟΛΟΓΩΝ", "Ekologlar"),
    ("ΑΝΕΞΑΡΤΗΤΟΙ", "Bağımsız"),
    ("ΜΗ ΕΓΓΕΓΡΑΜΜΕΝΟΙ", "Bağımsız"),
]


def harvest_attendance_with_parties(pages: list[str]) -> list[tuple[str, str]]:
    """Extract `(CLEAN_EL_PAIR, party_tr)` from post-sitting attendance rolls (party-grouped in 2021-2026)."""
    tail = "\n".join(pages[max(0, len(pages) - 8) :])
    m_end = _END_SITTING_RE.search(tail)
    if not m_end:
        return []
    att = tail[m_end.end() :]
    m_toc = re.search(r"\b(?:ΠΕΡΙΕΧΟΜΕΝΑ|ΑΠΟΝΤΕΣ)\b", att, re.I)
    if m_toc:
        att = att[: m_toc.start()]
    pat = re.compile(
        r"\b([Α-ΩΆ-ΏἈ-῿][α-ωά-ώἀ-ῷϊϋΐΰ]{2,20}\s+[Α-ΩΆ-ΏἈ-῿][α-ωά-ώἀ-ῷϊϋΐΰ]{2,20}(?:\s+(?:Στέλλα|Σκεύη))?)\b"
    )
    found: list[tuple[str, str]] = []
    cur_party = ""
    for ln in att.splitlines():
        s = re.sub(r"\s+", " ", ln.strip())
        if not s:
            continue
        up_ln = re.sub(r"[^Α-Ω]+", " ", clean_greek_token(s)).strip()
        matched_party = False
        for k, v in _PARTY_HEADER_MAP:
            if up_ln == k or up_ln.startswith(k + " ") or (len(k) >= 8 and k in up_ln):
                cur_party = v
                matched_party = True
                break
        if matched_party:
            continue
        if "ΘΡΗΣΚΕΥΤΙΚ" in up_ln:
            cur_party = ""
            continue
        for m in pat.finditer(s):
            up = clean_greek_token(m.group(1))
            if len(up.split()) not in (2, 3):
                continue
            if any(bad in up for bad in _ROSTER_BAD_SUBSTR):
                continue
            found.append((up, cur_party))
    return found


def build_consolidated_period(
    vol_dir: Path,
    donem: int,
    out_dir: Path,
    translations_path: Path | None = None,
) -> dict:
    """Split & parse consolidated multi-sitting `parliament.cy` PDFs (using PDF TOC bookmarks) into JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vols = json.loads((vol_dir / "manifest.json").read_text(encoding="utf-8"))
    trans_cache = load_cache(translations_path) if (translations_path and translations_path.exists()) else {}

    page_cache_dir = out_dir / "_pages_cache"
    page_cache_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[tuple[dict, str, int, int, int, list[str], str]] = []  # (vol, iso_date, seq, start_p0, yy, pages, tsrc)
    roster_counts: Counter = Counter()
    party_votes: dict[str, Counter] = defaultdict(Counter)
    day_seq: dict[str, int] = defaultdict(int)

    for v in vols:
        pdf_path = vol_dir / f"{v['code']}.pdf"
        doc = pymupdf.open(pdf_path)
        n_total = len(doc)
        toc = [t for t in doc.get_toc() if t[0] == 1 and "ΣΕΛΙΔΑ" not in t[1].upper()]
        sittings_meta: list[tuple[str, int, int, str]] = []  # (title, start_0idx, end_0idx_excl, iso_date)
        for ti, t_entry in enumerate(toc):
            title_raw = t_entry[1].replace("\u200e", "").strip()
            start_0 = max(0, int(t_entry[2]) - 1)
            end_0 = max(start_0 + 1, int(toc[ti + 1][2]) - 1) if (ti + 1) < len(toc) else n_total
            m_dt = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", title_raw)
            if not m_dt:
                continue
            dd, mm, yyyy = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
            iso_date = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
            sittings_meta.append((title_raw, start_0, end_0, iso_date))

        # Check if volume needs ToUnicode repair
        sample = "\n".join(doc[i].get_text("text") for i in range(min(5, n_total)))
        tsrc = "pdf_digital"
        if needs_tounicode_repair(sample):
            nfix = fix_broken_tounicode(doc)
            if nfix > 0:
                raw_bytes = doc.tobytes()
                doc.close()
                doc = pymupdf.open("pdf", raw_bytes)
                tsrc = "pdf_tounicode_repaired"

        for title_raw, start_0, end_0, iso_date in sittings_meta:
            day_seq[iso_date] += 1
            seq = day_seq[iso_date]
            cache_file = page_cache_dir / f"D{donem}_{iso_date}_{seq:02d}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                pages = cached["pages"]
                tsrc_use = cached["text_source"]
            else:
                pages = [doc[p_i].get_text("text") for p_i in range(start_0, end_0)]
                tsrc_use = tsrc
                cache_file.write_text(
                    json.dumps({"pages": pages, "text_source": tsrc_use}, ensure_ascii=False),
                    encoding="utf-8",
                )
            for pair, pty in harvest_attendance_with_parties(pages):
                roster_counts[pair] += 1
                if pty:
                    party_votes[pair][pty] += 1
            extracted.append((v, iso_date, seq, start_0, int(v["yy"]), pages, tsrc_use))
        doc.close()
        print(f"  [Consolidated {v['code']}] {len(sittings_meta)} sittings ({n_total} pages)", flush=True)

    valid_pairs = [pair for pair, cnt in roster_counts.most_common() if cnt >= 6]
    resolver = PeriodSpeakerResolver(donem, valid_pairs)

    # Map canonical name -> top party vote
    canon_party: dict[str, str] = {}
    for pair in valid_pairs:
        if party_votes.get(pair):
            canon_party[greek_fullname_to_turkish(pair)] = party_votes[pair].most_common(1)[0][0]
    if donem == 12:
        canon_party.setdefault("Annita Dimitriu", "DİSİ")

    docs: list[dict] = []
    speeches: list[dict] = []
    for v, iso_date, seq, start_0, yy, pages, tsrc_use in extracted:
        doc_row, sp_rows = parse_sitting(
            pages=pages,
            text_source=tsrc_use,
            donem=donem,
            iso_date=iso_date,
            seq_in_day=seq,
            file_title=f"{v['label']}.pdf",
            source_url=v["url"],
            resolver=resolver,
            trans_cache=trans_cache,
            page_offset=start_0,
            forced_yy=yy,
        )
        docs.append(doc_row)
        speeches.extend(sp_rows)

    for name, rows in (("documents", docs), ("speeches", speeches)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    roster_export = {
        f"KC:{donem}": {
            "meclis": "KC",
            "donem": donem,
            "partial": True,
            "source": "https://www.parliament.cy",
            "members": [
                {"name": nm, "constituency": "", "party": canon_party.get(nm, ""), "note": ""}
                for nm in resolver.members_canon
            ],
        }
    }
    (out_dir / "rosters.json").write_text(json.dumps(roster_export, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "donem": donem,
        "documents": len(docs),
        "speeches": len(speeches),
        "translated_speeches": sum(1 for s in speeches if s.get("text_tr")),
        "min_date": min((d["tarih"] for d in docs if d.get("tarih")), default=None),
        "max_date": max((d["tarih"] for d in docs if d.get("tarih")), default=None),
        "roles": dict(Counter(s["speaker_role"] for s in speeches)),
        "methods": dict(Counter(s["canon_method"] for s in speeches)),
        "roster_count": len(resolver.members_canon),
        "roster_with_party": sum(1 for nm in resolver.members_canon if canon_party.get(nm)),
        "top_speakers": Counter(s["speaker_name"] for s in speeches).most_common(15),
        "top_unresolved_raw": Counter(s["speaker_name"] for s in speeches if s["canon_method"] == "cy_raw").most_common(20),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_modern_archive(
    modern_dir: Path,
    ocr_root: Path,
    out_dir: Path,
    translations_path: Path | None = None,
    merge_cy_build: Path | None = None,
) -> dict:
    """Extract & parse all 378 modern/1977-79 sittings, optionally merging 1960-64 `data/cy_build`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_rows = [
        json.loads(ln)
        for ln in (modern_dir / "local_index.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    trans_cache = load_cache(translations_path) if (translations_path and translations_path.exists()) else {}

    # Pass 1: extract/load cached pages & harvest official attendance rosters per period
    page_cache_dir = out_dir / "_pages_cache"
    page_cache_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[tuple[dict, int, int, list[str], str]] = []
    roster_counts: dict[int, Counter] = defaultdict(Counter)
    day_seq: dict[tuple[int, str], int] = defaultdict(int)

    for i, rec in enumerate(idx_rows, start=1):
        donem = determine_donem(rec["period"], rec["date"])
        day_seq[(donem, rec["date"])] += 1
        seq = day_seq[(donem, rec["date"])]
        pdf_path = modern_dir / rec["local"]
        cache_file = page_cache_dir / f"D{donem}_{rec['date']}_{seq:02d}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            pages, tsrc = cached["pages"], cached["text_source"]
        else:
            pages, tsrc = extract_sitting_pages(pdf_path, ocr_root=ocr_root)
            if tsrc != "pdf_empty":
                cache_file.write_text(
                    json.dumps({"pages": pages, "text_source": tsrc}, ensure_ascii=False),
                    encoding="utf-8",
                )
        for pair in harvest_attendance_names(pages):
            roster_counts[donem][pair] += 1
        extracted.append((rec, donem, seq, pages, tsrc))
        if i % 50 == 0 or i == len(idx_rows):
            print(f"  [Pass 1/2] Extracted {i}/{len(idx_rows)} PDFs...", flush=True)

    # Keep pairs seen in >= 3 sittings (filters out 1-off OCR noise in 1977-79)
    resolvers: dict[int, PeriodSpeakerResolver] = {}
    rosters_export: dict[str, dict] = {}
    period_meta = {
        3: ("Γ΄ Dönem (1976–1981)", "1976–1981"),
        8: ("Η΄ Dönem (2001–2006)", "2001–2006"),
        10: ("Ι΄ Dönem (2011–2016)", "2011–2016"),
        11: ("ΙΑ΄ Dönem (2016–2021)", "2016–2021"),
    }
    for donem in (3, 8, 10, 11):
        min_freq = 4 if donem == 3 else 3
        valid_pairs = [pair for pair, cnt in roster_counts[donem].most_common() if cnt >= min_freq]
        resolvers[donem] = PeriodSpeakerResolver(donem, valid_pairs)
        lbl, yrs = period_meta[donem]
        rosters_export[f"KC:{donem}"] = {
            "label": f"Kıbrıs Cumhuriyeti Temsilciler Meclisi · {lbl}",
            "years": yrs,
            "partial": True,
            "members": [{"name": nm, "party": "", "constituency": ""} for nm in resolvers[donem].members_canon],
        }

    # Pass 2: parse all sittings with period-aware MP resolvers
    docs: list[dict] = []
    speeches: list[dict] = []
    for rec, donem, seq, pages, tsrc in extracted:
        doc_row, sp_rows = parse_sitting(
            pages=pages,
            text_source=tsrc,
            donem=donem,
            iso_date=rec["date"],
            seq_in_day=seq,
            file_title=rec["file"],
            source_url=rec["url"],
            resolver=resolvers[donem],
            trans_cache=trans_cache,
        )
        docs.append(doc_row)
        speeches.extend(sp_rows)

    # Optionally prepend 1960-64 `data/cy_build` rows so a single `--add-dir` updates all KC terms together
    if merge_cy_build and (merge_cy_build / "documents.jsonl").exists():
        base_docs = [json.loads(x) for x in (merge_cy_build / "documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        base_sp = [json.loads(x) for x in (merge_cy_build / "speeches.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        docs = base_docs + docs
        speeches = base_sp + speeches

    for name, rows in (("documents", docs), ("speeches", speeches)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_dir / "rosters.json").write_text(json.dumps(rosters_export, ensure_ascii=False, indent=2), encoding="utf-8")

    by_donem_docs = Counter(d["donem"] for d in docs)
    by_donem_sp = Counter(s["donem"] for s in speeches)
    by_donem_tr = Counter(s["donem"] for s in speeches if s.get("text_tr"))
    by_role = Counter(s["speaker_role"] for s in speeches if s["donem"] is not None)
    by_method = Counter(s["canon_method"] for s in speeches if s["donem"] is not None)
    unresolved = Counter(s["speaker_name"] for s in speeches if s["canon_method"] == "cy_raw").most_common(30)

    summary = {
        "documents": len(docs),
        "speeches": len(speeches),
        "translated_speeches": sum(1 for s in speeches if s.get("text_tr")),
        "docs_by_donem": {str(k): v for k, v in sorted(by_donem_docs.items(), key=lambda x: (x[0] or 0))},
        "speeches_by_donem": {str(k): v for k, v in sorted(by_donem_sp.items(), key=lambda x: (x[0] or 0))},
        "translated_by_donem": {str(k): v for k, v in sorted(by_donem_tr.items(), key=lambda x: (x[0] or 0))},
        "modern_roles": dict(by_role),
        "modern_methods": dict(by_method),
        "roster_counts": {k: len(v["members"]) for k, v in rosters_export.items()},
        "top_unresolved_raw": unresolved,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modern-dir", default="data/cy_raw/modern")
    ap.add_argument("--consolidated-dir", default="", help="Optional consolidated-volume folder with manifest.json")
    ap.add_argument("--donem", type=int, default=12, help="Donem number when using --consolidated-dir")
    ap.add_argument("--ocr-root", default="data/cy_pages/modern_1977_79")
    ap.add_argument("--out", default="data/cy_modern_build")
    ap.add_argument("--translations", default="data/cy_translations/cache.jsonl")
    ap.add_argument("--merge-cy-build", default="", help="Optional path to 1960-64 data/cy_build to merge")
    args = ap.parse_args()

    if args.consolidated_dir:
        summary = build_consolidated_period(
            vol_dir=Path(args.consolidated_dir),
            donem=args.donem,
            out_dir=Path(args.out),
            translations_path=Path(args.translations) if args.translations else None,
        )
    else:
        summary = build_modern_archive(
            modern_dir=Path(args.modern_dir),
            ocr_root=Path(args.ocr_root),
            out_dir=Path(args.out),
            translations_path=Path(args.translations) if args.translations else None,
            merge_cy_build=Path(args.merge_cy_build) if args.merge_cy_build else None,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
