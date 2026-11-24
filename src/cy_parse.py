"""Parse OCR'd Republic of Cyprus House of Representatives minutes (1960-64, Greek + Turkish) into speech turns.

Input : a directory of per-page OCR text files pNNNN.txt (Tesseract ell+tur) for one session volume.
Output: JSONL, one record per speech turn, in the KKTC dataset schema plus 'lang' (el / tr / mixed).

Speakers are labelled on their own line ("ΠΡΟΕΔΡΟΣ :", "ΑΙΜ. ΦΡΑΓΚΟΣ :", "H. A. RİZA:", "KÂTİP:").
Greek labels are often OCR'd with Latin look-alikes ("TPAMMATEYZ" = ΓΡΑΜΜΑΤΕΥΣ, "HPOEAPOZ" = ΠΡΟΕΔΡΟΣ), so
labels are compared in a 'shape' alphabet where look-alike Greek and Latin capitals coincide.

Usage: python src/cy_parse.py out/1_A --volume "A' Σύνοδος (1960-61)" --source-url <pdf url> > turns_A.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- normalisation helpers
_SHAPE = str.maketrans({
    "Α": "A", "Β": "B", "Γ": "T", "Δ": "A", "Ε": "E", "Ζ": "Z", "Η": "H", "Θ": "O", "Ι": "I", "Κ": "K", "Λ": "A",
    "Μ": "M", "Ν": "N", "Ξ": "E", "Ο": "O", "Π": "H", "Ρ": "P", "Σ": "Z", "Τ": "T", "Υ": "Y", "Φ": "O", "Χ": "X",
    "Ψ": "Y", "Ω": "O", "S": "Z", "R": "P", "G": "O", "D": "A", "L": "A", "C": "O", "Q": "O", "0": "O", "U": "Y", "V": "Y",
})


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def shape(s: str) -> str:
    """Upper-case, accent-free, look-alike-collapsed form: ΠΡΟΕΔΡΟΣ and HPOEAPOZ both -> HPOEAPOZ."""
    s = strip_accents(s).upper().replace("İ", "I").replace("Ş", "Z").replace("Ç", "O").replace("Ğ", "O").replace("Ö", "O").replace("Ü", "Y")
    s = s.translate(_SHAPE)
    return " ".join(re.sub(r"[^A-Z ]", " ", s).split())


def script_of(text: str) -> str:
    g = len(re.findall(r"[Ͱ-Ͽἀ-῿]", text))
    lat = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû]", text))
    if g + lat == 0:
        return ""
    r = g / (g + lat)
    return "el" if r > 0.7 else ("tr" if r < 0.3 else "mixed")


# ---------------------------------------------------------------- speakers
# Greek members of the 1960 House: (Greek surname, Greek first-name initial, roster name)
GREEK_MEMBERS = [
    ("ΚΛΗΡΙΔΗΣ", "Γ", "Glafkos Klerides"), ("ΣΑΒΒΙΔΗΣ", "Μ", "Mihalakis Savvidis"), ("ΣΤΥΛΙΑΝΟΥ", "Π", "Petros Stilyanu"),
    ("ΤΟΜΠΑΖΟΣ", "Γ", "Yorgos Tombazos"), ("ΚΟΣΙΗΣ", "Ν", "Nikos Koşis"),
    ("ΔΗΜΗΤΡΙΑΔΗΣ", "Λ", "Lellos Dimitriadis"), ("ΛΥΣΣΑΡΙΔΗΣ", "Β", "Vasos Lisaridis"), ("ΧΑΤΖΗΧΑΡΟΣ", "Χ", "Haridimos Hacıharos"),
    ("ΦΑΝΟΣ", "Κ", "Konstantinos Fanos"), ("ΓΙΑΜΑΚΗΣ", "Ι", "Yosif Yamakis"), ("ΖΙΑΡΤΙΔΗΣ", "Α", "Andreas Ziartidis"),
    ("ΠΑΠΑΪΩΑΝΝΟΥ", "Ε", "Ezekias Papayoannu"), ("ΠΑΠΑΔΟΠΟΥΛΛΟΣ", "Α", "Andreas Papadopulos"),
    ("ΑΝΑΣΤΑΣΙΑΔΗΣ", "Α", "Antonios Anastasiadis"), ("ΦΡΑΓΚΟΣ", "Α", "Emilios Frangos"),
    ("ΜΙΧΑΗΛΙΔΗΣ", "Χ", "Hristodulos Mihailidis"), ("ΡΟΔΟΣΘΕΝΟΥΣ", "Λ", "Lefkios Rodostenus"),
    ("ΜΙΧΑΗΛΙΔΗΣ", "Σ", "Stelyos Mihailidis"), ("ΠΟΤΑΜΙΤΗΣ", "Ι", "Con Potamitis"), ("ΠΑΝΑΓΙΔΗΣ", "Δ", "Dafnis Panayidis"),
    ("ΠΑΥΛΟΥ", "Π", "Pavlos Pavlu"), ("ΑΓΓΕΛΙΔΗΣ", "Ν", "Nikos Angelidis"), ("ΧΡΙΣΤΟΔΟΥΛΙΔΗΣ", "Κ", "Kostas Hristodulidis"),
    ("ΚΑΡΣΕΡΑΣ", "Α", "Andreas Karseras"), ("ΣΑΝΤΗΣ", "Γ", "Yorgos Santis"), ("ΤΟΥΜΑΖΗΣ", "Π", "Panayotis Tumazis"),
    ("ΜΙΧΑΗΛΙΔΗΣ", "Χ", "Haralambos Mihailidis"), ("ΓΙΑΝΝΑΚΗΣ", "Σ", "Sotiris Yannakis"), ("ΤΖΙΡΚΩΤΗΣ", "Γ", "Yorgos Cirkotis"),
    ("ΠΟΣΚΩΤΗΣ", "Σ", "Stavros Poskotis"), ("ΔΗΜΗΤΡΙΑΔΗΣ", "Χ", "Hrisis Dimitriadis"), ("ΙΩΑΝΝΙΔΗΣ", "Γ", "Yorgos Yoannidis"),
    ("ΝΙΚΟΛΑΪΔΗΣ", "Η", "İrodotos Nikolaidis"), ("ΣΙΒΙΤΑΝΙΔΗΣ", "Π", "Panikos Sivitanidis"), ("ΓΙΑΓΚΟΥ", "Α", "Andreas Yangu"),
    ("ΠΑΡΑΣΚΕΥΑΪΔΗΣ", "Φ", "Fidias Paraskevaidis"), ("ΚΥΠΡΙΑΝΟΥ", "Σ", "Spiros Kiprianu"),
    # Turkish members as printed in Greek script
    ("ΡΙΖΑ", "Χ", "Halit Ali Rıza"), ("ΕΙΖΑ", "Η", "Halit Ali Rıza"), ("ΧΑΛΙΤΑΛΗΡΙΖΑ", "Χ", "Halit Ali Rıza"), ("ΣΟΥΛΕΪΜΑΝ", "Ο", "Ümit Süleyman Onan"),
    ("ΝΑΜΠΙ", "Ν", "Nebil Nabi"), ("ΝΑΒΙ", "Ν", "Nebil Nabi"), ("ΝΑΛΜΠΑΝΤΟΓΛΟΥ", "Μ", "Burhan Nalbandoğlu"),
    ("ΤΖΕΜΙΛ", "Ρ", "Ramadan Cemil"), ("ΝΤΕΝΙΖ", "Κ", "Kemal Deniz"), ("ΜΠΕΡΜΠΕΡΟΓΛΟΥ", "Α", "Ahmet Mithat Berberoğlu"),
    ("ΜΟΥΝΤΕΡΡΙΣΟΓΛΟΥ", "Ο", "Orhan Müderrisoğlu"), ("ΝΤΕΡΒΙΣ", "Χ", "Hüseyin Derviş"), ("ΟΡΧΑΝ", "Ι", "İbrahim Orhan"),
    ("ΑΛΤΑΪ", "Α", "Aziz Altay"), ("ΓΚΙΟΥΒΕΝΕΡ", "Χ", "Hasan Güvener"), ("ΜΑΝΙΕΡΑ", "Ν", "Niyazi Manyera"),
]
TURKISH_MEMBERS = {
    "riza": "Halit Ali Rıza", "suleyman": "Ümit Süleyman Onan", "suleyamn": "Ümit Süleyman Onan", "nalbantoglu": "Burhan Nalbandoğlu",
    "nalbandoglu": "Burhan Nalbandoğlu", "cemil": "Ramadan Cemil", "deniz": "Kemal Deniz", "berberoglu": "Ahmet Mithat Berberoğlu",
    "nabi": "Nebil Nabi", "muderrisoglu": "Orhan Müderrisoğlu", "dervis": "Hüseyin Derviş", "orhan": "İbrahim Orhan",
    "altay": "Aziz Altay", "guvener": "Hasan Güvener", "manyera": "Niyazi Manyera", "orek": "Osman Örek", "plumer": "Fazıl Plümer",
    "somek": "Selçuk Şakir Somek", "semsettin": "Halit Kazım Şemsettin", "huseyin": "Necdet Hüseyin", "kucuk": "Fazıl Küçük",
}
TEXT_LABEL = "Tutanak Metni"
ROLES = [  # (shape or folded label, canonical name, role)
    ("HPOEAPEYON", "Meclis Başkanı", "baskan"), ("TOYPKOZ TPAMMATEYZ", "Katip", "katip"), ("EAAHN TPAMMATEYZ", "Katip", "katip"),
    ("HPOEAPOZ", "Glafkos Klerides", "baskan"), ("TPAMMATEYZ", "Katip", "katip"), ("AHTIHPOEAPOZ", "Orhan Müderrisoğlu", "baskan"),
    ("KATIP", "Katip", "katip"), ("MEOAIZ PEIZI", "Glafkos Klerides", "baskan"), ("MEOAIZ PEIZ YEKIAI", "Orhan Müderrisoğlu", "baskan"),
    ("PEIZ YEKIAI", "Orhan Müderrisoğlu", "baskan"),
]
_GM_SHAPES = [(shape(s), shape(i), n) for s, i, n in GREEK_MEMBERS]
_ROLE_SHAPES = [(r, n, role) for r, n, role in ROLES]
NOT_SPEAKERS = {"HMEPHZIA AIATAEIZ", "OYNAEM", "TYNAEM", "HPAKTIKA", "AIATAEIZ", "TYNAEMIN", "OYNAEMIN"}


def fold_tr(s: str) -> str:
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return strip_accents(s.replace("ı", "i")).replace("ş", "s")


_HEADING_WORDS = ("EKOEZIZ", "EHITPOHHZ", "NOMOZXEAIOY", "KOMITE", "PAHOPY", "PAPOPY", "ZOPYZY", "ZOTYZY", "TEKAIOI", "ONEPIZI")


def resolve_label(label: str) -> tuple[str, str | None, str]:
    """Return (speaker_name, role, method) for a raw label."""
    sh = shape(label)
    sh = " ".join(re.sub(r"^(?:TI|II|IT|TT)(?=P)", "H", t) for t in sh.split())  # Π read as 'TI' / 'II'
    if any(w in sh.replace(" ", "") for w in _HEADING_WORDS):
        return TEXT_LABEL, None, "text"
    best = None
    for r, n, role in _ROLE_SHAPES:
        ratio = 1.0 if sh == r else difflib.SequenceMatcher(None, sh, r).ratio()
        if ratio >= (0.75 if len(r) >= 7 else 0.86) and (best is None or ratio > best[0]):
            best = (ratio, n, role)
    if best:
        return best[1], best[2], "role"
    if len(sh) <= 6 and difflib.SequenceMatcher(None, sh.replace(" ", ""), "KATIP").ratio() >= 0.6 and sh[:1] in "KBR":
        return "Katip", "katip", "role"
    toks = sh.split()
    if not toks:
        return label.strip(), "milletvekili_bakan", "raw"
    surname, initial = toks[-1], toks[0][:1] if len(toks) > 1 else ""
    # Turkish members (Latin script, Turkish letters)
    ft = re.sub(r"[^a-z ]", " ", fold_tr(label)).split()
    if ft:
        key = re.sub(r"[^a-z]", "", ft[-1])
        if key in TURKISH_MEMBERS:
            return TURKISH_MEMBERS[key], "milletvekili_bakan", "tr_member"
        best = difflib.get_close_matches(key, TURKISH_MEMBERS.keys(), n=1, cutoff=0.84)
        if best and len(key) >= 5:
            return TURKISH_MEMBERS[best[0]], "milletvekili_bakan", "tr_member_fuzzy"
    # Greek members in shape space
    scored = []
    cands = {surname, toks[0]} if len(toks[0]) >= 5 else {surname}
    if len(toks) >= 2:
        cands.add(toks[-2] + toks[-1])
    if len(toks) >= 3:
        cands.add("".join(toks[-3:]))
    for gs, gi, n in _GM_SHAPES:
        r = max(difflib.SequenceMatcher(None, c, gs).ratio() for c in cands)
        if r >= 0.8:
            scored.append((r + (0.05 if initial and initial == gi[:1] else 0), n))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1], "milletvekili_bakan", "el_member"
    return re.sub(r"\s+", " ", label).strip(" :;.-—"), "milletvekili_bakan", "raw"


_LABEL_RE = re.compile(r"^([A-ZÇĞİÖŞÜÂÎÛΑ-ΩΆ-ΏἈ-῿ .,'’\-]{3,48}?)\s*[:;][\s:;.\-—–]*$")


def is_label(line: str) -> str | None:
    s = line.strip()
    m = _LABEL_RE.match(s)
    if not m:
        if shape(s) in ("HPOEAPOZ", "TPAMMATEYZ", "KATIP"):
            return s
        return None
    lab = m.group(1).strip()
    letters = [c for c in lab if c.isalpha()]
    if len(letters) < 3 or len(lab.split()) > 6:
        return None
    if shape(lab) in NOT_SPEAKERS or shape(lab).startswith(("HMEPHZIA", "OYNAEM", "TYNAEM")):
        return None
    return lab


# ---------------------------------------------------------------- sittings
TR_MONTHS = {"OCAK": 1, "ŞUBAT": 2, "SUBAT": 2, "MART": 3, "NİSAN": 4, "NISAN": 4, "MAYIS": 5, "HAZİRAN": 6, "HAZIRAN": 6,
             "TEMMUZ": 7, "AĞUSTOS": 8, "AGUSTOS": 8, "EYLÜL": 9, "EYLUL": 9, "EKİM": 10, "EKIM": 10, "KASIM": 11,
             "ARALIK": 12}
EL_MONTHS = {"IANOYAPIOY": 1, "OEBPOYAPIOY": 2, "MAPTIOY": 3, "AHPIAIOY": 4, "MAIOY": 5, "IOYNIOY": 6, "IOYAIOY": 7,
             "AYTOYZTOY": 8, "ZEHTEMBPIOY": 9, "OKTOBPIOY": 10, "NOEMBPIOY": 11, "AEKEMBPIOY": 12}
_TR_DATE = re.compile(r"(\d{1,2})\s*\.?\s*([A-ZÇĞİÖŞÜ]{4,8}),?\s+(1[9９]\s?[6]\s?[0-9oOο])\s+TAR[İI]HL[İI]\s+TOPLANT", re.I)
_EL_DATE = re.compile(r"ΣΥΝ\w*ΔΡΙΑΣ\w*.{0,8}?(\d{1,2})\w{0,3}\s+([Α-ΩΆ-ΏἈ-῿]{4,12})\s+(19[0-9oOο]{2})")


def _year(y: str) -> int | None:
    y = re.sub(r"\s", "", y).replace("o", "0").replace("O", "0").replace("ο", "0").replace("９", "9")
    return int(y) if re.fullmatch(r"19\d\d", y) else None


_EL_HEADER = re.compile(r"^\s*(?:ΕΚΤΑΚΤΟΣ\s+)?ΣΥΝ\w{0,3}ΕΔΡΙΑΣ(?:ΙΣ|Η)\b(.{0,40})$")
_EL_MONTH_YEAR = re.compile(r"([Α-ΩΆ-Ώ\u1f08-\u1fff]{4,12})\s+(1\s?9\s?[0-9oOο]\s?[0-9oOο])")


def _valid(y: int, mo: int, d: int | None, years: tuple[int, int] | None) -> bool:
    if years and not (years[0] <= y <= years[1]):
        return False
    return 1 <= mo <= 12 and (d is None or 1 <= d <= 31)


def find_date(line: str, years: tuple[int, int] | None = None) -> str | None:
    m = _TR_DATE.search(line)
    if m:
        mo = TR_MONTHS.get(m.group(2).upper())
        y = _year(m.group(3))
        d = int(m.group(1))
        if mo and y and _valid(y, mo, d, years):
            return f"{y}-{mo:02d}-{d:02d}"
    m = _EL_DATE.search(line)
    if m:
        mo = EL_MONTHS.get(shape(m.group(2)).replace(" ", ""))
        y = _year(m.group(3))
        d = int(m.group(1))
        if mo and y and _valid(y, mo, d, years):
            return f"{y}-{mo:02d}-{d:02d}"
    return None


def greek_header(line: str, years: tuple[int, int] | None = None) -> str | None:
    """A Greek sitting header whose day OCR'd badly ('ΣΥΝΕΔΡΙΑΣΙΣ ΤΗΣ Ί8ης ΙΑΝΟΥΑΡΙΟΥ 1961') -> 'YYYY-MM'."""
    m = _EL_HEADER.match(line)
    if not m:
        return None
    my = _EL_MONTH_YEAR.search(m.group(1))
    if not my:
        return "?"
    mo = EL_MONTHS.get(shape(my.group(1)).replace(" ", ""))
    y = _year(my.group(2))
    if mo and y and _valid(y, mo, None, years):
        return f"{y}-{mo:02d}"
    return "?"


# ---------------------------------------------------------------- main parse
def load_lines(page_dir: Path) -> list[tuple[int, str]]:
    out = []
    for p in sorted(page_dir.glob("p*.txt")):
        pno = int(p.stem[1:])
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.rstrip()
            if re.fullmatch(r"\s*\d{1,4}\s*", s) or not s.strip():
                out.append((pno, ""))
                continue
            out.append((pno, s))
    return out


def join_hyphens(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"(\w)[-‐]\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse(page_dir: Path, volume: str, source_url: str, meclis: str = "KC", donem=None,
          years: tuple[int, int] | None = None) -> list[dict]:
    lines = load_lines(page_dir)
    turns, cur, body, date, sitting = [], None, [], None, 0
    page_start = None

    def flush():
        nonlocal body, page_start
        text = join_hyphens(body)
        if cur and text:
            name, role, method = resolve_label(cur)
            turns.append({
                "meclis": meclis, "donem": donem, "tarih": date, "sitting": sitting, "page": page_start,
                "speaker_raw": cur, "speaker_name": name, "speaker_role": role, "canon_method": f"cy_{method}",
                "text": text, "lang": script_of(text), "source_url": source_url, "volume": volume,
            })
        body, page_start = [], None

    pending_month = None  # Greek header seen, exact day not yet known
    since_header = 0
    for pno, ln in lines:
        d = find_date(ln, years)
        if d:
            if pending_month is not None and since_header <= 40:
                date, pending_month = d, None  # same sitting: Turkish header right after the Greek one
                continue
            if d != date:
                flush()
                cur, date, sitting = None, d, sitting + 1
            continue
        gh = greek_header(ln, years)
        if gh:
            flush()
            cur, sitting = None, sitting + 1
            pending_month, since_header = gh, 0
            date = gh if gh != "?" else (date[:7] if date else None)
            continue
        since_header += 1
        lab = is_label(ln) if ln else None
        if lab:
            flush()
            cur, page_start = lab, pno
            continue
        if cur is not None:
            if page_start is None:
                page_start = pno
            body.append(ln)
    flush()
    for i, t in enumerate(turns):
        t["turn_index"] = i
        t["char_count"] = len(t["text"])
        t["word_count"] = len(t["text"].split())
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page_dir")
    ap.add_argument("--volume", default="")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--years", nargs=2, type=int, default=None, help="Valid year window, e.g. 1960 1961")
    a = ap.parse_args()
    turns = parse(Path(a.page_dir), a.volume, a.source_url, years=tuple(a.years) if a.years else None)
    if a.stats:
        from collections import Counter
        print("turns", len(turns), "sittings", len({t["tarih"] for t in turns}), file=sys.stderr)
        print("lang", Counter(t["lang"] for t in turns), file=sys.stderr)
        print("method", Counter(t["canon_method"] for t in turns), file=sys.stderr)
        print("speakers", Counter(t["speaker_name"] for t in turns).most_common(60), file=sys.stderr)
        print("raw_unresolved", Counter(t["speaker_raw"] for t in turns if t["canon_method"] == "cy_raw").most_common(40), file=sys.stderr)
        print("dates", sorted({t["tarih"] for t in turns if t["tarih"]}), file=sys.stderr)
    for t in turns:
        sys.stdout.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
