"""Parse extracted KKTC Meclisi minutes into structured speaker turns and document records."""

import argparse
import concurrent.futures
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

EXTRACTED_DIR = Path("data/extracted")
PARSED_DIR = Path("data/parsed")

# --- Turkish case helpers & OCR cleanup ---
_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı", "Î": "î", "Â": "â", "Û": "û"})
_TR_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I", "î": "Î", "â": "Â", "û": "Û"})
_TR_NO_CFX = str.maketrans({"î": "i", "Î": "İ", "â": "a", "Â": "A", "û": "u", "Û": "U"})
_MOJIBAKE_TR = str.maketrans({"þ": "ş", "Þ": "Ş", "ð": "ğ", "Ð": "Ğ", "ý": "ı", "Ý": "İ"})
_LOWER_WORDS = {"ve", "ile", "veya", "da", "de", "için", "adına"}


def fix_turkish_mojibake(s: str) -> str:
    return s.translate(_MOJIBAKE_TR)


def collapse_spaced_letters(s: str) -> str:
    """Collapse PDF spaced-letter typography: 'B A Ş K A N' -> 'BAŞKAN', 'H Ü S E Y İ N  A N G O L E M L İ' -> 'HÜSEYİN ANGOLEMLİ'.
    Uses single-space matching so double spaces between first and last name are preserved!
    """
    def _join(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    s = re.sub(
        r"(?<![A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû.])([A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû](?: [A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]){2,})(?![A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû])",
        _join,
        s,
    )
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s


def tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER_MAP).lower().replace("\u0307", "")


def tr_upper(s: str) -> str:
    return s.translate(_TR_UPPER_MAP).upper()


def tr_title_word(w: str, is_first: bool = False) -> str:
    if not w:
        return w
    wl = tr_lower(w)
    if not is_first and wl in _LOWER_WORDS:
        return wl
    # Handle hyphenated words and apostrophes
    if "-" in w:
        return "-".join(tr_title_word(part, is_first=(i == 0 and is_first)) for i, part in enumerate(w.split("-")))
    # Capitalize first alphabetic char
    chars = list(wl)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = tr_upper(ch)
            break
    return "".join(chars)


def tr_title(s: str) -> str:
    tokens = s.strip().split()
    return " ".join(tr_title_word(tok, is_first=(i == 0)) for i, tok in enumerate(tokens))


# --- Date & Birlesim extraction ---
MONTH_MAP = {
    "ocak": 1,
    "subat": 2,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "mayıs": 5,
    "nayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "ağustos": 8,
    "eylul": 9,
    "eylül": 9,
    "ekim": 10,
    "kasim": 11,
    "kasım": 11,
    "aralik": 12,
    "aralık": 12,
}

MONTH_PATTERN = "|".join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
DATE_TEXT_RE = re.compile(
    rf"\b(\d{{1,2}})\s*\.?\s*({MONTH_PATTERN})\s*,?\s*(19[5-9]\d|20[0-3]\d)\b",
    re.I,
)
DATE_NUM_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](19[5-9]\d|20[0-3]\d)\b")
DATE_NO_DAY_RE = re.compile(
    rf"\b({MONTH_PATTERN})\s*,?\s*(19[5-9]\d|20[0-3]\d)\b",
    re.I,
)

ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20,
}

BIRLESIM_TEXT_RE = re.compile(
    r"\b(\d{1,3}|[IVX]{1,5})\s*(?:['’.]?\s*(?:inci|ıncı|üncü|uncu|nci|ncı|üncü|uncu)?)?\s*(?:BİRLEŞİM|Birleşim|birleşim|İNİKAT|İnikat)\b"
)
INIKAT_NO_RE = re.compile(r"(?:İnikat|Birleşim)\s*(?:No|Na|Sayı)\s*[:.]?\s*(\d{1,3})", re.I)
BIRLESIM_FILE_RE = re.compile(r"(?:^|[^0-9a-zA-Z])(?:[bB][-_\s]*(\d{1,3})|(\d{1,3})[-_\s]*[bB])(?:[^0-9]|$)")

ORDINAL_OTURUM = {
    "birinci": 1, "bilinci": 1, "ikinci": 2, "üçüncü": 3, "ucuncu": 3, "dördüncü": 4, "dorduncu": 4,
    "beşinci": 5, "besinci": 5, "altıncı": 6, "altinci": 6, "yedinci": 7, "sekizinci": 8,
    "dokuzuncu": 9, "onuncu": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
}

OTURUM_RE = re.compile(
    r"^[\s\-–—•·*]*([A-ZÇĞİÖŞÜa-zçğıöşüîÎ1-9\s]{4,35}?)\s+O\s*T\s*U\s*R\s*U\s*M[\s\-–—•·*]*$",
    re.I,
)

CHAIR_HANDOFF_RE = re.compile(
    r"\([^)]*Başkanlık\s+(?:Kürsüsünü|Makamını|Kolt[uü]ğunu|Görevini)\s+(?:.*?Sayın\s+)?([A-ZÇĞİÖŞÜa-zçğıöşü.\s\-]+?)(?:['’][aeyıiüu]+|\s+devreder|\s+devralır)",
    re.I,
)
CHAIR_TAKEOVER_RE = re.compile(
    r"\([^)]*(?:Başkanlık\s+(?:Kürsüsüne|Makamına)\s+(?:.*?Sayın\s+)?([A-ZÇĞİÖŞÜa-zçğıöşü.\s\-]+?)\s+(?:geçti|geçer|oturdu))",
    re.I,
)

# Constituencies & modifiers
CONSTITUENCY_CANON = {
    "lefkoşa": "Lefkoşa",
    "lefkosa": "Lefkoşa",
    "gazimağusa": "Gazimağusa",
    "gazimagusa": "Gazimağusa",
    "gazi mağusa": "Gazimağusa",
    "gazi magusa": "Gazimağusa",
    "mağusa": "Gazimağusa",
    "magusa": "Gazimağusa",
    "girne": "Girne",
    "güzelyurt": "Güzelyurt",
    "guzelyurt": "Güzelyurt",
    "iskele": "İskele",
    "lefke": "Lefke",
}

MODIFIER_PATTERNS = [
    "yerinden", "devamla", "kürsüden", "kursuden", "oturduğu yerden",
    "söz alarak", "tekrar", "komite adına", "grubu adına", "hükümet adına",
    "şahsı adına", "bakanlık sırasından", "araya girerek",
]

# Reject lines whose core uppercase prefix is clearly a structural heading, not a speaker
REJECT_EXACT_OR_PREFIX = {
    "İÇİNDEKİLER", "GÜNDEM", "GELEN EVRAK", "TEZKERELER", "YASA TASARILARI",
    "YASA ÖNERİLERİ", "YASA GÜCÜNDE KARARNAMELER", "SAYIŞTAY RAPORLARI",
    "BAŞKANLIK DİVANI KARARLARI", "DANIŞMA KURULU VE BAŞKANLIK DİVANI KARARLARI",
    "DÖNEM", "YASAMA YILI", "YIL", "TARİH", "SAYI", "KONU", "EK", "MADDE",
    "BİRİNCİ OTURUM", "İKİNCİ OTURUM", "ÜÇÜNCÜ OTURUM", "DÖRDÜNCÜ OTURUM",
    "BEŞİNCİ OTURUM", "TUTANAK DERGİSİ", "KUZEY KIBRIS TÜRK CUMHURİYETİ",
    "CUMHURİYET MECLİSİ", "KIBRIS TÜRK FEDERE MECLİSİ", "B.D.K.NO", "KARAR NO",
    "AÇILIŞ SAATİ", "KAPANIŞ SAATİ", "GÜNDEME GEÇİLİŞ", "NOT", "AÇIKLAMA",
    "GENEL GEREKÇE", "MADDE GEREKÇELERİ", "GEREKÇE", "TASARI", "RAPOR",
    "KOMİTE RAPORU", "HÜKÜMET PROGRAMI", "ÖNERGE", "KARAR", "İMZA",
}

ROLE_KEYWORDS = (
    "BAŞKAN", "BAŞKAM", "EAŞKAN", "BAŞKANLIĞI", "KATİP", "KATIP", "KÂTİP", "KATİF", "BAŞBAKAN",
    "BAKAN", "MİLLETVEKİL", "SÖZCÜ", "CUMHURBAŞKAN", "KOMİTE", "MÜSTEŞAR",
    "GENEL BAŞKAN", "LİDER", "ÜYE", "RAPORTÖR", "TEMSİLCİ", "MÜDÜR",
    "KOMİSER", "NAZIR", "SESLER", "SIRALARDAN", "SALONDAN", "MÜDAHALE",
)

# Ordered longest-first for splitting Title + Person Name
TITLE_BOUNDARY_WORDS = [
    "İKİNCİ BAŞKAN VEKİLİ", "BİRİNCİ BAŞKAN VEKİLİ", "BAŞKAN YARDIMCISI", "BAŞKAN VEKİLİ",
    "GENEL BAŞKAN YARDIMCISI", "GENEL BAŞKANI", "GENEL BAŞKAN", "GENEL SEKRETERİ",
    "GENEL SEKRETER", "CUMHURBAŞKANI MUAVİNİ", "CUMHURBAŞKANI VEKİLİ", "CUMHURBAŞKANI",
    "BAŞBAKAN YARDIMCISI", "BAŞBAKAN VEKİLİ", "BAŞBAKANI", "BAŞBAKAN", "DEVLET BAKANI",
    "BAKANLIĞI", "BAKANI", "BAKAMI", "BAKAN", "MİLLETVEKİLİ", "SÖZCÜSÜ", "MÜSTEŞARI",
    "LİDERİ", "ÜYESİ", "RAPORTÖRÜ", "TEMSİLCİSİ", "MÜDÜRÜ", "KOMİSERİ", "NAZIRI",
    "YARDIMCISI", "VEKİLİ", "BAŞKANI", "BAŞKAN", "BAŞKAM", "EAŞKAN",
    "KATİBİ", "KATİP", "KATIP", "KATİF", "KÜRSÜSÜNDEN", "ADINA",
]
TITLE_BOUNDARY_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TITLE_BOUNDARY_WORDS) + r")\b"
)

ACADEMIC_PREFIX_RE = re.compile(
    r"^(?:(?:Prof|Doç|Yrd\.?\s*Doç|Dr|Dt|Dri|D\s+R|Ör|Bt|Ot|Av|Müh|Uzm|Sayın|Sn)[.»',#•/]*\s*\.?\s*)+",
    re.I,
)
# Only strip Dr./Dt. without space when followed by a capitalized/letter name (e.g., Dr.Derviş, Dt.Sait)
ACADEMIC_TIGHT_RE = re.compile(r"^(?:£>R|Dr|Dt|Dri|Ör|Prof|Doç|Av)[.,/#•]\s*", re.I)

SPEECH_OPENERS_LOWER = (
    "sayın", "saygıdeğer", "değerli", "muhterem", "arkadaşlar", "toplantı",
    "teşekkür", "cumhuriyet", "açıyorum", "vardır", "var ", "evet", "hayır",
    "buyurun", "kabul", "şimdi", "söz", "nisap", "nisab", "okuyorum", "gelecek",
    "oturumu", "birleşimi", "özür", "peki", "lütfen", "gündem", "dinleyelim",
    "ad okunarak", "yoklama", "maalesef", "milletvekilleri",
)

CHAIR_REJECT_SUBSTRS = (
    "nisap", "nisab", "mevcuttur", "vardır", "yoktur", "yoklama", "yapıldı",
    "dinleyelim", "efendim", "lütfen", "sayın başkan", "buyurun", "teşekkür",
    "açıyorum", "okunarak", "kabul", "toplantı", "yeter sayısı", "birleşim",
    "gündem", "oturum", "evet", "hayır", "maalesef", "bir dakika", "arkadaşlar",
    "geldi", "siyasal hak", "madde", "kanun", "yasa", "tasarı", "milletvekilleri",
    "cumhuriyet", "meclis", "dönem", "sayı", "tarih", "yanıt", "verecek", "kısım",
)

CHAIR_HEADER_PREFIX_RE = re.compile(
    r"^(?:Mecliste\s+Bulunan\s+)?(?:En\s+Yaşlı\s+Üye\s+Sıfat[ıi](?:yla|\s+ile)\s+)?(?:Geçici\s+)?(?:Cumhuriyet\s+Meclisi\s+|Meclis\s+)?(?:Başkanlık\s+)?(?:Başkan\s+Yardımcısı|Başkan\s+Vekili|Başkanı|Başkan|K[âaäföl]tibi|Katibi|Katip|Katipler|Kâtip|Kâtipler|Sayın|Sn\.?)\s+",
    re.I,
)


MECLIS_MIN_YEAR = {
    "KKTC": 1985,
    "KKTC_ARA": 1983,
    "KKTC_KM": 1983,
    "KTFD": 1975,
    "KTFD_KM": 1975,
    "KTY": 1967,
    "TCM": 1960,
    "KC": 1960,
}


def extract_date(header_lines: list[str], meclis: str = "KKTC", filename: str = "") -> str | None:
    min_yr = MECLIS_MIN_YEAR.get(meclis, 1960)
    # For historic 1967-1970 KTY Dönem I files whose filename explicitly carries DD-MM-YYYY
    # (e.g. "O.25-03-1970.pdf", "V.24-07-1970.pdf"), prefer filename date over earlier cover-letter dates.
    if meclis == "KTY" and filename:
        m_fn = re.search(r"(?:^|[.\s_])(\d{1,2})-(\d{1,2})-(19[67]\d)(?:[^0-9]|$)", filename)
        if m_fn:
            day, month, year = int(m_fn.group(1)), int(m_fn.group(2)), int(m_fn.group(3))
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}-{day:02d}"
    # Search the cover/title lines (first 25 lines) first so TOC references to old laws don't override session date
    windows = ["\n".join(header_lines[:25]), "\n".join(header_lines[:100])]
    if filename:
        windows.append(filename)
    for header_text in windows:
        for m in DATE_TEXT_RE.finditer(header_text):
            day = int(m.group(1))
            month = MONTH_MAP[tr_lower(m.group(2))]
            year = int(m.group(3))
            if 1 <= day <= 31 and min_yr <= year <= 2027:
                return f"{year:04d}-{month:02d}-{day:02d}"
        for m in DATE_NUM_RE.finditer(header_text):
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= day <= 31 and 1 <= month <= 12 and min_yr <= year <= 2027:
                return f"{year:04d}-{month:02d}-{day:02d}"
        # Also support DD-MM-YYYY in historic filenames like "A.26-10-1967.pdf"
        for m in re.finditer(r"\b(\d{1,2})-(\d{1,2})-(19[5-9]\d|20[0-3]\d)\b", header_text):
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= day <= 31 and 1 <= month <= 12 and min_yr <= year <= 2027:
                return f"{year:04d}-{month:02d}-{day:02d}"
    for header_text in windows:
        for m in DATE_NO_DAY_RE.finditer(header_text):
            month = MONTH_MAP[tr_lower(m.group(1))]
            year = int(m.group(2))
            if min_yr <= year <= 2027:
                return f"{year:04d}-{month:02d}"
    return None


def extract_birlesim(header_text: str, filename: str) -> int | None:
    m = BIRLESIM_TEXT_RE.search(header_text)
    if m:
        tok = m.group(1).upper()
        if tok.isdigit():
            return int(tok)
        if tok in ROMAN_MAP:
            return ROMAN_MAP[tok]
    m = INIKAT_NO_RE.search(header_text)
    if m:
        return int(m.group(1))
    m = BIRLESIM_FILE_RE.search(filename)
    if m:
        val = m.group(1) or m.group(2)
        if val and val.isdigit():
            return int(val)
    return None


def parse_oturum_line(line: str) -> int | None:
    s = line.strip()
    if len(s) > 45:
        return None
    m = OTURUM_RE.match(s)
    if not m:
        return None
    # Collapse OCR spaced letters like "b i r i n c i" -> "birinci"
    raw = re.sub(r"\s+", "", tr_lower(m.group(1))).replace("î", "i").replace(".", "").replace("’", "").replace("'", "")
    for k, v in ORDINAL_OTURUM.items():
        if raw.startswith(k):
            return v
    return None


def clean_person_name(raw: str) -> str:
    s = fix_turkish_mojibake(raw).strip().strip(".,;:-–—()•·*'’\" ")
    if any(ch in s for ch in (";", "?", "!", "\t")) or any(ch.isdigit() for ch in s):
        return ""
    # Remove parenthetical explanations, even if unclosed due to line wrap
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\(.*$", "", s).strip()
    s = collapse_spaced_letters(s)
    s = re.sub(r"[*»«^#£>]", " ", s)
    # Repeatedly strip leading role/honorific/academic prefixes
    for _ in range(6):
        s2 = CHAIR_HEADER_PREFIX_RE.sub("", s).strip(" .,;:-–—'’\"")
        s2 = ACADEMIC_TIGHT_RE.sub("", s2).strip(" .,;:-–—'’\"")
        s2 = re.sub(r"^(?:(?:Prof|Doç|Dr|Dt|Dri|Ör|D\s*R|Av|Müh|Sayın|Sn)[.,/#•]*\s+)+", "", s2, flags=re.I).strip(" .,;:-–—'’\"")
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\s+", " ", s).strip(".,;:-–—'’\" ")
    if not s or len(s) > 48 or len(s.split()) > 5:
        return ""
    if any(bad in tr_lower(s) for bad in CHAIR_REJECT_SUBSTRS):
        return ""
    return tr_title(s)


def looks_like_chair_declaration(role_tok: str, sep: str, rest: str, lines_since_oturum: int) -> bool:
    """Distinguish 'BAŞKAN: Meclis Başkan Vekili Ünal ÜSTEL' from speeches."""
    rest_s = rest.strip()
    if not rest_s or len(rest_s) > 95:
        return False
    rl = tr_lower(rest_s)
    if any(rl.startswith(op) for op in SPEECH_OPENERS_LOWER):
        return False
    if "?" in rest_s or "!" in rest_s or ";" in rest_s:
        return False
    words = rest_s.split()
    if len(words) > 9:
        return False
    # Strip legitimate chair title prefixes BEFORE checking reject words (so "Meclis Başkan Vekili Ünal Üstel" passes!)
    person = clean_person_name(rest_s)
    if not person or len(person.split()) < 2:
        return False
    if sep == ":":
        return True
    # With dash, only treat as chair declaration right after OTURUM header
    if lines_since_oturum <= 7:
        if any(k in rl for k in ("geçici başkan", "geçici katip", "başkanlık", "yardımcısı", "vekili")) or len(words) <= 4:
            return True
    return False


def is_page_number_or_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if re.match(r"^[-–—./\s]*\d{1,4}[-–—./\s]*$", s):
        return True
    if re.match(r"^[\-_–—=.,:;*•·\s]+$", s):
        return True
    return False


def split_speaker_candidate(line: str) -> tuple[str, str, str] | None:
    """
    If `line` starts with a speaker label followed by a dash/colon separator,
    return (label_raw, sep, speech_rest). Otherwise None.
    """
    s = line.strip()
    if not s or s.startswith("("):
        return None
    # Strip occasional OCR bullet/dots/leading stray dash before uppercase speaker label
    s = re.sub(r"^[•·▪►\*.\-–—'\s]+(?=[A-ZÇĞİÖŞÜ])", "", s)
    if not s or not s[0].isalpha():
        return None

    # Normalize spaced-out letters on the prefix side for reliable matching
    # (only touch prefix before the first dash/colon/tilde so speech body isn't altered)
    m_pre = re.match(r"^([^:\-–—~\n]{2,155})([:\-–—~])(.*)$", s)
    if m_pre:
        s_norm = collapse_spaced_letters(m_pre.group(1)) + m_pre.group(2) + m_pre.group(3)
    else:
        s_norm = s

    # Special check for session header with colon (case-insensitive: "Başkan: ...", "BAŞKAN VEKİLİ : Zorlu TÖRE", "Kâtipler: ...")
    m_col = re.match(
        r"^[.\s]*((?:Geçici\s+|Meclis\s+|Oturum\s+)?(?:Başkan\s+Vekili|Başkan\s+Yardımcısı|Başkanı|Başkan|Başkam|Eaşkan|Katip|Kâtip|Katipler|Kâtipler|Kâtipltr|BAŞKAN\s+VEKİLİ|BAŞKAN\s+YARDIMCISI|BAŞKANI|BAŞKAN|BAŞKAM|EAŞKAN|KATİP|KATIP|KATİPLER|KÂTİP|KÂTİPLER|KATİF))\s*(:)\s*(.*)$",
        s_norm,
        re.I,
    )
    if m_col:
        return tr_upper(m_col.group(1).strip()), ":", m_col.group(3).strip()

    # Main speaker turn separator: dash (hyphen / en-dash / em-dash / OCR tilde)
    patterns = [
        r"^([^\-–—~\n]{2,150}?)\s+([-–—~])(?:\s+|\.)(.*)$",
        r"^([^\-–—~\n]{2,150}?)\s+([-–—~])\s*$",
        r"^([^\-–—~\n]{2,150}?\))\s*([-–—~])\s*(.*)$",
        r"^((?:GEÇİCİ\s+)?(?:BAŞKAN|BAŞKAM|EAŞKAN|KATİP|KATIP|KÂTİP|KATİF))\s*([-–—~])\s*(.*)$",
        r"^([^\-–—~\n]{8,150}?(?:BAKANI|BAŞBAKAN|BAŞKANI|MİLLETVEKİLİ)\s+[A-ZÇĞİÖŞÜ.\s]{4,50}?)\s*([-–—~])\s*$",
        r"^([A-ZÇĞİÖŞÜ\s.,]{4,140}?)\s*([-–—~])\s+(.*)$",
    ]
    label, sep, rest = None, None, None
    for pat in patterns:
        m = re.match(pat, s_norm)
        if m:
            label = m.group(1).strip()
            sep = m.group(2)
            rest = m.group(3).strip() if m.lastindex >= 3 else ""
            break
    if not label:
        return None

    # Validate label outside parentheses
    core = re.sub(r"\([^)]*\)", " ", label)
    core = collapse_spaced_letters(core)
    core = re.sub(r"\s+", " ", core).strip(" .,;:")
    if not core or len(core) < 3 or len(core) > 135:
        return None
    if any(ch.isdigit() for ch in core):
        return None

    alpha_chars = [c for c in core if c.isalpha()]
    if len(alpha_chars) < 3:
        return None
    upper_chars = [c for c in alpha_chars if c.isupper() or c in "ÇĞİÖŞÜÂÎÛ"]
    if len(upper_chars) / len(alpha_chars) < 0.78:
        return None

    core_up = tr_upper(core.translate(_TR_NO_CFX))
    if core_up in REJECT_EXACT_OR_PREFIX:
        return None
    for rej in ("DÖNEM", "YASAMA YILI", "BİRİNCİ OTURUM", "İKİNCİ OTURUM", "ÜÇÜNCÜ OTURUM", "KUZEY KIBRIS", "CUMHURİYET MECLİSİ", "KISIM ", "BÖLÜM "):
        if core_up == rej.strip() or core_up.startswith(rej):
            return None

    words = core_up.split()
    if len(words) > 18:
        return None

    # Must either contain a known role keyword OR look like a 2-5 word person name
    has_role = any(rk in core_up for rk in ROLE_KEYWORDS)
    if not has_role:
        has_parens = "(" in label
        if len(words) < 2 and not has_parens:
            return None
        if len(words) > 6:
            return None
        if any(p in core for p in (":", ";", "?", "!", '"', "“")):
            return None

    return label, sep, rest


def parse_speaker_label(
    label_raw: str,
    current_chair: str | None,
    current_clerk: str | None,
    doc_title_to_name: dict[str, str],
) -> dict:
    parens = re.findall(r"\(([^)]+)\)", label_raw)
    constituency = None
    modifiers = []
    for p in parens:
        pl = tr_lower(p).strip()
        if pl in CONSTITUENCY_CANON:
            constituency = CONSTITUENCY_CANON[pl]
        elif any(mp in pl for mp in MODIFIER_PATTERNS):
            modifiers.append(tr_title(p.strip()))
        else:
            # Could be OCR-damaged district or modifier
            matched = False
            for k, v in CONSTITUENCY_CANON.items():
                if k in pl:
                    constituency = v
                    matched = True
                    break
            if not matched and len(p.strip()) <= 30:
                modifiers.append(tr_title(p.strip()))

    core = re.sub(r"\([^)]*\)", " ", label_raw)
    core = collapse_spaced_letters(core)
    # Fix stray tab inside names (e.g. "HASAN K\tÜÇÜK" -> "HASAN KÜÇÜK" if single letter + tab + word)
    core = re.sub(r"\b([A-ZÇĞİÖŞÜ])\t([A-ZÇĞİÖŞÜ]{2,})\b", r"\1\2", core)
    core = core.replace("\t", " ")
    # Clean stray OCR punctuation inside names (e.g. "EŞBER*SERÂKINCI", "GUNAY.CAYMAZ", "Dr» DERVİŞ,EROGLU")
    core = re.sub(r"[*»«^#£>]", " ", core)
    core = re.sub(r"(?<=[A-ZÇĞİÖŞÜa-zçğıöşü]),(?=[A-ZÇĞİÖŞÜa-zçğıöşü])", " ", core)
    core = re.sub(r"(?<=[A-ZÇĞİÖŞÜ]{2})\.(?=[A-ZÇĞİÖŞÜ]{2})", " ", core)
    core = re.sub(r"\s+", " ", core).strip(" .,;:'’\"")
    core_up = tr_upper(core.translate(_TR_NO_CFX))

    # 1. Pure Chair (BAŞKAN / GEÇİCİ BAŞKAN / BAŞKAM / EAŞKAN / İKİNCİ BAŞKAN VEKİLİ)
    if core_up in (
        "BAŞKAN", "BAŞKAM", "EAŞKAN", "GEÇİCİ BAŞKAN", "GEÇİCİ BAŞKAM",
        "MECLİS BAŞKANI", "OTURUM BAŞKANI", "BAŞKAN VEKİLİ", "BAŞKAN YARDIMCISI",
        "İKİNCİ BAŞKAN VEKİLİ", "BİRİNCİ BAŞKAN VEKİLİ",
    ):
        title = "Geçici Başkan" if "GEÇİCİ" in core_up else ("Meclis Başkanı" if core_up in ("BAŞKAN", "BAŞKAM", "EAŞKAN") else tr_title(core))
        return {
            "speaker_raw": label_raw,
            "speaker_name": current_chair or title,
            "speaker_role": "gecici_baskan" if "GEÇİCİ" in core_up else "baskan",
            "speaker_title": title,
            "constituency": constituency,
            "modifiers": "; ".join(modifiers) if modifiers else None,
        }

    # 2. Pure Clerk (KATİP / GEÇİCİ KATİP / KATİF)
    if core_up in ("KATİP", "KATIP", "KÂTİP", "KATİF", "GEÇİCİ KATİP", "GEÇİCİ KATIP", "KATİPLER", "KÂTİPLER"):
        title = "Geçici Katip" if "GEÇİCİ" in core_up else "Katip"
        return {
            "speaker_raw": label_raw,
            "speaker_name": current_clerk or title,
            "speaker_role": "katip",
            "speaker_title": title,
            "constituency": constituency,
            "modifiers": "; ".join(modifiers) if modifiers else None,
        }

    # 3. Floor/audience noise
    if any(k in core_up for k in ("SESLER", "SIRALARDAN", "SALONDAN", "BİR MİLLETVEKİLİ", "BAZI MİLLETVEKİLLERİ", "MİLLETVEKİLLERİ")):
        return {
            "speaker_raw": label_raw,
            "speaker_name": tr_title(core),
            "speaker_role": "diger",
            "speaker_title": None,
            "constituency": constituency,
            "modifiers": "; ".join(modifiers) if modifiers else None,
        }

    # 4. Title + Name or Name only
    matches = list(TITLE_BOUNDARY_RE.finditer(core_up))
    speaker_title = None
    name_part = core
    if matches:
        last_m = matches[-1]
        title_part = core[: last_m.end()].strip(" ,.;:-")
        after_part = core[last_m.end() :].strip(" ,.;:-")
        if after_part and len(after_part) >= 3:
            speaker_title = tr_title(title_part)
            name_part = after_part
            # Remember mapping within this document for bare title references
            doc_title_to_name[tr_upper(title_part)] = tr_title(after_part)
        else:
            # Bare title only (e.g., "BAŞBAKAN - ...", "MALİYE BAKANI - ...")
            speaker_title = tr_title(title_part)
            resolved = doc_title_to_name.get(tr_upper(title_part))
            name_part = resolved if resolved else speaker_title

    speaker_name = tr_title(name_part)
    # Strip academic/professional prefixes (Dr., Prof. Dr., Dt., Av., Sayın, OCR Dri/Ör/£>R) into clean canonical form
    speaker_name = ACADEMIC_TIGHT_RE.sub("", speaker_name).strip(" .,;:'’\"")
    speaker_name = re.sub(r"^(?:(?:Prof|Doç|Dr|Dt|Dri|Ör|D\s*R|Av|Müh|Uzm|Sayın|Sn)[.,/#•]*\s+)+", "", speaker_name, flags=re.I).strip(" .,;:'’\"")
    # Remove trailing ", Yok" from roll-call lines mistakenly parsed as names
    speaker_name = re.sub(r",\s*Yok$", "", speaker_name, flags=re.I).strip()
    # Fix missing space after middle initial dot like "Ferdi S.soyer" -> "Ferdi S. Soyer"
    speaker_name = re.sub(r"\b([A-ZÇĞİÖŞÜ])\.([A-ZÇĞİÖŞÜa-zçğıöşü]{2,})", lambda m: f"{m.group(1)}. {tr_title(m.group(2))}", speaker_name)
    # Fix space before middle initial dot like "Naci T . Usar" -> "Naci T. Usar"
    speaker_name = re.sub(r"\b([A-ZÇĞİÖŞÜ])\s+\.\s*", r"\1. ", speaker_name)
    # Strip single stray leading OCR letter like "V Mehmet Bayram" -> "Mehmet Bayram"
    if re.match(r"^[A-ZÇĞİÖŞÜ]\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}", speaker_name):
        if speaker_name[0] not in ("M", "A", "H", "R", "S", "İ", "O", "K", "T", "E", "N", "F", "C", "B", "D"):
            speaker_name = speaker_name[2:].strip()

    return {
        "speaker_raw": label_raw,
        "speaker_name": speaker_name,
        "speaker_role": "milletvekili_bakan",
        "speaker_title": speaker_title,
        "constituency": constituency,
        "modifiers": "; ".join(modifiers) if modifiers else None,
    }


def clean_speech_body(lines: list[str], is_pdf: bool) -> str:
    if not lines:
        return ""
    out_lines = []
    for line in lines:
        s = line.strip()
        if is_page_number_or_noise(s):
            continue
        if is_pdf and out_lines and out_lines[-1].endswith("-") and s and s[0].islower():
            # De-hyphenate wrapped Turkish word from scanned PDF
            out_lines[-1] = out_lines[-1][:-1] + s
        else:
            out_lines.append(s)
    text = "\n".join(out_lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def merge_wrapped_pdf_speaker_lines(lines: list[str]) -> list[str]:
    """Re-join 2-line wrapped speaker headers in PDFs (both scanned & 2025/2026 digital PDFs):
    - 'BAŞKAN' + '- Sayın milletvekilleri...'
    - 'ÖZKAN MURAT' + '(Devamla) - ...'
    - 'KOMİTESİ BAŞKANI KENAN' + 'AKIN - ...'
    """
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i].strip()
        if i + 1 < n and cur and not cur.startswith("(") and 4 <= len(cur) <= 110:
            if not re.search(r"[-–—:;.!?]$", cur) and not any(ch.isdigit() for ch in cur):
                core_cur = re.sub(r"\([^)]*\)", " ", cur)
                alphas = [c for c in core_cur if c.isalpha()]
                if len(alphas) >= 4 and sum(1 for c in alphas if c.isupper() or c in "ÇĞİÖŞÜÂÎÛ") / len(alphas) >= 0.82:
                    nxt = lines[i + 1].strip()
                    # Case 0: next line starts directly with dash ("- Sayın milletvekilleri...")
                    if re.match(r"^[-–—](?:\s+|$)", nxt):
                        out.append(cur + " " + nxt)
                        i += 2
                        continue
                    # Case 1: next line starts with (Yerinden)/(Devamla)/(District) followed by dash
                    if re.match(r"^\([A-ZÇĞİÖŞÜa-zçğıöşü\s.]{3,35}\)\s*[-–—]", nxt):
                        out.append(cur + " " + nxt)
                        i += 2
                        continue
                    # Case 2: next line is 1-3 ALL-CAPS words (surname) followed by optional (...) and dash
                    m_sur = re.match(r"^([A-ZÇĞİÖŞÜ]{2,25}(?:\s+[A-ZÇĞİÖŞÜ]{2,25}){0,2}(?:\s*\([^)]+\))?)\s*[-–—](?:\s+|$)", nxt)
                    if m_sur and len(cur.split()) >= 2:
                        out.append(cur + " " + nxt)
                        i += 2
                        continue
        out.append(lines[i])
        i += 1
    return out


def parse_document(doc: dict) -> tuple[dict, list[dict]]:
    text = fix_turkish_mojibake(doc.get("text") or "")
    lines = text.splitlines()
    is_pdf = (doc.get("text_source") or "").startswith("pdf")
    if is_pdf:
        lines = merge_wrapped_pdf_speaker_lines(lines)

    header_window = "\n".join(lines[:100])
    tarih = extract_date(lines, meclis=doc.get("meclis", "KKTC"), filename=doc.get("filename", ""))
    birlesim = extract_birlesim(header_window, doc["filename"])

    # Pre-scan TOC / header for title -> name mappings (e.g., "Başbakan Sayın Ünal Üstel'in ...")
    doc_title_to_name: dict[str, str] = {}
    for line in lines[:80]:
        m_toc = re.search(
            r"((?:Meclis\s+Başkanı|Başbakan\s+Yardımcısı[^,]*|Başbakan|[A-ZÇĞİÖŞÜa-zçğıöşü,\s]+Bakanı))\s+Sayın\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,2})",
            line,
        )
        if m_toc:
            t_key = tr_upper(m_toc.group(1).strip())
            p_name = re.sub(r"[’'](?:nın|nin|nun|nün|ın|in|un|ün)$", "", m_toc.group(2).strip())
            doc_title_to_name[t_key] = tr_title(p_name)

    current_oturum = 0
    lines_since_oturum = 999
    current_chair: str | None = None
    current_clerk: str | None = None
    seen_chairs: list[str] = []

    turns: list[dict] = []
    cur_turn: dict | None = None
    cur_body: list[str] = []

    def flush_turn():
        nonlocal cur_turn, cur_body
        if cur_turn is not None:
            body_text = clean_speech_body(cur_body, is_pdf=is_pdf)
            if body_text:
                cur_turn["text"] = body_text
                cur_turn["char_count"] = len(body_text)
                cur_turn["word_count"] = len(body_text.split())
                turns.append(cur_turn)
        cur_turn = None
        cur_body = []

    pending_header_role: str | None = None

    for line in lines:
        stripped = line.strip()
        lines_since_oturum += 1

        # 1. Check Oturum transition
        ot = parse_oturum_line(stripped)
        if ot is not None:
            flush_turn()
            current_oturum = ot
            lines_since_oturum = 0
            pending_header_role = None
            continue

        # 1b. Handle two-line chair/clerk headers in old PDFs:
        #     Line N:   "Başkan" or "BAŞKAN" or "Kâtipler"
        #     Line N+1: ": Başkan Yardımcısı Dt. Sait Güven" or "-' Geçici Başkan Ziya M. Rızkı"
        if lines_since_oturum <= 10:
            if pending_header_role and stripped:
                m_val = re.match(r"^[:\-–—'’.\s]+(.+)$", stripped)
                val_raw = m_val.group(1) if m_val else stripped
                person = clean_person_name(val_raw)
                if person and len(person.split()) >= 2 and len(person) <= 60:
                    if pending_header_role == "baskan":
                        current_chair = person
                        if person not in seen_chairs:
                            seen_chairs.append(person)
                    else:
                        current_clerk = person
                pending_header_role = None
                continue
            # Only treat bare "Başkan" / "Kâtipler" (NOT "KATİP -" speech turns!) as two-line headers
            if not stripped.endswith(("-", "–", "—")):
                low_bare = tr_lower(stripped.strip(" .•·:"))
                if low_bare in ("başkan", "baskan", "geçici başkan", "gecici baskan"):
                    pending_header_role = "baskan"
                    continue
                if low_bare in ("katip", "kâtip", "katipler", "kâtipler", "kâtipltr", "geçici katip"):
                    pending_header_role = "katip"
                    continue

        # 2. Check mid-session chair handoff stage direction
        if stripped.startswith("(") and "Başkanlık" in stripped:
            mh = CHAIR_HANDOFF_RE.search(stripped) or CHAIR_TAKEOVER_RE.search(stripped)
            if mh:
                new_chair = clean_person_name(mh.group(1))
                if new_chair and len(new_chair.split()) >= 2:
                    current_chair = new_chair
                    if new_chair not in seen_chairs:
                        seen_chairs.append(new_chair)
            if cur_turn is not None:
                cur_body.append(stripped)
            continue

        # 3. Check speaker turn or chair/clerk header
        cand = split_speaker_candidate(stripped)
        if cand is not None:
            label_raw, sep, rest = cand
            core_up = tr_upper(re.sub(r"\([^)]*\)", " ", label_raw).strip())

            # Check if this is a chair/clerk session header declaration
            if core_up in (
                "BAŞKAN", "BAŞKAM", "EAŞKAN", "GEÇİCİ BAŞKAN", "MECLİS BAŞKANI", "OTURUM BAŞKANI",
                "BAŞKAN VEKİLİ", "BAŞKAN YARDIMCISI", "MECLİS BAŞKAN VEKİLİ", "MECLİS BAŞKAN YARDIMCISI",
                "KATİP", "KATIP", "KÂTİP", "KATİPLER", "KÂTİPLER", "KÂTİPLTR", "KATİF",
                "GEÇİCİ KATİP", "GEÇİCİ KATİPLER",
            ):
                if looks_like_chair_declaration(core_up, sep, rest, lines_since_oturum):
                    person = clean_person_name(rest)
                    if person and len(person.split()) >= 2:
                        if "BAŞK" in core_up:
                            current_chair = person
                            if person not in seen_chairs:
                                seen_chairs.append(person)
                        else:
                            current_clerk = person
                    continue

            # Ignore colon-separated lines unless handled above (those are document metadata like DÖNEM: X)
            if sep == ":":
                if cur_turn is not None:
                    cur_body.append(stripped)
                continue

            # Valid speaker turn!
            flush_turn()
            # If we haven't seen an explicit OTURUM marker yet, mark as session 1 once speeches start
            if current_oturum == 0:
                current_oturum = 1

            spk = parse_speaker_label(label_raw, current_chair, current_clerk, doc_title_to_name)
            cur_turn = {
                "doc_id": doc["local_rel_path"],
                "meclis": doc["meclis"],
                "donem": doc["donem"],
                "yasama_yili": doc["yasama_yili"],
                "birlesim": birlesim,
                "oturum": current_oturum,
                "tarih": tarih,
                **spk,
                "text_source": doc.get("text_source"),
                "source_file": doc["local_rel_path"],
                "source_url": doc.get("download_url"),
            }
            if rest:
                cur_body.append(rest)
            continue

        # 4. Continuation line of current speech
        if cur_turn is not None:
            cur_body.append(stripped)

    flush_turn()

    # Fallback: Dedicated 1960-1970 historic colon-format parser (KTY Dönem I, Kıbrıs Cumhuriyeti, TCM)
    # Only activates when standard dash-parser found <10 turns in a historic 1960-1970 document.
    if doc.get("meclis") in ("KC", "TCM", "KTY") and len(turns) < 10 and len(text) > 1000:
        hist_turns, hist_chairs = parse_historic_colon_turns(doc, lines, tarih, birlesim)
        if len(hist_turns) > len(turns):
            turns = hist_turns
            seen_chairs = hist_chairs

    # Assign sequential turn_index and speech_id
    doc_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", doc["local_rel_path"]).strip("_")
    for idx, t in enumerate(turns):
        t["turn_index"] = idx
        t["speech_id"] = f"{doc_slug}#o{t['oturum']}#t{idx:04d}"

    unique_speakers = sorted({t["speaker_name"] for t in turns if t["speaker_name"]})
    doc_record = {
        "doc_id": doc["local_rel_path"],
        "meclis": doc["meclis"],
        "donem": doc["donem"],
        "yasama_yili": doc["yasama_yili"],
        "birlesim": birlesim,
        "tarih": tarih,
        "oturum_sayisi": max((t["oturum"] for t in turns), default=0),
        "konusma_sayisi": len(turns),
        "konusmacilar": unique_speakers,
        "baskanlar": seen_chairs,
        "text": text,
        "char_count": len(text),
        "page_count": doc.get("page_count"),
        "text_source": doc.get("text_source"),
        "source_file": doc["local_rel_path"],
        "source_url": doc.get("download_url"),
        "sha256": doc.get("sha256"),
    }
    return doc_record, turns


# --- Dedicated 1960-1970 Historic Colon-Format Parser (KC, TCM, KTY Dönem I) ---

_HISTORIC_ALIASES = {
    "ü. süleyman": "Ümit Süleyman",
    "ü.süleyman": "Ümit Süleyman",
    "üosüleyman": "Ümit Süleyman",
    "süleyman": "Ümit Süleyman",
    "ümit süleyman": "Ümit Süleyman",
    "a. mithat berberoğlu": "Ahmet Mithat Berberoğlu",
    "a.mithat berberoğlu": "Ahmet Mithat Berberoğlu",
    "a. m. berberoğlu": "Ahmet Mithat Berberoğlu",
    "a. m. berberoglu": "Ahmet Mithat Berberoğlu",
    "a. m. berberoölu": "Ahmet Mithat Berberoğlu",
    "mithat berberoğlu": "Ahmet Mithat Berberoğlu",
    "bay berberoğlu": "Ahmet Mithat Berberoğlu",
    "berberoğlu": "Ahmet Mithat Berberoğlu",
    "doktor orhan bey": "Orhan Müderrisoğlu",
    "dr. orhan müderrisoğlu": "Orhan Müderrisoğlu",
    "orhan müderrisoğlu": "Orhan Müderrisoğlu",
    "müderrisoğlu": "Orhan Müderrisoğlu",
    "denktaş": "Rauf Raif Denktaş",
    "başkan denktaş": "Rauf Raif Denktaş",
    "başkan denkpaş": "Rauf Raif Denktaş",
    "rauf r. denktaş": "Rauf Raif Denktaş",
    "rauf r. denktag": "Rauf Raif Denktaş",
    "başkan rauf r. denktaş": "Rauf Raif Denktaş",
    "başkan rauf r. denktag": "Rauf Raif Denktaş",
    "h. a. rıza": "Halit Ali Rıza",
    "h. a. riza": "Halit Ali Rıza",
    "h. a. rtza": "Halit Ali Rıza",
    "h. a. rîza": "Halit Ali Rıza",
    "b. nalbantoğlu": "Burhan Nalbantoğlu",
    "b. nalbantoglu": "Burhan Nalbantoğlu",
    "hasan guvener": "Hasan Güvener",
    "hasan güvener": "Hasan Güvener",
    "dr. hasan güvener": "Hasan Güvener",
    "dr. fazıl küçük": "Fazıl Küçük",
    "fazıl küçük": "Fazıl Küçük",
    "dr. niyazi manyera": "Niyazi Manyera",
    "niyazi manyera": "Niyazi Manyera",
    "glafkos kliridis": "Glafkos Kliridis",
    "bay glafkos kliridis": "Glafkos Kliridis",
}

_HIST_REJECT_WORDS = (
    "saat", "tarih", "madde", "konu", "sayı", "gündem", "bölüm", "bolum", "bolom",
    "kapanış", "açılış", "not", "cetvel", "hazır", "encümen", "komite", "daire",
    "müdür", "üyeleri", "sayısı", "isimleri", "teklif", "rapor", "kanun", "tasarı",
    "kural", "fıkra", "hakkında", "olanlar", "bulunanlar", "dağıtılan", "içtima",
    "inikat", "rasırlıyan", "hazırlayan", "meclis idaresi", "ek", "e k", "e e",
    "s e", "devre", "yıl", "dönem", "tutana", "zabıt", "imza", "karar", "sonuç",
    "lehte", "aleyhte", "çekimser", "toplam", "nisap", "yoklama", "devamla",
)

# Reject Greek-script OCR artifacts in 1960 bilingual Cyprus House minutes (e.g. TPAMMATEYZ, AHMHTPIAAHZ, IİPOEAPOZ)
_GREEK_OCR_RE = re.compile(r"(?:AAHZ|IAHZ|ATEYZ|EAPOZ|ZABBI|XP\.|IİP|TPAMM)", re.I)


def _normalize_historic_name(raw_label: str, current_chair: str | None) -> tuple[str, str, str | None] | None:
    """Return (speaker_name, speaker_role, speaker_title) or None if not a valid 1960-1970 speaker."""
    s = re.sub(r"\s+", " ", raw_label).strip(" .,;:-–—'’\"*^")
    if not s or len(s) < 3 or len(s) > 48:
        return None
    if any(ch.isdigit() for ch in s):
        return None
    if _GREEK_OCR_RE.search(s):
        return None

    sl = tr_lower(s)
    # Direct reject of structural/procedural labels
    if sl in _HIST_REJECT_WORDS or any(w in sl for w in _HIST_REJECT_WORDS):
        # Exception: "Meclis Reisi Glafkos Kliridis", "Geçici Reis Vekili Dr. Niyazi Manyera", "Başkan Denktaş"
        if not re.match(r"^(?:geçici\s+|meclis\s+)?(?:reis|başkan)", sl):
            return None

    # Check explicit alias map first
    if sl in _HISTORIC_ALIASES:
        role = "baskan" if ("başkan" in sl or "reis" in sl) else "milletvekili_bakan"
        title = "Meclis Başkanı" if role == "baskan" else None
        return _HISTORIC_ALIASES[sl], role, title

    # Pure Chair / Reis / Başkan / Başkan Vekili
    if sl in ("reis", "reîs", "reas", "reıs", "reis vekili", "reîs vekili", "başkan", "baskan",
              "başkan vekili", "meclis reisi", "geçici meclis reisi", "geçici reis vekili", "meclis başkanı"):
        title = "Reis Vekili" if "vekil" in sl else ("Meclis Reisi" if "reis" in sl else "Meclis Başkanı")
        return (current_chair or title), "baskan", title

    # Pure Clerk / Katip
    if sl in ("katip", "kâtip", "katîp", "kâtipler", "katipler"):
        return "Katip", "katip", "Katip"

    # Named Chair: "Meclis Reisi Glafkos Kliridis", "Geçici Reis Vekili Dr. Niyazi Manyera", "Başkan X"
    m_ch = re.match(
        r"^((?:geçici\s+)?(?:meclis\s+)?(?:reis(?:\s+vekili)?|başkan(?:\s+vekili)?))\s+(.+)$",
        s,
        re.I,
    )
    if m_ch:
        title = tr_title(m_ch.group(1))
        nm_raw = clean_person_name(m_ch.group(2))
        if nm_raw:
            nm_l = tr_lower(nm_raw)
            nm_canon = _HISTORIC_ALIASES.get(nm_l, nm_raw)
            if len(nm_canon.split()) >= 1 and len(nm_canon) >= 4:
                return nm_canon, "baskan", title

    # Strip "Bay ", "Bayan ", "Dr.", "Doktor ", trailing " Bey"
    cleaned = re.sub(r"^(?:Bay|Bayan|Doktor|Dr\.?|Sayın)\s+", "", s, flags=re.I)
    cleaned = re.sub(r"^(?:Dr|Dt|Av)\.\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+Bey$", "", cleaned, flags=re.I).strip(" .")
    # Fix missing space after initial dot ("A.Mithat" -> "A. Mithat", "Ü.Süleyman" -> "Ü. Süleyman")
    cleaned = re.sub(r"\b([A-ZÇĞİÖŞÜ])\.([A-ZÇĞİÖŞÜa-zçğıöşü]{2,})", r"\1. \2", cleaned)
    cl_l = tr_lower(cleaned)
    if cl_l in _HISTORIC_ALIASES:
        return _HISTORIC_ALIASES[cl_l], "milletvekili_bakan", None

    # Validate looks like a real person name (2-4 Title-Case or ALL-CAPS words, no lowercase prose sentences)
    words = cleaned.split()
    if not (2 <= len(words) <= 4):
        return None
    # Reject lines with weird non-letter OCR junk
    if any(not re.match(r"^[A-ZÇĞİÖŞÜa-zçğıöşüâîûÂÎÛ.\-]+$", w) for w in words):
        return None
    # Every word (except single-letter initial or 've') must start with uppercase (or OCR 'ümit')
    cap_words = sum(1 for w in words if w[0].isupper() or w[0] in "ÇĞİÖŞÜ" or tr_lower(w) == "ümit")
    if cap_words < len(words):
        return None
    # Avoid Turkish common non-name starters
    if tr_lower(words[0]) in ("bir", "bu", "şu", "her", "iki", "üç", "dört", "beş", "türk", "kıbrıs", "işbu", "ancak", "fakat", "ayrıca", "olarak"):
        return None

    return tr_title(cleaned), "milletvekili_bakan", None


def parse_historic_colon_turns(
    doc: dict, lines: list[str], tarih: str | None, birlesim: int | None
) -> tuple[list[dict], list[str]]:
    """Parse 1960-1970 colon-formatted minutes (KC, TCM, KTY Dönem I)."""
    # Special case: Dr. Fazıl Küçük's standalone 16 August 1960 Vice-President inauguration speech
    if "cb muavini konusma metni" in tr_lower(doc["filename"]):
        body = clean_speech_body(lines, is_pdf=True)
        if body:
            return (
                [
                    {
                        "doc_id": doc["local_rel_path"],
                        "meclis": doc["meclis"],
                        "donem": doc["donem"],
                        "yasama_yili": doc["yasama_yili"],
                        "birlesim": birlesim,
                        "oturum": 1,
                        "tarih": tarih or "1960-08-16",
                        "speaker_raw": "Cumhurbaşkanı Muavini Dr. Fazıl Küçük",
                        "speaker_name": "Fazıl Küçük",
                        "speaker_role": "milletvekili_bakan",
                        "speaker_title": "Cumhurbaşkanı Muavini",
                        "constituency": None,
                        "modifiers": None,
                        "text": body,
                        "char_count": len(body),
                        "word_count": len(body.split()),
                        "text_source": doc.get("text_source"),
                        "source_file": doc["local_rel_path"],
                        "source_url": doc.get("download_url"),
                    }
                ],
                [],
            )

    # Pre-merge 2-line split colon headers: "Başkan" + ": Oturumu açıyorum."
    merged_lines: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if i + 1 < len(lines) and cur and len(cur) <= 35 and not cur.endswith((":", ".", ";", "?", "!")):
            nxt = lines[i + 1].strip()
            if nxt.startswith(":"):
                merged_lines.append(cur + nxt)
                i += 2
                continue
        merged_lines.append(lines[i])
        i += 1

    # Detect session chair from "Hazır bulunanlar: İbrahim Orhan, Başkan" / "Meclis Başkanı, Dr. Orhan Müderrisoğlu"
    current_chair: str | None = None
    seen_chairs: list[str] = []
    for line in merged_lines[:45]:
        m_hz = re.search(
            r"Hazır\s+(?:bulunanlar|olanlar)\s*[:]\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü.\s]+?)\s*,\s*(?:Başkan|Reis(?:\s+vekili)?)",
            line,
            re.I,
        )
        if m_hz:
            ch = clean_person_name(m_hz.group(1))
            if ch and len(ch.split()) >= 2:
                current_chair = ch
                if ch not in seen_chairs:
                    seen_chairs.append(ch)
                break

    turns: list[dict] = []
    cur_turn: dict | None = None
    cur_body: list[str] = []

    def flush():
        nonlocal cur_turn, cur_body
        if cur_turn is not None:
            body_text = clean_speech_body(cur_body, is_pdf=True)
            if body_text:
                cur_turn["text"] = body_text
                cur_turn["char_count"] = len(body_text)
                cur_turn["word_count"] = len(body_text.split())
                turns.append(cur_turn)
        cur_turn = None
        cur_body = []

    for line in merged_lines:
        s = line.strip()
        if not s:
            continue
        # Match "<Label> : [-–—^]? <optional speech rest>"
        m = re.match(r"^([A-ZÇĞİÖŞÜü][A-ZÇĞİÖŞÜa-zçğıöşüâîûÂÎÛ.\s]{1,46}?)\s*:\s*[-–—^]?\s*(.*)$", s)
        if m:
            label_raw = m.group(1).strip()
            rest = m.group(2).strip()
            norm = _normalize_historic_name(label_raw, current_chair)
            if norm is not None:
                spk_name, spk_role, spk_title = norm
                if spk_role == "baskan" and spk_name not in ("Meclis Başkanı", "Meclis Reisi", "Reis Vekili"):
                    current_chair = spk_name
                    if spk_name not in seen_chairs:
                        seen_chairs.append(spk_name)
                flush()
                cur_turn = {
                    "doc_id": doc["local_rel_path"],
                    "meclis": doc["meclis"],
                    "donem": doc["donem"],
                    "yasama_yili": doc["yasama_yili"],
                    "birlesim": birlesim,
                    "oturum": 1,
                    "tarih": tarih,
                    "speaker_raw": label_raw,
                    "speaker_name": spk_name,
                    "speaker_role": spk_role,
                    "speaker_title": spk_title,
                    "constituency": None,
                    "modifiers": None,
                    "text_source": doc.get("text_source"),
                    "source_file": doc["local_rel_path"],
                    "source_url": doc.get("download_url"),
                }
                if rest:
                    cur_body.append(rest)
                continue

        if cur_turn is not None:
            cur_body.append(s)

    flush()
    return turns, seen_chairs


def parse_file_worker(path_str: str) -> tuple[dict, list[dict]]:
    doc = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return parse_document(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in EXTRACTED_DIR.rglob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"Parsing {len(files)} extracted documents...")

    t0 = time.time()
    docs_out = PARSED_DIR / "documents.jsonl"
    speeches_out = PARSED_DIR / "speeches.jsonl"

    total_turns = 0
    docs_with_date = 0
    docs_with_turns = 0
    speaker_counts = Counter()

    with docs_out.open("w", encoding="utf-8") as f_doc, speeches_out.open("w", encoding="utf-8") as f_sp:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for idx, (doc_rec, turns) in enumerate(ex.map(parse_file_worker, files, chunksize=16), start=1):
                f_doc.write(json.dumps(doc_rec, ensure_ascii=False) + "\n")
                if doc_rec["tarih"]:
                    docs_with_date += 1
                if turns:
                    docs_with_turns += 1
                for t in turns:
                    f_sp.write(json.dumps(t, ensure_ascii=False) + "\n")
                    speaker_counts[t["speaker_name"]] += 1
                total_turns += len(turns)
                if idx % 200 == 0 or idx == len(files):
                    print(
                        f"  [{idx}/{len(files)}] turns={total_turns} dated={docs_with_date} "
                        f"with_turns={docs_with_turns} elapsed={time.time() - t0:.1f}s",
                        flush=True,
                    )

    print(f"\nDONE in {time.time() - t0:.1f}s")
    print(f"Documents: {len(files)} | With date: {docs_with_date} | With speeches: {docs_with_turns}")
    print(f"Total speeches: {total_turns} | Unique speakers: {len(speaker_counts)}")
    print("Top 15 speakers:")
    for name, cnt in speaker_counts.most_common(15):
        print(f"  {cnt:6d}  {name}")


if __name__ == "__main__":
    main()
