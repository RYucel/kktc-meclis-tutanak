"""Third-pass speaker resolution: anchor-based OCR rescue, multi-person detection and role-title cleanup.

Runs *after* the two-pass canonicalization in ``build_dataset.canonicalize_speakers`` and works on
aggregate (speaker_name, speaker_role, year) groups, so the exact same logic can be applied to:

* in-memory turn lists (``build_dataset.py``, ``sync_incremental.py``) via ``apply_to_records``
* the published ``speeches.parquet`` (``recanonicalize.py``) without the raw archive

Design rules (precision over recall — a wrong merge puts words in a real politician's mouth):

1. **Anchors** are trusted identities: clean person names that are either in the official
   KKTC rosters (``data/official_rosters.json``) or have >= ``min_anchor_turns`` turns.
2. **Anchor consolidation** merges split identities that are provably the same person:
   initial-only differences ("Osman N. Örek" == "Osman Örek"), a dropped first name confirmed by
   strong evidence ("Ersan Saner" -> "Hamza Ersan Saner") and low-frequency OCR twins
   ("Ut. Sait Güven" -> "Sait Güven"). Every merge requires temporal overlap.
3. **Rescue** maps a non-anchor label to an anchor only when, after stripping (possibly glued)
   titles and OCR prefixes, a token window matches exactly one anchor with a clear margin AND
   that anchor was active within +/- 2 years of the turn.
4. **Multi-person labels** ("Kutlu Evren, Ünal Üstel") are never attributed to either person;
   chair/clerk turns fall back to the generic role title.
5. **Protected surnames** (father/son families such as Denktaş) are excluded from every fuzzy step.
6. Anything not resolved is left untouched and simply reported as *unverified*.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # rapidfuzz is ~100x faster; difflib keeps the module dependency-free
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    def _ratio(a: str, b: str) -> float:
        return _rf_fuzz.ratio(a, b)

    def _extract(query: str, choices: list[str], limit: int = 8, cutoff: float = 80.0):
        return [(c, s, i) for c, s, i in _rf_process.extract(query, choices, scorer=_rf_fuzz.ratio, limit=limit, score_cutoff=cutoff)]

except ImportError:  # pragma: no cover - exercised only without rapidfuzz
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0

    def _extract(query: str, choices: list[str], limit: int = 8, cutoff: float = 80.0):
        scored = [(c, _ratio(query, c), i) for i, c in enumerate(choices) if abs(len(c) - len(query)) <= max(3, len(query) // 4)]
        scored = [x for x in scored if x[1] >= cutoff]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]


ROSTERS_PATH = Path(__file__).resolve().parent.parent / "data" / "official_rosters.json"

# KKTC Cumhuriyet Meclisi term years (used to give roster members an activity window)
KKTC_TERM_YEARS = {
    1: (1985, 1990), 2: (1990, 1993), 3: (1993, 1998), 4: (1998, 2003), 5: (2003, 2005),
    6: (2005, 2009), 7: (2009, 2013), 8: (2013, 2018), 9: (2018, 2022), 10: (2022, 2027),
}

PERSON_ROLES = {"milletvekili_bakan", "baskan", "gecici_baskan", "katip"}

GENERIC_BY_ROLE = {
    "baskan": "Meclis Başkanı",
    "gecici_baskan": "Geçici Başkan",
    "katip": "Katip",
}

# Families where first names differ by one letter (Rauf / Raif) — never fuzzy-merge these.
PROTECTED_SURNAMES = {"denktas"}

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)

# Folded title vocabulary: tokens that are never part of a person's name
TITLE_WORDS = {
    "baskan", "baskani", "baskam", "baskanligi", "baskanlik", "vekili", "vekil", "yardimcisi",
    "basbakan", "basbakani", "bakan", "bakani", "bakanligi", "devletbakani", "meclis", "meclisi",
    "heclis", "pleclis", "reisi", "katip", "katibi", "katipler", "katiplik", "katiplar", "gecici",
    "kursusunden", "kursuden", "kursusu", "komitesi", "komite", "sozcusu", "uyesi", "uye", "milletvekili",
    "en", "yasli", "sayin", "sn", "dr", "dt", "dtt", "prof", "doc", "av", "muh", "ut", "bt", "ot", "hr",
    "ikinci", "birinci", "ucuncu", "oturum", "yerinden", "devamla", "ve", "adina", "cumhurbaskani",
    "genel", "sekreteri", "lideri", "musteşari", "mustesari", "raportoru",
    # Ministry / committee subject words that precede the title
    "maliye", "ekonomi", "butce", "plan", "icisleri", "disisleri", "savunma", "tarim", "orman",
    "egitim", "milli", "kultur", "genclik", "spor", "saglik", "calisma", "sosyal", "guvenlik",
    "bayindirlik", "ulastirma", "isleri", "koy", "koyisleri", "hizmetler", "kooperatifler", "eski", "eser", "turizm", "cevre", "ticaret", "sanayi", "enerji", "gida", "kaynaklar",
    "dogal", "hukuk", "idari", "isler", "dilekce", "anayasa", "koyisleri", "iskan", "imar",
    "iktisadi", "siyasi", "baskanyardimcisi",
}

# Title stems that are often glued to the following name by OCR ("Başbakanhakkı Atun",
# "Ekonomi ve Maliyebakanısalihcoşar", "İçişleri Bakanımustafaadaoğlu")
_GLUED_STEMS = sorted(
    [
        "basbakanyardimcisi", "baskanyardimcisi", "devletbakani", "yardimcisi", "basbakani",
        "basbakan", "bakanligi", "bakani", "baskani", "vekili", "katipligi", "kursusunden",
    ],
    key=len,
    reverse=True,
)
_GLUED_RE = re.compile("(" + "|".join(_GLUED_STEMS) + ")")

# Canonical generic role labels (display form). Used to fold OCR variants like
# "Îkînci Başkan", "İkinci Başkam", "Pleclis Başkan Vekili".
ROLE_CANON = {
    "meclis baskani": "Meclis Başkanı",
    "baskan": "Meclis Başkanı",
    "baskan vekili": "Başkan Vekili",
    "meclis baskan vekili": "Başkan Vekili",
    "baskan yardimcisi": "Başkan Yardımcısı",
    "ikinci baskan": "İkinci Başkan",
    "ikinci baskan vekili": "İkinci Başkan Vekili",
    "birinci baskan": "Birinci Başkan",
    "birinci baskan vekili": "Birinci Başkan Vekili",
    "gecici baskan": "Geçici Başkan",
    "gecici baskan vekili": "Geçici Başkan Vekili",
    "meclis reisi": "Meclis Reisi",
    "katip": "Katip",
    "gecici katip": "Geçici Katip",
    "basbakan": "Başbakan",
    "basbakan yardimcisi": "Başbakan Yardımcısı",
}
ROLE_TITLE_NAMES = set(ROLE_CANON.values())

# Tokens that mark floor noise / sentence fragments mis-parsed as speaker labels
NON_PERSON_TOKENS = {
    "bir", "ses", "sesler", "tesbit", "tespit", "edilemeyen", "edilemiyen", "anlasilmayan", "siralardan",
    "siralarindan", "salondan", "izleyici", "milletvekilleri", "bazi", "kadar", "gecelim", "yoksa", "icin",
    "mi", "olmasi", "mevcut", "degildir", "yani", "soyle", "evet", "hayir", "tamam", "efendim", "oturum",
    "birlesim", "gundem", "madde", "kisim", "bolum", "sunuslari", "baskanligin", "kurula", "konusmaya",
    "gorusulen", "gorusme", "gorusmeler", "oylama", "oylar", "kabul", "ret", "cekimser", "yasa", "tasari",
}

CONSTITUENCY_FOLDS = ("lefkosa", "gazimagusa", "magusa", "girne", "guzelyurt", "iskele", "lefke")

_CLEAN_NAME_RE = re.compile(r"^[A-ZÇĞİÖŞÜa-zçğıöşüâîûÂÎÛ.\-\s]+$")
_SPLIT_RE = re.compile(r"\s*(?:,|;|/|&|\bve\b)\s*", re.I)


def fold(s: str) -> str:
    s = (s or "").translate(_TR_FOLD).lower().replace("̇", "")
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def looks_like_person(name: str) -> bool:
    """Light structural check for anchor eligibility (stricter checks live in build_dataset)."""
    if not name or not (5 <= len(name) <= 55) or not _CLEAN_NAME_RE.match(name):
        return False
    words = name.split()
    if not (2 <= len(words) <= 4):
        return False
    if any(not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşüâîûÂÎÛ]{2}", w) and not re.fullmatch(r"[A-ZÇĞİÖŞÜ]\.", w) for w in words):
        return False  # stray punctuation tokens: "Başbakandr . Dervişeroğlu"
    f = fold(name).split()
    if any(_GLUED_RE.search(t) for t in f):
        return False  # title glued to a name: "Başbakanderviş Eroğlu", "Maliyebakanısalihcoşar"
    if len(f) < 2 or len(f[-1]) <= 2:
        return False
    if all(t in TITLE_WORDS for t in f) or any(t in TITLE_WORDS and len(t) > 3 for t in f):
        return False
    if any(t in NON_PERSON_TOKENS for t in f):
        return False
    # Initials are allowed only as middle/leading tokens ("Osman N. Örek", "O. Nejat Konuk")
    if any(len(w.rstrip(".")) < 2 and not w.endswith(".") for w in words):
        return False
    return True


def ocr_norm(s: str) -> str:
    """Undo the two most frequent scan artefacts after folding: 'ıı'/'ii' for 'n', 'tj' for 'ü'."""
    return s.replace("ii", "n").replace("tj", "u")


def _ocr_ratio(a: str, b: str) -> float:
    return max(_ratio(a, b), _ratio(ocr_norm(a), b))


def _is_initial(tok: str) -> bool:
    return len(tok) == 1


@dataclass
class Anchor:
    name: str
    count: int
    years: set[int] = field(default_factory=set)
    roster: bool = False
    year_counts: Counter = field(default_factory=Counter)

    def __post_init__(self):
        self.fk = fold(self.name)
        self.tokens = self.fk.split()
        self.ns = self.fk.replace(" ", "")
        self.full_tokens = [t for t in self.tokens if not _is_initial(t)]
        self.initials = [t for t in self.tokens if _is_initial(t)]

    @property
    def surname(self) -> str:
        return self.tokens[-1] if self.tokens else ""

    def active(self, year: int | None, slack: int = 2) -> bool:
        if year is None or not self.years:
            return year is None
        return any((year + d) in self.years for d in range(-slack, slack + 1))

    def overlaps(self, other: "Anchor", slack: int = 1) -> bool:
        if not self.years or not other.years:
            return False
        return any(other.active(y, slack) for y in self.years)


@dataclass
class Resolution:
    name: str
    method: str


def load_rosters(path: Path | str | None = None) -> dict:
    p = Path(path) if path else ROSTERS_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def roster_years(rosters: dict) -> dict[str, set[int]]:
    """Folded roster name -> active years from official term windows."""
    out: dict[str, set[int]] = defaultdict(set)
    for key, term in (rosters or {}).items():
        donem = term.get("donem")
        if not isinstance(donem, int) or donem not in KKTC_TERM_YEARS:
            m = re.match(r"KKTC:(\d+)", key)
            donem = int(m.group(1)) if m else None
        if donem not in KKTC_TERM_YEARS:
            continue
        y0, y1 = KKTC_TERM_YEARS[donem]
        for mem in term.get("members", []):
            fk = fold(mem.get("name", ""))
            if fk:
                out[fk].update(range(y0, y1 + 1))
    return out


class SpeakerResolver:
    """Resolve noisy speaker labels onto trusted anchor identities.

    ``stats`` maps speaker_name -> {"count": int, "years": Counter(year -> turns)}.
    """

    def __init__(
        self,
        stats: dict[str, dict],
        rosters: dict | None = None,
        min_anchor_turns: int = 100,
        min_year_turns: int = 2,
    ):
        self.stats = stats
        self.report: dict[str, list] = defaultdict(list)
        r_years = roster_years(rosters or {})
        roster_display = {}
        for term in (rosters or {}).values():
            for mem in term.get("members", []):
                if mem.get("name"):
                    roster_display.setdefault(fold(mem["name"]), mem["name"])

        anchors: dict[str, Anchor] = {}
        seen_fk: set[str] = set()
        for nm, st in sorted(stats.items(), key=lambda kv: -kv[1]["count"]):
            if nm in ROLE_TITLE_NAMES or not looks_like_person(nm):
                continue
            fk = fold(nm)
            in_roster = fk in r_years
            # Pre-1983 assemblies have no official roster: a clean name recurring across many
            # documents is trusted even below the turn threshold (e.g. "Ümit Süleyman").
            recurring = st["count"] >= 40 and st.get("docs", 0) >= 5
            if not in_roster and st["count"] < min_anchor_turns and not recurring:
                continue
            if fk in seen_fk:  # spelling twin of a bigger anchor (handled by fold elsewhere)
                continue
            seen_fk.add(fk)
            years = {int(y) for y, c in st.get("years", {}).items() if c >= min_year_turns}
            years |= r_years.get(fk, set())
            anchors[nm] = Anchor(nm, st["count"], years, in_roster, Counter({int(y): c for y, c in st.get("years", {}).items()}))

        # Roster members never seen verbatim in the data still act as anchors (e.g. OCR-only spellings)
        for fk, disp in roster_display.items():
            if fk not in seen_fk:
                anchors[disp] = Anchor(disp, 0, set(r_years[fk]), True)
                seen_fk.add(fk)

        self._add_core_anchors(stats, anchors, min_anchor_turns)

        # Vocabulary of trusted name tokens: a "wrecked" token equal to one of these is a real name
        self._vocab: set[str] = set()
        for fk in r_years:
            self._vocab.update(t for t in fk.split() if len(t) >= 3)
        for a in anchors.values():
            if a.count >= 1000:
                self._vocab.update(t for t in a.full_tokens if len(t) >= 3)

        self.anchor_map: dict[str, str] = {}  # demoted anchor name -> target anchor name
        self._consolidate(anchors)
        self.anchors = {n: a for n, a in anchors.items() if n not in self.anchor_map}
        self._by_ns: dict[str, Anchor] = {}
        for a in sorted(self.anchors.values(), key=lambda a: -a.count):
            self._by_ns.setdefault(a.ns, a)
        self._by_surname: dict[str, list[Anchor]] = defaultdict(list)
        self._by_first: dict[str, list[Anchor]] = defaultdict(list)
        for a in self.anchors.values():
            self._by_surname[a.surname].append(a)
            if a.full_tokens:
                self._by_first[a.full_tokens[0]].append(a)
        self._surname_keys = [k for k in self._by_surname if len(k) >= 5]
        self._ns_keys = list(self._by_ns.keys())
        self._top2_cache: dict[str, tuple] = {}
        self._cache: dict[tuple, Resolution | None] = {}

    # ------------------------------------------------------------------ core anchors
    @classmethod
    def core_tokens(cls, name: str) -> list[str] | None:
        """Name tokens left after stripping titles, constituency, OCR prefixes: the identity 'core'."""
        toks = cls._candidate_tokens(fold(name))
        segs = [seg.split() for seg in " ".join(toks).split("|")]
        segs = [[t for t in seg if not any(_ratio(t, c) >= 80 for c in CONSTITUENCY_FOLDS)] for seg in segs]
        segs = [seg for seg in segs if seg]
        if len(segs) != 1:
            return None
        seg = segs[0]
        while seg and len(seg[0]) <= 2 and len(seg) > 2:  # leading OCR junk ("ut", "bt", "o")
            seg = seg[1:]
        full = [t for t in seg if len(t) > 1]
        if not (2 <= len(full) <= 3) or len(seg) > 4 or any(len(t) == 2 for t in seg):
            return None
        if any(t in NON_PERSON_TOKENS or re.fullmatch(r"[ivx]+", t) for t in seg):
            return None
        return seg

    def _add_core_anchors(self, stats: dict, anchors: dict[str, "Anchor"], min_turns: int) -> None:
        """Identities that only ever appear wrapped in titles ("İkinci Başkan Vekili Ümit Süleyman")."""
        agg: dict[str, dict] = {}
        anchor_ns = {a.ns for a in anchors.values()}
        for nm, st in stats.items():
            if nm in anchors or nm in ROLE_TITLE_NAMES:
                continue
            core = self.core_tokens(nm)
            if not core:
                continue
            key = " ".join(core)
            g = agg.setdefault(key, {"count": 0, "docs": 0, "years": Counter(), "variants": Counter()})
            g["count"] += st["count"]
            g["docs"] += st.get("docs", 0)
            g["years"].update({int(y): c for y, c in st.get("years", {}).items()})
            g["variants"][nm] += st["count"]
        for key, g in agg.items():
            if not (g["count"] >= min_turns or (g["count"] >= 40 and g["docs"] >= 5)):
                continue
            ns = key.replace(" ", "")
            if ns in anchor_ns or any(_ratio(ns, a_ns) >= 88 for a_ns in anchor_ns if abs(len(a_ns) - len(ns)) <= 3):
                continue  # already represented (the rescue step will map it)
            display = self._display_from_variants(key.split(), g["variants"])
            if not display or display in anchors:
                continue
            years = {y for y, c in g["years"].items() if c >= 2}
            a = Anchor(display, g["count"], years, False, Counter(g["years"]))
            anchors[display] = a
            anchor_ns.add(a.ns)
            self.report["core_anchors"].append([display, g["count"], g["docs"]])

    @staticmethod
    def _display_from_variants(core: list[str], variants: Counter) -> str | None:
        best, best_key = None, None
        for v, c in variants.items():
            words = [w.strip(".,;:()'’\"{}[]") for w in v.split()]
            picked = [w for w in words if fold(w) in core]
            if [fold(w) for w in picked] != core:
                continue
            cand = " ".join(w[:1].upper() + w[1:] if w[:1].islower() else w for w in picked)
            if not looks_like_person(cand) or any(ch in cand for ch in "îâûÎÂÛ"):
                continue
            score = (c, sum(ch in "çğıöşüÇĞİÖŞÜ" for ch in cand))
            if best_key is None or score > best_key:
                best, best_key = cand, score
        return best

    # ------------------------------------------------------------------ consolidation
    def _consolidate(self, anchors: dict[str, Anchor]) -> None:
        ordered = sorted(anchors.values(), key=lambda a: (-a.count, a.name))
        for weak in sorted(anchors.values(), key=lambda a: (a.count, a.name)):
            if weak.name in self.anchor_map or weak.surname in PROTECTED_SURNAMES:
                continue
            best = None
            for strong in ordered:
                if strong is weak or strong.name in self.anchor_map or strong.surname in PROTECTED_SURNAMES:
                    continue
                if not weak.overlaps(strong):
                    continue
                reason = self._same_person(weak, strong, self._vocab)
                if reason:
                    best = (strong, reason)
                    break
            if not best:
                continue
            strong, reason = best
            # Initial-only variants resolve to the fuller spelling-without-initials form
            if reason == "initial_variant" and len(weak.full_tokens) == len(weak.tokens) and len(strong.full_tokens) < len(strong.tokens):
                weak, strong = strong, weak
                if weak.name in self.anchor_map or strong.name in self.anchor_map:
                    continue
            self.anchor_map[weak.name] = strong.name
            strong.years |= weak.years
            strong.year_counts.update(weak.year_counts)
            strong.count += weak.count
            self.report["anchor_merges"].append([weak.name, strong.name, reason])
        # Flatten chains (A -> B -> C)
        for k in list(self.anchor_map):
            seen = {k}
            t = self.anchor_map[k]
            while t in self.anchor_map and t not in seen:
                seen.add(t)
                t = self.anchor_map[t]
            self.anchor_map[k] = t

    @staticmethod
    def _same_person(weak: Anchor, strong: Anchor, vocab: frozenset | set = frozenset()) -> str | None:
        wt, st = weak.tokens, strong.tokens
        # (a) identical once single-letter initials are removed: "Osman N. Örek" == "Osman Örek".
        #     Conflicting initials ("Mehmet A. Yılmaz" vs "Mehmet B. Yılmaz") are different people.
        if weak.full_tokens == strong.full_tokens and len(weak.full_tokens) >= 2 and wt != st:
            if weak.initials and strong.initials and weak.initials != strong.initials:
                return None
            return "initial_variant"
        # (b) dropped first name, confirmed by overwhelming evidence: "Ersan Saner" -> "Hamza Ersan Saner"
        if (
            len(wt) == 2
            and len(st) == 3
            and wt == st[1:]
            and not weak.roster
            and strong.count >= 5 * max(weak.count, 1)
            and (strong.roster or strong.count >= 1000)
        ):
            return "dropped_first_name"
        # (c) low-frequency OCR twin of a much stronger anchor: "Ut. Sait Güven" -> "Sait Güven"
        if weak.roster and strong.roster:
            return None
        if strong.count >= 4 * max(weak.count, 1):
            w_core = " ".join(weak.full_tokens)
            s_core = " ".join(strong.full_tokens)
            w_ns, s_ns = w_core.replace(" ", ""), s_core.replace(" ", "")
            if (
                len(w_core) >= 8
                and w_core.split()[0][:1] == s_core.split()[0][:1]
                and abs(len(w_ns) - len(s_ns)) <= 2  # OCR keeps length; "Bayram" vs "Bayraktar" does not
                and _ratio(w_ns, s_ns) >= 90
            ):
                return "ocr_twin"
            # leading OCR junk token: "Bt. Sait Güven", "Hr. Erbay Kanatlı"
            if len(wt) == len(st) + 1 and wt[1:] == st and len(wt[0]) <= 3:
                return "ocr_prefix"
        # (d) whole-label OCR twin of a 20x stronger anchor, initials included ("Osmc.n Örek" ~ "Osman Örek")
        if (
            not weak.roster
            and strong.count >= 20 * max(weak.count, 1)
            and weak.ns[:1] == strong.ns[:1]
            and abs(len(weak.ns) - len(strong.ns)) <= 2
            and len(weak.ns) >= 8
            and _ocr_ratio(weak.ns, strong.ns) >= 88
            and not (set(weak.full_tokens) - set(strong.full_tokens)) & vocab
        ):
            return "ocr_twin"
        # (e) one token exact, the other an OCR wreck of a 20x stronger anchor ("Hakkı Atuıı", "Ergtjn Vehbi")
        if strong.count >= 20 * max(weak.count, 1) and len(weak.full_tokens) == len(strong.full_tokens) == 2:
            (wf, wl), (sf, sl) = weak.full_tokens, strong.full_tokens
            for same, w_tok, s_tok in ((wf == sf, wl, sl), (wl == sl, wf, sf)):
                if (
                    same and w_tok != s_tok and w_tok[:1] == s_tok[:1]
                    and abs(len(w_tok) - len(s_tok)) <= 2 and _ocr_ratio(w_tok, s_tok) >= 70
                    and w_tok not in vocab
                ):
                    return "ocr_token"
        return None

    # ------------------------------------------------------------------ helpers
    def is_anchor(self, name: str) -> bool:
        return name in self.anchors

    def verified_names(self) -> set[str]:
        return set(self.anchors) | ROLE_TITLE_NAMES

    @staticmethod
    def _role_title(fk: str) -> str | None:
        toks = fk.split()
        if not toks or len(toks) > 4:
            return None
        best, best_s = None, 0.0
        for key, disp in ROLE_CANON.items():
            if abs(len(key.split()) - len(toks)) > 0:
                continue
            if "basbakan" in key:  # "Bakan"/"Başkan" OCR twins must never become the Prime Minister
                if fk == key:
                    return disp
                continue
            # a different, real title word ("bakan", "bakani") is not an OCR variant of "baskan"
            if len(toks) == 1 and fk in TITLE_WORDS and fk not in ("baskan", "baskani", "baskam", "katip", "katibi"):
                continue
            s = _ratio(fk, key)
            if s > best_s:
                best, best_s = disp, s
        if best and best_s >= 86:
            return best
        return None

    @staticmethod
    def _candidate_tokens(fk: str) -> list[str]:
        out: list[str] = []
        for tok in fk.split():
            m = None
            for m in _GLUED_RE.finditer(tok):
                pass
            if m and m.end() < len(tok):
                rest = tok[m.end():]
                rest = re.sub(r"^(?:dr|dt|av)(?=[a-z]{4})", "", rest)
                if len(rest) >= 4:
                    out.append(rest)
                continue
            if tok in TITLE_WORDS:
                out.append("|")  # boundary marker: windows never span a removed title
                continue
            out.append(tok)
        return out

    @staticmethod
    def _windows(tokens: list[str]) -> list[tuple[int, int, str]]:
        wins = []
        segs, cur, start = [], [], 0
        for i, t in enumerate(tokens + ["|"]):
            if t == "|":
                if cur:
                    segs.append((start, cur))
                cur, start = [], i + 1
            else:
                cur.append(t)
        for s0, seg in segs:
            for i in range(len(seg)):
                for j in range(i + 1, min(len(seg), i + 4) + 1):
                    w = "".join(seg[i:j])
                    if len(w) >= 7:
                        wins.append((s0 + i, s0 + j, w))
        return wins

    def _top2(self, ns: str) -> tuple[Anchor | None, float, Anchor | None, float]:
        hit = self._top2_cache.get(ns)
        if hit is not None:
            return hit
        exact = self._by_ns.get(ns) or self._by_ns.get(ocr_norm(ns))
        if exact:
            out = (exact, 100.0 if exact.ns == ns else 97.0, None, 0.0)
        else:
            L = len(ns)
            ranked = [
                (self._by_ns[c], sc)
                for c, sc, _ in _extract(ns, self._ns_keys, limit=8, cutoff=80.0)
                if abs(len(c) - L) <= max(3, L // 4)
            ]
            best, bs = ranked[0] if ranked else (None, 0.0)
            second, ss = ranked[1] if len(ranked) > 1 else (None, 0.0)
            out = (best, bs, second, ss)
        self._top2_cache[ns] = out
        return out

    # ------------------------------------------------------------------ main API
    def resolve(self, name: str, role: str | None = None, year: int | None = None) -> Resolution | None:
        key = (name, role, year)
        if key in self._cache:
            return self._cache[key]
        res = self._resolve(name, role, year)
        self._cache[key] = res
        return res

    def _resolve(self, name: str, role: str | None, year: int | None) -> Resolution | None:
        if not name or (role is not None and role not in PERSON_ROLES):
            return None
        if name in self.anchor_map:
            return Resolution(self.anchor_map[name], "anchor_merge")
        if name in self.anchors or name in ROLE_TITLE_NAMES:
            return None
        fk = fold(name)
        if not fk:
            return None

        # 1. Generic role labels with OCR noise -> one canonical role title
        rt = self._role_title(fk)
        if rt:
            return Resolution(rt, "role_title") if rt != name else None

        if any(t in PROTECTED_SURNAMES for t in fk.split()):
            ns = fk.replace(" ", "")
            a = self._by_ns.get(ns)
            return Resolution(a.name, "spaceless_exact") if a and a.name != name else None

        tokens = self._candidate_tokens(fk)
        wins = self._windows(tokens)

        # 2. Multi-person labels ("Kutlu Evren, Ünal Üstel", "Mustafa Akıncı Hüseyin Özgürgün")
        strong_hits: list[tuple[int, int, Anchor]] = []
        for i, j, w in wins:
            a, s, _, _ = self._top2(w)
            if a and s >= 92 and j - i >= 2:
                strong_hits.append((i, j, a))
        distinct = {}
        for i, j, a in sorted(strong_hits, key=lambda h: -(h[1] - h[0])):
            if a.name in distinct:
                continue
            if all(j <= i2 or i >= j2 for (i2, j2) in distinct.values()):
                distinct[a.name] = (i, j)
        parts = [p for p in _SPLIT_RE.split(name) if p.strip(" .,;:-")]
        person_parts = [p for p in parts if self._part_is_person(p)]
        if len(distinct) >= 2 or (len(parts) >= 2 and len(person_parts) == len(parts)):
            self.report["multi_person"].append([name, role, year, sorted(distinct)])
            generic = GENERIC_BY_ROLE.get(role or "")
            return Resolution(generic, "multi_person") if generic else None

        # 3. Best single-anchor rescue with margin + temporal guard
        best: tuple[float, int, Anchor] | None = None
        for i, j, w in wins:
            a, s, a2, s2 = self._top2(w)
            if not a or s < 88:
                continue
            if self._leftover_conflict(tokens, i, j, a):
                continue
            # a single token may only stand for a whole multi-token name when it is that name glued
            # together ("Dervişeroğlu"); a bare surname goes through the stricter surname rule below
            if j - i == 1 and (len(a.full_tokens) < 2 or len(w) < 0.85 * len("".join(a.full_tokens))):
                continue
            if a2 is not None and s2 > 0 and a2.name != a.name and s - s2 < 6 and s < 96:
                continue
            if a.surname in PROTECTED_SURNAMES:
                continue
            if year is None:
                if s < 94:
                    continue
            elif not a.active(year):
                continue
            cand = (s, j - i, a)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best:
            s, _, a = best
            return Resolution(a.name, "anchor_exact" if s >= 100 else "anchor_fuzzy") if a.name != name else None

        # 3b. Token-aligned rescue: one name token exact, the other a same-initial OCR wreck,
        #     and exactly one anchor active that year fits ("Hakkı Attın" -> "Hakkı Atun").
        if year is not None:
            tt = self._token_temporal(tokens, year)
            if tt is not None:
                return Resolution(tt.name, "token_temporal") if tt.name != name else None

        # 4. Bare surname resolved by unique active anchor (cross-document, temporally scoped)
        real = [t for t in tokens if t != "|"]
        if len(real) == 1 and len(real[0]) >= 5 and year is not None:
            # A token that is also the first name of someone active that year ("Tahsin", "Murat") is ambiguous
            if any(a.year_counts.get(year, 0) > 0 or a.active(year, slack=0) for a in self._by_first.get(real[0], [])):
                return None
            cands = [
                a for a in self._by_surname.get(real[0], [])
                if a.year_counts.get(year, 0) >= 20 or (a.roster and year in a.years and a.year_counts.get(year, 0) >= 5)
            ]
            if len(cands) == 1 and real[0] not in PROTECTED_SURNAMES:
                return Resolution(cands[0].name, "surname_temporal")
        return None

    def _token_temporal(self, tokens: list[str], year: int) -> Anchor | None:
        segs = " ".join(tokens).split("|")
        seg = max((x.split() for x in segs), key=len, default=[])
        core = [t for t in seg if len(t) >= 3]
        if len(core) != 2 or len(seg) - len(core) > 1:
            return None
        f, l = core
        matches = []
        for a in self.anchors.values():
            if len(a.full_tokens) != 2 or a.surname in PROTECTED_SURNAMES:
                continue
            if not (a.year_counts.get(year, 0) > 0 or a.active(year, slack=1)):
                continue
            af, al = a.full_tokens
            if f == af and l != al and l[:1] == al[:1] and _ocr_ratio(l, al) >= 65 and abs(len(l) - len(al)) <= 2:
                matches.append(a)
            elif l == al and f != af and f[:1] == af[:1] and _ocr_ratio(f, af) >= 65 and abs(len(f) - len(af)) <= 2:
                matches.append(a)
        if len(matches) != 1:
            return None
        # the wrecked token must not itself be a real token of another active anchor
        a = matches[0]
        other = l if f == a.full_tokens[0] else f
        if other in self._by_surname or other in self._by_first or other in self._vocab:
            return None
        return a

    def _leftover_conflict(self, tokens: list[str], i: int, j: int, a: Anchor) -> bool:
        """A window that leaves another known surname next to it is a different/compound name.

        "Ahmet Yusuf Atamsoy" must not become "Ahmet Yusuf" just because that two-token window
        matches an anchor exactly: "atamsoy" is itself a known surname.
        """
        # every other token of the same segment (not only the adjacent ones: "Vasfi Candan Kan Murat")
        lo = i
        while lo > 0 and tokens[lo - 1] != "|":
            lo -= 1
        hi = j
        while hi < len(tokens) and tokens[hi] != "|":
            hi += 1
        for k in list(range(lo, i)) + list(range(j, hi)):
            if 0 <= k < len(tokens):
                t = tokens[k]
                if t == "|" or len(t) < 5 or t in a.tokens:
                    continue
                if any(_ratio(t, c) >= 70 for c in CONSTITUENCY_FOLDS):
                    continue
                if t in self._by_surname or any(_ratio(t, sn) >= 88 for sn in self._surname_keys if abs(len(sn) - len(t)) <= 2):
                    return True
        return False

    @staticmethod
    def _part_is_person(part: str) -> bool:
        """A comma/"ve"-separated chunk that is plainly a 2–3 word personal name, not a title fragment."""
        raw = re.sub(r"^(?:(?:Prof|Doç|Dr|Dt|Av|Sayın|Sn)[.,]*\s*)+", "", part.strip(" .,;:-"), flags=re.I)
        words = raw.split()
        if not (2 <= len(words) <= 3):
            return False
        for w in words:
            if not re.fullmatch(r"[A-ZÇĞİÖŞÜÂÎÛ][a-zçğıöşüâîû]{2,}", w):
                return False
            fw = fold(w)
            if fw in TITLE_WORDS or (len(fw) >= 6 and any(_ratio(fw, t) >= 85 for t in TITLE_WORDS if len(t) >= 6)):
                return False
        return True


# ---------------------------------------------------------------------- record helpers
def _year(tarih) -> int | None:
    if not tarih:
        return None
    m = re.match(r"(\d{4})", str(tarih))
    return int(m.group(1)) if m else None


def stats_from_records(records: Iterable[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for r in records:
        nm = r.get("speaker_name")
        if not nm:
            continue
        st = stats.setdefault(nm, {"count": 0, "years": Counter(), "_docs": set()})
        st["count"] += 1
        if r.get("doc_id"):
            st["_docs"].add(r["doc_id"])
        y = _year(r.get("tarih"))
        if y:
            st["years"][y] += 1
    for st in stats.values():
        st["docs"] = len(st.pop("_docs"))
    return stats


def apply_to_records(records: list[dict], resolver: SpeakerResolver) -> Counter:
    """Rewrite speaker_name/canon_method in place. Returns method counts."""
    methods: Counter = Counter()
    for r in records:
        res = resolver.resolve(r.get("speaker_name"), r.get("speaker_role"), _year(r.get("tarih")))
        if res and res.name != r.get("speaker_name"):
            if not r.get("speaker_name_parsed"):
                r["speaker_name_parsed"] = r.get("speaker_name")
            r["speaker_name"] = res.name
            r["canon_method"] = res.method
            methods[res.method] += 1
    return methods


def resolve_records(records: list[dict], rosters: dict | None = None, base_stats: dict | None = None) -> dict:
    """Convenience wrapper: build a resolver (optionally seeded with existing corpus stats) and apply it."""
    stats = stats_from_records(records)
    if base_stats:
        merged = {k: {"count": v["count"], "years": Counter(v["years"]), "docs": v.get("docs", 0)} for k, v in base_stats.items()}
        for k, v in stats.items():
            m = merged.setdefault(k, {"count": 0, "years": Counter(), "docs": 0})
            m["count"] += v["count"]
            m["years"].update(v["years"])
            m["docs"] += v.get("docs", 0)
        stats = merged
    resolver = SpeakerResolver(stats, rosters)
    methods = apply_to_records(records, resolver)
    return {
        "method_counts": dict(methods),
        "anchors": len(resolver.anchors),
        "anchor_merges": resolver.report.get("anchor_merges", []),
        "multi_person": resolver.report.get("multi_person", [])[:500],
    }


def alias_folds(canonical: str, variants: Iterable[str], limit: int = 8) -> list[str]:
    """Folded search aliases for a canonical speaker (used by the static Space autocomplete)."""
    base = fold(canonical)
    out: list[str] = []
    for v in variants or []:
        fv = fold(v)
        if not fv or fv == base or fv in out:
            continue
        toks = [t for t in fv.split() if t not in TITLE_WORDS]
        if not (1 <= len(toks) <= 4):
            continue
        fv = " ".join(toks)
        if fv != base and fv not in out:
            out.append(fv)
        if len(out) >= limit:
            break
    return out
