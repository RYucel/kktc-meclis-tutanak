"""Build clean MP biographies (space_static/mp_bios.json) from cm.gov.ct.tr sources.

Combines:
1. Former MPs from cm_gecmis_milletvekilleri.json (317 records from cm.gov.ct.tr/GecmisMiletvekilleri),
   re-downloading and cleanly decoding any .doc/.docx entries that had binary/PNG extraction corruption.
2. Current 10th-term MPs from https://cm.gov.ct.tr/Milletvekillerimiz1.aspx (Word .docx/.doc bios).

Maps each biography to the canonical speaker_name in space_static/speakers_index.json so the
Static Space Speaker Dossier can display it in O(1) time.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import ssl
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_text import clean_word_text, detect_magic, extract_docx, extract_pdf  # noqa: E402
from ole_doc_fallback import extract_ole_doc_text  # noqa: E402
from speaker_resolution import fold  # noqa: E402

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

FORMER_JSON = ROOT / "cm_gecmis_milletvekilleri.json"
SPEAKERS_INDEX = ROOT / "space_static" / "speakers_index.json"
OUT_PATH = ROOT / "space_static" / "mp_bios.json"
CURRENT_URL = "https://cm.gov.ct.tr/Milletvekillerimiz1.aspx"
BASE_URL = "https://cm.gov.ct.tr/"

_ACADEMIC_RE = re.compile(
    r"\b(?:Prof\.?\s*Dr\.?|Doç\.?\s*Dr\.?|Yrd\.?\s*Doç\.?\s*Dr\.?|Dr\.?|Dt\.?|Av\.?|Müh\.?|Uzm\.?|Sn\.?|Sayın)\s*",
    re.I,
)
_OLE_TRAILER_RE = re.compile(
    r"\n(?:Normal|Heading \d|Default Paragraph Font|Table Normal|No List|Plain Text|Root Entry|1Table|WordDocument|SummaryInformation|Microsoft (?:Office )?Word)\b.*",
    re.S,
)
_FIELD_RE = re.compile(r"\x13[^\x14\x15]*(?:\x14([^\x15]*))?\x15")
_INCLUDEPIC_RE = re.compile(r"INCLUDEPICTURE\s+\"[^\"]*\"(?:\s*\\[*a-z]+\s*\S*)*\s*(?:MERGEFORMATINET)?\s*", re.I)


def fetch_bytes(url: str, timeout: int = 25) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, parsed.query, parsed.fragment))
    req = urllib.request.Request(safe_url, headers={"User-Agent": "Mozilla/5.0 (KKTC-Meclis-Archive/1.0)"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as r:
        return r.read()


def extract_doc_bytes(data: bytes, url: str) -> str:
    """Decode .docx, .doc (OLE2), .pdf, or .rtf bytes into clean text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(urllib.parse.urlsplit(url).path).suffix or ".doc") as tf:
        tf.write(data)
        tmp_path = Path(tf.name)
    try:
        magic = detect_magic(tmp_path)
        if magic == "zip_docx":
            return extract_docx(tmp_path)["text"]
        if magic == "pdf":
            return extract_pdf(tmp_path)["text"]
        if magic == "ole_doc":
            return clean_word_text(extract_ole_doc_text(tmp_path))
        if magic == "rtf":
            raw = data.decode("cp1254", errors="replace")
            txt = re.sub(r"\\[a-z]+\d*\s?", " ", raw)
            txt = re.sub(r"[{}]", "", txt)
            return clean_word_text(txt)
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


def clean_bio_text(raw: str) -> str:
    """Strip OLE2 trailers, Word field codes, and stray binary artifacts from a biography."""
    s = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    # Unwrap Word field codes (\x13 ... \x14 display \x15)
    s = _FIELD_RE.sub(lambda m: m.group(1) or "", s)
    s = _INCLUDEPIC_RE.sub("", s)
    s = re.sub(r"MERGEFORMATINET|HYPERLINK\s+\"[^\"]*\"", "", s, flags=re.I)
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", "", s)
    # Strip OLE2 trailer if present
    s = _OLE_TRAILER_RE.sub("", s)
    # Clean lines
    lines = []
    for ln in s.split("\n"):
        t = re.sub(r"\s+", " ", ln).strip()
        if not t:
            continue
        # Drop pure punctuation / image placeholder lines
        if re.fullmatch(r"[-–—•·*._/\s]+", t):
            continue
        # Join broken mid-word wraps from old .doc extraction (e.g. "seçilmişt\nir.")
        if lines and len(lines[-1]) > 25 and not re.search(r"[.!?:;…\"”')]$", lines[-1]) and t[0].islower():
            if len(t.split()[0]) <= 4 and not t.startswith(("ve ", "ile ", "da ", "de ", "ki ", "bir ", "bu ")):
                lines[-1] = lines[-1] + t
            else:
                lines[-1] = lines[-1] + " " + t
        else:
            lines.append(t)
    # Drop redundant leading all-caps name / party / constituency header lines (keep actual bio sentences)
    while len(lines) > 1 and len(lines[0]) < 65 and not re.search(r"\d{4}|doğdu|mezun|yılında", lines[0], re.I):
        if lines[0].isupper() or _ACADEMIC_RE.match(lines[0]):
            lines.pop(0)
        else:
            break
    out = "\n\n".join(lines).strip()
    # Collapse excessive dots like "......"
    out = re.sub(r"\.{3,}", ".", out)
    out = re.sub(r"\.\s*\.", ".", out)
    return out


def is_corrupted(raw: str) -> bool:
    t = (raw or "").strip()
    if len(t) < 40:
        return True
    if "IDAT" in t or "bjbj" in t:
        return True
    head = _OLE_TRAILER_RE.sub("", t).strip()
    return len(head) < 40


def parse_member_name(given: str, surname: str) -> str:
    g = re.sub(r"\s+", " ", (given or "").replace("\xa0", " ")).strip()
    s = re.sub(r"\s+", " ", (surname or "").replace("\xa0", " ")).strip()
    if g and s:
        raw = f"{g} {s}"
    else:
        # Sometimes surname holds "ADİLOĞLU DR. Naim" or "BARÇIN Ahmet"
        parts = s.split()
        if len(parts) >= 2 and parts[0].isupper() and not parts[-1].isupper():
            raw = " ".join(parts[1:] + [parts[0]])
        else:
            raw = s or g
    raw = _ACADEMIC_RE.sub("", raw)
    return re.sub(r"\s+", " ", raw).strip(" .,;-")


def fetch_current_mps() -> list[dict]:
    """Fetch current 10th-term MPs and their Word bio URLs from cm.gov.ct.tr/Milletvekillerimiz1.aspx."""
    html = fetch_bytes(CURRENT_URL).decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id=re.compile("GridView1"))
    if not tbl:
        return []
    out = []
    for tr in tbl.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        name = re.sub(r"\s+", " ", tds[0].get_text(" ", strip=True)).strip()
        region = tds[1].get_text(" ", strip=True)
        party = tds[2].get_text(" ", strip=True)
        inp = tds[3].find("input", attrs={"navigateurl": True})
        doc_url = urllib.parse.urljoin(BASE_URL, inp["navigateurl"]) if inp else ""
        if name and doc_url:
            out.append({"name": name, "region": region, "party": party, "doc_url": doc_url})
    return out


def build_speaker_matcher(speakers_index_path: Path):
    data = json.loads(speakers_index_path.read_text(encoding="utf-8"))
    speakers = data.get("speakers", [])
    by_fold: dict[str, str] = {}
    entries: list[tuple[str, list[str], int]] = []
    for sp in speakers:
        nm = sp["n"]
        if sp.get("r"):
            continue
        # Only match against verified/plausible speakers or those with >=3 speeches
        if not (sp.get("v") or sp.get("p") or sp.get("s", 0) >= 3):
            continue
        fk = fold(nm)
        if fk and fk not in by_fold:
            by_fold[fk] = nm
        toks = fk.split()
        if len(toks) >= 2:
            entries.append((nm, toks, int(sp.get("s", 0))))
        # Also index explicit aliases from f field ("base | alias1 | alias2")
        for alt in str(sp.get("f", "")).split(" | ")[1:]:
            alt = alt.strip()
            if alt and alt not in by_fold:
                by_fold[alt] = nm

    def match(raw_name: str) -> str | None:
        fk = fold(raw_name)
        if not fk:
            return None
        if fk in by_fold:
            return by_fold[fk]
        toks = fk.split()
        if len(toks) < 2:
            return None
        # Try reversed order if "SURNAME Given" slipped through
        rev = " ".join([toks[-1]] + toks[:-1])
        if rev in by_fold:
            return by_fold[rev]
        # Token subset / first+last match (prefer verified/highest speech count)
        cands = []
        for nm, st, sc in entries:
            if all(t in st for t in toks) or all(t in toks for t in st) or (toks[0] == st[0] and toks[-1] == st[-1]):
                cands.append((sc, nm))
        if cands:
            cands.sort(reverse=True)
            return cands[0][1]
        return None

    return match


def main():
    match_speaker = build_speaker_matcher(SPEAKERS_INDEX)

    # 1. Load former MPs from cm_gecmis_milletvekilleri.json
    former_raw = json.loads(FORMER_JSON.read_text(encoding="utf-8"))["members"]
    print(f"Loaded {len(former_raw)} former MPs from {FORMER_JSON.name}")

    # Re-download corrupted .doc entries in parallel
    to_refetch = [(i, m) for i, m in enumerate(former_raw) if is_corrupted(m.get("text", "")) and m.get("doc_url")]
    print(f"Re-downloading {len(to_refetch)} corrupted .doc bios from cm.gov.ct.tr...")

    def _refetch(item):
        idx, m = item
        try:
            raw_bytes = fetch_bytes(m["doc_url"])
            return idx, extract_doc_bytes(raw_bytes, m["doc_url"]), None
        except Exception as e:
            return idx, "", str(e)

    refetched_ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for idx, txt, err in ex.map(_refetch, to_refetch):
            if txt and len(txt.strip()) >= 40:
                former_raw[idx]["text"] = txt
                refetched_ok += 1
            else:
                print(f"  [WARN] Could not decode {former_raw[idx]['doc_url']}: {err or 'empty'}")
    print(f"Recovered {refetched_ok}/{len(to_refetch)} corrupted former MP bios!")

    # 2. Fetch current 10th-term MPs from Milletvekillerimiz1.aspx
    print(f"Fetching current MPs from {CURRENT_URL}...")
    current_mps = fetch_current_mps()
    print(f"Found {len(current_mps)} current MPs; downloading their Word bios...")

    def _fetch_cur(m):
        try:
            raw_bytes = fetch_bytes(m["doc_url"])
            return m, extract_doc_bytes(raw_bytes, m["doc_url"]), None
        except Exception as e:
            return m, "", str(e)

    current_ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for m, txt, err in ex.map(_fetch_cur, current_mps):
            if txt and len(txt.strip()) >= 40:
                m["text"] = txt
                current_ok += 1
            else:
                m["text"] = ""
                print(f"  [WARN] Current MP {m['name']} ({m['doc_url']}): {err or 'empty'}")
    print(f"Decoded {current_ok}/{len(current_mps)} current MP bios!")

    # 3. Clean & map to canonical speaker_name (current MPs override former MPs if newer/longer)
    bios: dict[str, dict] = {}
    unmatched: list[str] = []

    def _add(raw_name: str, raw_text: str, doc_url: str, is_current: bool):
        bio = clean_bio_text(raw_text)
        if len(bio) < 40 or "IDAT" in bio:
            return
        canon = match_speaker(raw_name)
        if not canon:
            unmatched.append(raw_name)
            return
        prev = bios.get(canon)
        # Prefer current MP bio (more up-to-date) or longer bio if duplicate
        if prev is None or (is_current and not prev.get("_cur")) or len(bio) > len(prev["bio"]) + 80:
            bios[canon] = {"bio": bio, "url": doc_url, "_cur": is_current}

    for m in former_raw:
        nm = parse_member_name(m.get("given", ""), m.get("surname", ""))
        _add(nm, m.get("text", ""), m.get("doc_url", ""), is_current=False)

    for m in current_mps:
        nm = _ACADEMIC_RE.sub("", m["name"]).strip()
        _add(nm, m.get("text", ""), m.get("doc_url", ""), is_current=True)

    # Strip internal _cur flag and sort cleanly
    clean_bios = {
        k: {"bio": v["bio"], "url": v["url"]}
        for k, v in sorted(bios.items(), key=lambda kv: kv[0])
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sources": [
            "https://cm.gov.ct.tr/GecmisMiletvekilleri",
            "https://cm.gov.ct.tr/Milletvekillerimiz1.aspx",
        ],
        "count": len(clean_bios),
        "bios": clean_bios,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB, "
        f"{len(clean_bios)} matched MPs, {len(unmatched)} unmatched)"
    )
    if unmatched:
        print(f"Sample unmatched ({min(15, len(unmatched))}): {unmatched[:15]}")


if __name__ == "__main__":
    main()
