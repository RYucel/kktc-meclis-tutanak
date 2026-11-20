"""Fetch official KKTC Cumhuriyet Meclisi term-by-term MP rosters (Dönem 1–10, 1985–2026)
from Wikipedia (YSK-sourced tables) and cm.gov.ct.tr/Milletvekillerimiz1.aspx,
saving a clean curated JSON to data/official_rosters.json.
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup

OUT_PATH = Path("data/official_rosters.json")

PARTY_ABBR = {
    "ulusal birlik partisi": "UBP",
    "cumhuriyetçi türk partisi": "CTP",
    "demokrat parti": "DP",
    "toplumcu kurtuluş partisi": "TKP",
    "toplumcu demokrasi partisi": "TDP",
    "halkın partisi": "HP",
    "yeniden doğuş partisi": "YDP",
    "yeni doğuş partisi": "YDP",
    "barış ve demokrasi hareketi": "BDH",
    "özgürlük ve reform partisi": "ÖRP",
    "demokratik mücadele partisi": "DMP",
    "yeni kıbrıs partisi": "YKP",
    "bağımsız": "Bağımsız",
}


def clean_wiki_text(s: str) -> str:
    # Remove <ref>...</ref> and <ref .../>
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S | re.I)
    s = re.sub(r"<ref[^>]*/\s*>", "", s, flags=re.I)
    # Remove HTML comments
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    # Remove templates like {{...}} (non-nested)
    for _ in range(3):
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    # Replace [[Target|Display]] -> Display, [[Target]] -> Target
    s = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    # Remove bold/italic wiki markup & HTML tags
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_party(raw: str) -> str:
    t = clean_wiki_text(raw).strip()
    if not t:
        return ""
    tl = t.replace("İ", "i").replace("I", "ı").lower()
    for full, abbr in PARTY_ABBR.items():
        if full in tl:
            return abbr
    # Already short?
    if len(t) <= 10:
        return t
    return t


def norm_constituency(raw: str) -> str:
    t = clean_wiki_text(raw).strip()
    t = re.sub(r"\s+İlçesi$", "", t, flags=re.I).strip()
    # Canonicalize casing
    m = {
        "lefkoşa": "Lefkoşa",
        "gazimağusa": "Gazimağusa",
        "mağusa": "Gazimağusa",
        "girne": "Girne",
        "güzelyurt": "Güzelyurt",
        "iskele": "İskele",
        "lefke": "Lefke",
    }
    key = t.replace("İ", "i").replace("I", "ı").lower()
    return m.get(key, t)


def fetch_wiki_html(title: str) -> str:
    q = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
    })
    req = urllib.request.Request(
        f"https://tr.wikipedia.org/w/api.php?{q}",
        headers={"User-Agent": "KKTCMeclisArchiveBot/1.0 (https://github.com/RYucel/kktc-meclis-tutanak)"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["parse"]["text"]["*"]


def parse_wiki_mp_table(html: str) -> list[dict]:
    """Parse rendered HTML wikitable (handles rowspan accurately!)."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_=lambda c: c and "wikitable" in c)
    mps = []
    seen = set()

    for tbl in tables:
        # Check header to ensure this is an MP table
        header_row = tbl.find("tr")
        if not header_row:
            continue
        headers = [th.get_text(" ", strip=True).lower() for th in header_row.find_all(["th", "td"])]
        if not any("milletvekili" in h or "ad" in h for h in headers):
            continue

        # Expand rowspan/colspan into a clean 2D grid
        grid = []
        active_spans = {}  # col_idx -> (remaining_rows, text)
        for tr in tbl.find_all("tr")[1:]:
            # Skip sub-header / divider rows that only have <th> and reset active spans
            tds = tr.find_all(["td", "th"])
            if not tds or all(el.name == "th" for el in tds):
                active_spans.clear()
                continue
            row = []
            col = 0
            td_iter = iter(tds)
            while True:
                if col in active_spans:
                    rem, val = active_spans[col]
                    row.append(val)
                    if rem <= 1:
                        del active_spans[col]
                    else:
                        active_spans[col] = (rem - 1, val)
                    col += 1
                    continue
                try:
                    cell = next(td_iter)
                except StopIteration:
                    break
                # Strip citation superscripts [1], [2]
                for sup in cell.find_all("sup"):
                    sup.decompose()
                val = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                rowspan = int(cell.get("rowspan", 1) or 1)
                colspan = int(cell.get("colspan", 1) or 1)
                for _ in range(colspan):
                    row.append(val)
                    if rowspan > 1:
                        active_spans[col] = (rowspan - 1, val)
                    col += 1
            if row:
                grid.append(row)

        # Map columns by header positions
        # Typical columns: [Seçim Bölgesi, Milletvekili, (color swatch), Siyasi Parti, (Değişiklik/Not)]
        const_idx = 0
        name_idx = 1
        party_idx = None
        note_idx = None
        # Compute expanded header indices
        exp_headers = []
        for th in header_row.find_all(["th", "td"]):
            cs = int(th.get("colspan", 1) or 1)
            txt = th.get_text(" ", strip=True).lower()
            for _ in range(cs):
                exp_headers.append(txt)
        for idx, h in enumerate(exp_headers):
            if "bölge" in h or "ilçe" in h:
                const_idx = idx
            elif "milletvekili" in h or "üye" in h:
                name_idx = idx
            elif "parti" in h:
                party_idx = idx  # takes last column of colspan=2 (actual party name, not swatch)
            elif "değişiklik" in h or "not" in h or "görev" in h:
                note_idx = idx

        for row in grid:
            if len(row) <= name_idx:
                continue
            raw_name = row[name_idx].strip()
            if not raw_name or len(raw_name) < 4:
                continue
            # Clean footnotes / parenthetical dates if any
            name = re.sub(r"\s*\[[^\]]*\]", "", raw_name).strip()
            const = norm_constituency(row[const_idx]) if const_idx < len(row) else ""
            party = norm_party(row[party_idx]) if (party_idx is not None and party_idx < len(row)) else ""
            if not party and party_idx is not None and party_idx - 1 < len(row):
                party = norm_party(row[party_idx - 1])
            note = row[note_idx].strip() if (note_idx is not None and note_idx < len(row)) else ""

            key = name.lower()
            if key not in seen:
                seen.add(key)
                mps.append({
                    "name": name,
                    "constituency": const,
                    "party": party,
                    "note": note,
                })

            # Also check if a by-election replacement / resigned MP is named in note (e.g., Kudret Özersay)
            # We'll keep note so we can inspect/extract replacements explicitly if needed.
    return mps


def fetch_cm_current_d10() -> list[dict]:
    """Fetch current 10th term MPs from cm.gov.ct.tr/Milletvekillerimiz1.aspx."""
    url = "https://cm.gov.ct.tr/Milletvekillerimiz1.aspx"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", id=re.compile("GridView1"))
    if not tbl:
        return []
    out = []
    for tr in tbl.find_all("tr")[1:]:
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) >= 3 and tds[0]:
            out.append({
                "name": tds[0].strip(),
                "constituency": norm_constituency(tds[1]),
                "party": norm_party(tds[2]),
                "note": "Resmî cm.gov.ct.tr",
            })
    return out


WIKI_TYPO_FIXES = {
    "İsmail Koreli": "İbrahim Koreli",
    "Hasan Ataker": "Hakan Ataker",
    "Biray Hamzaoğluları": "Biray Hamzaoğulları",
    "Yasemin Öztürk": "Yasemi Öztürk",
    "Ahmet Gulle": "Ahmet Gülle",
}

EXTRA_TERM_MEMBERS = {
    # Resigned during 10th term; replaced by Sami Özuslu in 2023 by-election
    "KKTC:10": [
        {"name": "Kudret Özersay", "constituency": "Lefkoşa", "party": "HP", "note": "İstifa etti (yerine ara seçimle Sami Özuslu seçildi)."},
    ],
}


def main():
    rosters = {}
    for d in range(1, 11):
        title = f"KKTC Cumhuriyet Meclisi {d}. dönem milletvekilleri listesi"
        html = fetch_wiki_html(title)
        mps = parse_wiki_mp_table(html)
        for m in mps:
            m["name"] = WIKI_TYPO_FIXES.get(m["name"], m["name"])
        key = f"KKTC:{d}"
        existing = {m["name"].lower() for m in mps}
        for extra in EXTRA_TERM_MEMBERS.get(key, []):
            if extra["name"].lower() not in existing:
                mps.append(extra)
                existing.add(extra["name"].lower())
        rosters[key] = {
            "meclis": "KKTC",
            "donem": d,
            "source": f"https://tr.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "members": mps,
        }
        print(f"  {key:8s}: {len(mps):3d} MPs parsed (sample: {[m['name'] for m in mps[:3]]})")

    # Also inspect cm.gov.ct.tr D10 and merge any differences
    cm_d10 = fetch_cm_current_d10()
    print(f"  cm.gov.ct.tr D10: {len(cm_d10)} current MPs")
    rosters["KKTC:10_cm_official"] = {
        "meclis": "KKTC",
        "donem": 10,
        "source": "https://cm.gov.ct.tr/Milletvekillerimiz1.aspx",
        "members": cm_d10,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rosters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
