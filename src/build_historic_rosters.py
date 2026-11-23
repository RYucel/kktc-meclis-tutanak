"""Add pre-1985 assembly rosters (TCM, KTY:2, KTFD_KM:1, KTFD:1, KTFD:2, KKTC_KM:1) to data/official_rosters.json.

Source: data/cm_bios_terms.json (parse_cm_bios.py over cm.gov.ct.tr former-MP biographies). Names are mapped to
the archive's canonical speaker names through space_static/mp_bios.json (same biography URL), so roster members
match existing speakers. These rosters are marked "partial": the biography collection does not cover every member,
so a speaker missing from them is NOT evidence that he or she was not a member.

Usage: python src/build_historic_rosters.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIOS_TERMS = ROOT / "data" / "cm_bios_terms.json"
MP_BIOS = ROOT / "space_static" / "mp_bios.json"
ROSTERS = ROOT / "data" / "official_rosters.json"

TERMS = {
    "TCM": {"meclis": "TCM", "donem": None, "years": [1960, 1970], "label": "Türk Cemaat Meclisi (1960–1970)"},
    "KTY:2": {"meclis": "KTY", "donem": 2, "years": [1970, 1975], "label": "Kıbrıs Türk Yönetimi 2. Dönem (1970–1975)"},
    "KTFD_KM:1": {"meclis": "KTFD_KM", "donem": 1, "years": [1975, 1976], "label": "KTFD Kurucu Meclisi (1975–1976)"},
    "KTFD:1": {"meclis": "KTFD", "donem": 1, "years": [1976, 1981], "label": "KTFD 1. Dönem (1976–1981)"},
    "KTFD:2": {"meclis": "KTFD", "donem": 2, "years": [1981, 1983], "label": "KTFD 2. Dönem (1981–1983)"},
    "KKTC_KM:1": {"meclis": "KKTC_KM", "donem": 1, "years": [1983, 1985], "label": "KKTC Kurucu Meclisi (1983–1985)"},
}

# Biography names whose spelling the bio parser could not recover (dotless-i OCR, truncated surnames)
NAME_FIX = {
    "Ahmet Mehmet Mutallıp": "Ahmet Mehmet Mutallip",
    "Ali Niyazi Fıkret": "Ali Niyazi Fikret",
    "Kadriye Ahmet Herkecı": "Kadriye Ahmet Herkeci",
    "Mehmet Derviş Kayımbasoglu": "Mehmet Derviş Kayımbaşoğlu",
    "Orhan Zıhnı": "Orhan Zihni Bilgehan",
    "Turgut Alı": "Turgut Ali",
    "Haluk Avni": "Haluk Avni Akman",
    "Nevzat S Uzunoğlu": "Nevzat S. Uzunoğlu",
}


def build() -> dict:
    people = json.loads(BIOS_TERMS.read_text(encoding="utf-8"))
    bios = json.loads(MP_BIOS.read_text(encoding="utf-8")).get("bios", {}) if MP_BIOS.exists() else {}
    url2name = {(v.get("url") or "").lower(): nm for nm, v in bios.items() if v.get("url")}
    out = {tid: {**meta, "partial": True, "source": "https://cm.gov.ct.tr/GecmisMiletvekilleri", "members": []} for tid, meta in TERMS.items()}
    for p in people:
        if p.get("alias_of"):
            continue
        name = url2name.get((p.get("doc_url") or "").lower()) or NAME_FIX.get(p["name"], p["name"])
        for tid, ev in (p.get("terms") or {}).items():
            if tid not in out:
                continue
            aliases = [a for a in {p["name"], *p.get("aliases", [])} if a and a != name]
            out[tid]["members"].append({
                "name": name,
                "constituency": ev.get("constituency") or "",
                "party": ev.get("party") or "",
                "note": "",
                "evidence": ev.get("evidence", ""),
                "bio_url": p.get("doc_url", ""),
                **({"aliases": sorted(aliases)} if aliases else {}),
            })
    for t in out.values():
        seen, uniq = set(), []
        for m in sorted(t["members"], key=lambda m: m["name"]):
            if m["name"] not in seen:
                seen.add(m["name"])
                uniq.append(m)
        t["members"] = uniq
    return out


def main():
    raw = ROSTERS.read_text(encoding="utf-8")
    rosters = json.loads(raw)
    new = build()
    rosters.update(new)
    ROSTERS.write_text(json.dumps(rosters, ensure_ascii=False, indent=2) + ("\n" if raw.endswith("\n") else ""), encoding="utf-8")
    print({k: len(v["members"]) for k, v in new.items()})


if __name__ == "__main__":
    main()
