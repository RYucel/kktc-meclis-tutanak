"""Re-pack speeches.parquet (sorted by speaker_name, tarih DESC) and documents.parquet (sorted by doc_id)
with small row groups for instant HTTP Range-Request zone-map pruning in DuckDB-WASM,
and build compact space_static/speakers_index.json + space_static/terms_index.json
for 0ms autocomplete, speaker dossiers, and term-by-term MP browsing.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

HF_DATA = Path("data/hf_dataset/data")
STATIC_SPACE = Path("space_static")
OFFICIAL_ROSTERS_PATH = Path("data/official_rosters.json")
CANON_REPORT_PATH = Path("data/parsed/canonicalization_report.json")
STATIC_SPACE.mkdir(parents=True, exist_ok=True)

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)

_CLEAN_NAME_RE = re.compile(r"^[A-ZÇĞİÖŞÜa-zçğıöşüâîûÂÎÛ.\-\s]+$")
_GENERIC_SKIP = {
    "meclis baskani", "gecici baskan", "katip", "gecici katip", "baskan",
    "baskan yardimcisi", "baskan vekili", "divan katibi", "basbakan",
    "meclis reisi", "reis", "reis vekili", "reisi",
    "sesler", "siralardan", "salondan", "milletvekilleri",
    "bir milletvekili", "bazi milletvekilleri", "oturum baskani",
    "tesbit edilemiyen bir milletvekili", "tespit edilemeyen bir milletvekili",
}

# Known spelling aliases for search discoverability (canonical -> extra folded search tokens)
EXTRA_SEARCH_ALIASES = {
    "Yasemi Öztürk": ["yasemin ozturk"],
    "Hüseyin Çavuş": ["huseyin cavus kelle"],
    "İzlem Gürçağ Altuğra": ["izlem gurcag"],
    "Resmiye Eroğlu Canaltay": ["resmiye canaltay"],
    "Sıla Usar İncirli": ["sila usar"],
    "Rauf Raif Denktaş": ["rauf denktas"],
    "Ferdi Sabit Soyer": ["ferdi soyer"],
    "Erol Kazım Andaç": ["erol andac"],
    # Republic of Cyprus (KC) international/Greek transliteration aliases
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

# Ordered newest -> oldest for the UI term switcher
TERM_CATALOG = [
    ("KKTC", 10, "KKTC 10. Dönem", "2022–2026", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 9, "KKTC 9. Dönem", "2018–2022", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 8, "KKTC 8. Dönem", "2013–2018", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 7, "KKTC 7. Dönem", "2009–2013", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 6, "KKTC 6. Dönem", "2005–2009", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 5, "KKTC 5. Dönem", "2003–2005", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 4, "KKTC 4. Dönem", "1998–2003", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 3, "KKTC 3. Dönem", "1993–1998", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 2, "KKTC 2. Dönem", "1990–1993", "KKTC Cumhuriyet Meclisi"),
    ("KKTC", 1, "KKTC 1. Dönem", "1985–1990", "KKTC Cumhuriyet Meclisi"),
    ("KKTC_KM", 1, "KKTC Kurucu Meclisi", "1983–1985", "Kurucu & Geçiş Meclisleri"),
    ("KKTC_ARA", None, "KKTC Aradönem", "1983", "Kurucu & Geçiş Meclisleri"),
    ("KTFD", 2, "KTFD 2. Dönem", "1981–1983", "Kıbrıs Türk Federe Devleti"),
    ("KTFD", 1, "KTFD 1. Dönem", "1976–1981", "Kıbrıs Türk Federe Devleti"),
    ("KTFD_KM", 1, "KTFD Kurucu Meclisi", "1975–1976", "Kıbrıs Türk Federe Devleti"),
    ("KTY", 2, "KTY 2. Dönem", "1970–1975", "Kıbrıs Türk Yönetimi"),
    ("KTY", 1, "KTY 1. Dönem", "1967–1970", "Kıbrıs Türk Yönetimi"),
    ("TCM", None, "Türk Cemaat Meclisi", "1963", "1960–1963 Tarihsel Meclisler"),
    ("KC", 12, "Kıbrıs Cumhuriyeti · ΙΒ΄ Dönem", "2021–2026", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
    ("KC", 11, "Kıbrıs Cumhuriyeti · ΙΑ΄ Dönem", "2016–2021", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
    ("KC", 10, "Kıbrıs Cumhuriyeti · Ι΄ Dönem", "2011–2016", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
    ("KC", 8, "Kıbrıs Cumhuriyeti · Η΄ Dönem", "2001–2006", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
    ("KC", 3, "Kıbrıs Cumhuriyeti · Γ΄ Dönem", "1976–1981", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
    ("KC", None, "Kıbrıs Cumhuriyeti · Α΄ Dönem", "1960–1964", "Kıbrıs Cumhuriyeti Temsilciler Meclisi"),
]


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


def is_displayable_person(name: str) -> bool:
    if not name or len(name) < 5 or len(name) > 52:
        return False
    if not _CLEAN_NAME_RE.match(name):
        return False
    words = name.split()
    if not (2 <= len(words) <= 4):
        return False
    if any(len(w.rstrip(".")) < 2 and not w.endswith(".") for w in words):
        return False
    if len(words[-1].rstrip(".")) <= 2:
        return False
    f = fold_ascii(name)
    if f in _GENERIC_SKIP:
        return False
    if any(
        f.endswith(" " + kw) or f == kw
        for kw in ("bakani", "baskani", "baskan", "katip", "milletvekili", "sozcusu", "komitesi", "bakanligi", "uyesi", "yok", "yardimcisi", "vekili")
    ):
        return False
    if any(f.startswith(pfx) for pfx in ("sayin ", "madde ", "kisim ", "bolum ", "milletvekilleri")):
        return False
    return True


def build_terms_index(
    con: duckdb.DuckDBPyConnection,
    sp_path: str,
    global_speaker_counts: dict[str, int],
    official_rosters_path: Path = OFFICIAL_ROSTERS_PATH,
) -> dict:
    """Build term-by-term MP & speaker directory combining official KKTC 1-10 rosters with parsed archive stats."""
    official_rosters = {}
    if official_rosters_path.exists():
        official_rosters = json.loads(official_rosters_path.read_text(encoding="utf-8"))

    # Global folded -> canonical lookup (highest speech count wins)
    global_by_fold: dict[str, str] = {}
    global_list: list[tuple[str, str, int]] = []
    for nm, cnt in sorted(global_speaker_counts.items(), key=lambda kv: -kv[1]):
        fn = fold_ascii(nm)
        if fn and fn not in global_by_fold:
            global_by_fold[fn] = nm
        global_list.append((nm, fn, cnt))

    # Term summary stats
    term_totals: dict[tuple, tuple[int, int, str | None, str | None]] = {}
    for m, d, docs, turns, mn, mx in con.execute(
        f"""
        SELECT meclis, donem, COUNT(DISTINCT doc_id), COUNT(*), MIN(tarih), MAX(tarih)
        FROM read_parquet('{sp_path}')
        GROUP BY 1, 2
        """
    ).fetchall():
        term_totals[(m, d)] = (int(docs), int(turns), mn, mx)

    # Per-term speaker aggregates
    per_term_rows: dict[tuple, list[dict]] = {}
    rows = con.execute(
        f"""
        SELECT meclis, donem, speaker_name,
               COUNT(*) AS turns,
               SUM(word_count) AS words,
               COUNT(DISTINCT doc_id) AS docs,
               MODE(constituency) FILTER (WHERE constituency IS NOT NULL) AS const,
               MODE(speaker_title) FILTER (
                   WHERE speaker_title IS NOT NULL
                     AND speaker_title NOT IN ('Meclis Başkanı','Katip','Geçici Başkan','Geçici Katip')
               ) AS title
        FROM read_parquet('{sp_path}')
        GROUP BY 1, 2, 3
        ORDER BY turns DESC
        """
    ).fetchall()
    for m, d, nm, turns, words, docs, const, title in rows:
        per_term_rows.setdefault((m, d), []).append(
            {
                "n": nm,
                "f": fold_ascii(nm),
                "s": int(turns),
                "w": int(words or 0),
                "d": int(docs),
                "c": const or "",
                "t": title or "",
            }
        )

    def match_official_to_speaker(off_name: str, term_spks: list[dict]) -> tuple[dict | None, str]:
        fn = fold_ascii(off_name)
        toks = fn.split()
        # 1. Exact folded match in this term
        for sp in term_spks:
            if sp["f"] == fn:
                return sp, sp["n"]
        # 2. Token subset / first+last match in this term
        if len(toks) >= 2:
            cands = []
            for sp in term_spks:
                st = sp["f"].split()
                if len(st) < 2:
                    continue
                if (
                    all(t in st for t in toks)
                    or all(t in toks for t in st)
                    or (toks[0] == st[0] and toks[-1] == st[-1])
                ):
                    cands.append(sp)
            if cands:
                cands.sort(key=lambda x: -x["s"])
                return cands[0], cands[0]["n"]
        # 3. No speeches in THIS term -> resolve to global canonical speaker name if they spoke in other terms
        if fn in global_by_fold:
            return None, global_by_fold[fn]
        if len(toks) >= 2:
            for gnm, gfn, _ in global_list:
                gt = gfn.split()
                if len(gt) >= 2 and (
                    all(t in gt for t in toks)
                    or (toks[0] == gt[0] and toks[-1] == gt[-1])
                ):
                    return None, gnm
        return None, off_name

    def non_member_note(title: str) -> str:
        """Label for someone who spoke in a term with a complete official roster but was not on it."""
        t = fold_ascii(title or "")
        if "muavin" in t or "cumhurbaskan" in t:
            return "Üye değil (cumhurbaşkanlığı makamı)"
        if "bakan" in t:
            return "Bakan, milletvekili değil"
        return "Resmi üye listesinde yok"

    terms_out = []
    for meclis, donem, label, years, group_label in TERM_CATALOG:
        key_tuple = (meclis, donem)
        docs_cnt, turns_cnt, min_d, max_d = term_totals.get(key_tuple, (0, 0, None, None))
        term_id = f"{meclis}:{donem}" if donem is not None else meclis
        term_spks = per_term_rows.get(key_tuple, [])
        off_entry = official_rosters.get(term_id)

        members = []
        used_names = set()

        if off_entry and off_entry.get("members"):
            # Official roster term (KKTC D1-D10)
            for om in off_entry["members"]:
                matched, canon_nm = match_official_to_speaker(om["name"], term_spks)
                if canon_nm in used_names:
                    continue
                used_names.add(canon_nm)
                s_term = matched["s"] if matched else 0
                w_term = matched["w"] if matched else 0
                d_term = matched["d"] if matched else 0
                title = (matched["t"] if matched else "") or ""
                note = om.get("note", "") or ""
                if term_id == "KKTC:2" and om.get("party") == "DMP" and s_term == 0 and not note:
                    note = "1990 DMP Meclis boykotu nedeniyle bu dönemde konuşma kaydı yok"
                members.append(
                    {
                        "n": canon_nm,
                        "o": 1,
                        "p": om.get("party", ""),
                        "c": om.get("constituency", "") or (matched["c"] if matched else ""),
                        "t": title,
                        "s": s_term,
                        "ts": int(global_speaker_counts.get(canon_nm, s_term)),
                        "w": w_term,
                        "d": d_term,
                        "nt": note,
                    }
                )
            # Also add significant non-roster speakers in this term (e.g. external Ministers, President, by-election MPs).
            # Partial rosters (pre-1985, rebuilt from biographies) miss some members, so keep the looser
            # historical thresholds there and never label anyone "not a member".
            partial = bool(off_entry.get("partial"))
            min_turns = 1 if docs_cnt <= 5 else (2 if docs_cnt <= 40 else 3)
            for sp in term_spks:
                if sp["n"] in used_names:
                    continue
                if not is_displayable_person(sp["n"]):
                    continue
                keep = (sp["s"] >= min_turns or sp["d"] >= 2) if partial else (sp["s"] >= 8 and sp["d"] >= 2)
                if keep:
                    used_names.add(sp["n"])
                    members.append(
                        {
                            "n": sp["n"],
                            "o": 0,
                            "p": "",
                            "c": sp["c"],
                            "t": sp["t"],
                            "s": sp["s"],
                            "ts": int(global_speaker_counts.get(sp["n"], sp["s"])),
                            "w": sp["w"],
                            "d": sp["d"],
                            "nt": "" if partial else non_member_note(sp["t"]),
                        }
                    )
        else:
            # 1960-1985 historical terms: derived cleanly from parsed speeches
            min_turns = 1 if docs_cnt <= 5 else (2 if docs_cnt <= 40 else 3)
            min_docs = 1 if docs_cnt <= 40 else 2
            for sp in term_spks:
                if not is_displayable_person(sp["n"]):
                    continue
                if sp["s"] < min_turns and sp["d"] < min_docs:
                    continue
                if docs_cnt > 40 and sp["d"] < 2 and sp["s"] < 6:
                    continue
                used_names.add(sp["n"])
                members.append(
                    {
                        "n": sp["n"],
                        "o": 0,
                        "p": "",
                        "c": sp["c"],
                        "t": sp["t"],
                        "s": sp["s"],
                        "ts": int(global_speaker_counts.get(sp["n"], sp["s"])),
                        "w": sp["w"],
                        "d": sp["d"],
                        "nt": "",
                    }
                )

        # Sort: official MPs first (by speech count desc, then name), then other speakers
        members.sort(key=lambda x: (-x["o"], -x["s"], x["n"]))
        terms_out.append(
            {
                "id": term_id,
                "meclis": meclis,
                "donem": donem,
                "label": label,
                "years": years,
                "group": group_label,
                "docs": docs_cnt,
                "speeches": turns_cnt,
                "min_date": min_d,
                "max_date": max_d,
                "official": bool(off_entry),
                "partial": bool(off_entry and off_entry.get("partial")),
                "members": members,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "terms": terms_out,
    }


def main():
    t0 = time.perf_counter()
    con = duckdb.connect(":memory:")

    sp_in = str(HF_DATA / "speeches.parquet")
    sp_tmp = str(HF_DATA / "speeches_sorted.parquet")
    doc_in = str(HF_DATA / "documents.parquet")
    doc_tmp = str(HF_DATA / "documents_sorted.parquet")
    spk_in = str(HF_DATA / "speakers.parquet")

    print("1. Sorting speeches.parquet by (speaker_name, tarih DESC, turn_index ASC) with row_group_size=32768...")
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{sp_in}')
            ORDER BY speaker_name ASC, tarih DESC NULLS LAST, turn_index ASC
        ) TO '{sp_tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)
        """
    )
    os.replace(sp_tmp, sp_in)
    print(f"   Done ({Path(sp_in).stat().st_size / (1024*1024):.1f} MB) in {time.perf_counter() - t0:.1f}s")

    print("2. Sorting documents.parquet by doc_id with row_group_size=128...")
    t1 = time.perf_counter()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{doc_in}')
            ORDER BY doc_id ASC
        ) TO '{doc_tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128)
        """
    )
    os.replace(doc_tmp, doc_in)
    print(f"   Done ({Path(doc_in).stat().st_size / (1024*1024):.1f} MB) in {time.perf_counter() - t1:.1f}s")

    # Verify zone-map pruning row groups
    rg_info = con.execute(
        f"SELECT COUNT(DISTINCT row_group_id) FROM parquet_metadata('{sp_in}')"
    ).fetchone()[0]
    print(f"   speeches.parquet row groups: {rg_info}")

    print("3. Building compact space_static/speakers_index.json + space_static/terms_index.json...")
    st_row = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT doc_id), SUM(word_count), MIN(tarih), MAX(tarih) FROM read_parquet('{sp_in}')"
    ).fetchone()
    total_all_docs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{doc_in}')").fetchone()[0]
    latest_row = con.execute(
        f"""
        SELECT tarih, meclis, donem, yasama_yili, birlesim, doc_id, source_url
        FROM read_parquet('{doc_in}')
        WHERE tarih IS NOT NULL
        ORDER BY tarih DESC, doc_id DESC
        LIMIT 1
        """
    ).fetchone()

    # Load dedup aliases from canonicalization report if present
    dedup_aliases: dict[str, set[str]] = {}
    if CANON_REPORT_PATH.exists():
        rep = json.loads(CANON_REPORT_PATH.read_text(encoding="utf-8"))
        for canon_nm, items in rep.get("roster_dedup_merges", {}).items():
            for raw_item in items:
                alias_nm = re.sub(r"\s*\[[^\]]+\]$", "", raw_item).strip()
                fa = fold_ascii(alias_nm)
                if fa:
                    dedup_aliases.setdefault(canon_nm, set()).add(fa)
    for canon_nm, extra_list in EXTRA_SEARCH_ALIASES.items():
        for fa in extra_list:
            dedup_aliases.setdefault(canon_nm, set()).add(fa)

    from speaker_index import build_speaker_entries

    speakers_list, spk_summary = build_speaker_entries(
        con, sp_in, spk_in, extra_aliases={k: set(v) for k, v in dedup_aliases.items()}
    )
    global_speaker_counts: dict[str, int] = {e["n"]: e["s"] for e in speakers_list}
    print(
        f"   Speakers: {spk_summary['speakers_verified']:,} verified, {spk_summary['speakers_role_labels']:,} role labels, "
        f"{spk_summary['speakers_plausible']:,} plausible, {spk_summary['speakers_unverified']:,} unmatched "
        f"(verified turn share {spk_summary['verified_turn_share']:.2%})"
    )

    index_payload = {
        "stats": {
            "speeches": int(st_row[0]),
            "docs": int(st_row[1]),
            "total_docs": int(total_all_docs),
            "words": int(st_row[2] or 0),
            "min_date": st_row[3],
            "max_date": st_row[4],
            "speakers": len(speakers_list),
            "speakers_verified": spk_summary["speakers_verified"],
            "verified_turn_share": spk_summary["verified_turn_share"],
            "latest_date": latest_row[0] if latest_row else st_row[4],
            "latest_meclis": latest_row[1] if latest_row else "KKTC",
            "latest_donem": int(latest_row[2]) if (latest_row and latest_row[2] is not None) else None,
            "latest_yasama_yili": int(latest_row[3]) if (latest_row and latest_row[3] is not None) else None,
            "latest_birlesim": int(latest_row[4]) if (latest_row and latest_row[4] is not None) else None,
            "latest_doc_id": latest_row[5] if latest_row else None,
            "latest_source_url": latest_row[6] if latest_row else None,
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "speakers": speakers_list,
    }
    out_json = STATIC_SPACE / "speakers_index.json"
    out_json.write_text(json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"   Wrote {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    terms_payload = build_terms_index(con, sp_in, global_speaker_counts)
    out_terms = STATIC_SPACE / "terms_index.json"
    out_terms.write_text(json.dumps(terms_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    total_term_members = sum(len(t["members"]) for t in terms_payload["terms"])
    print(
        f"   Wrote {out_terms} ({out_terms.stat().st_size / 1024:.1f} KB, "
        f"{len(terms_payload['terms'])} terms, {total_term_members} term-member entries)"
    )

    from analyze_topics import build_speaker_topics, build_topics_index
    topics_payload = build_topics_index(con, sp_in)
    out_topics = STATIC_SPACE / "topics_index.json"
    out_topics.write_text(json.dumps(topics_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"   Wrote {out_topics} ({out_topics.stat().st_size / 1024:.1f} KB, "
        f"{len(topics_payload['topics'])} topics, {len(topics_payload['concepts'])} concepts)"
    )

    spk_topics_payload = build_speaker_topics(con, sp_in)
    out_spk_topics = STATIC_SPACE / "speaker_topics.json"
    out_spk_topics.write_text(json.dumps(spk_topics_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"   Wrote {out_spk_topics} ({out_spk_topics.stat().st_size / 1024:.1f} KB, "
        f"{len(spk_topics_payload['speakers'])} speakers)"
    )

    print("4. Exporting per-term Parquet shards to data/hf_dataset/terms_cache/ for fast term/topic queries...")
    t_sh = time.perf_counter()
    export_term_shards(con, sp_in, HF_DATA.parent / "terms_cache")
    print(f"   Exported term shards in {time.perf_counter() - t_sh:.1f}s")


def term_shard_filename(meclis: str, donem: int | None) -> str:
    return f"{meclis}_{donem}.parquet" if donem is not None else f"{meclis}.parquet"


def export_term_shards(
    con: duckdb.DuckDBPyConnection,
    sp_path: str,
    out_dir: Path,
    only_terms: set[tuple[str, int | None]] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for meclis, donem, _, _, _ in TERM_CATALOG:
        if only_terms is not None and (meclis, donem) not in only_terms:
            continue
        fname = term_shard_filename(meclis, donem)
        out_p = (out_dir / fname).as_posix()
        where = f"meclis = '{meclis}' AND donem = {donem}" if donem is not None else f"meclis = '{meclis}' AND donem IS NULL"
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{sp_path}')
                WHERE {where}
                ORDER BY tarih DESC NULLS LAST, turn_index ASC
            ) TO '{out_p}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)
            """
        )


if __name__ == "__main__":
    main()
