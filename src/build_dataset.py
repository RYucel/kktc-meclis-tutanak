"""Canonicalize speaker names, export Hugging Face Parquet dataset + card, and build SQLite FTS5 search index."""

import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PARSED_DIR = Path("data/parsed")
HF_DIR = Path("data/hf_dataset")
HF_DATA_DIR = HF_DIR / "data"
DB_PATH = Path("data/search.db")

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)
_DIACRITICS = set("çğıöşüÇĞİÖŞÜ")
_CIRCUMFLEX = set("âîûÂÎÛ")
_CLEAN_NAME_RE = re.compile(r"^[A-ZÇĞİÖŞÜa-zçğıöşüâîûÂÎÛ.\-\s]+$")

GENERIC_NON_PERSON = {
    "meclis baskani", "gecici baskan", "katip", "gecici katip", "baskan",
    "basbakan", "sesler", "siralardan", "salondan", "milletvekilleri",
    "bir milletvekili", "bazi milletvekilleri", "tesbit edilemiyen bir milletvekili",
    "tespit edilemeyen bir milletvekili",
}


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def diacritic_score(s: str) -> int:
    return sum(1 for c in s if c in _DIACRITICS)


def is_clean_person_name(name: str) -> bool:
    if not name or len(name) < 5 or len(name) > 55:
        return False
    if not _CLEAN_NAME_RE.match(name):
        return False
    words = name.split()
    if not (2 <= len(words) <= 4):
        return False
    if any(len(w.rstrip(".")) < 2 and not w.endswith(".") for w in words):
        return False
    f = fold_ascii(name)
    if f in GENERIC_NON_PERSON:
        return False
    # Reject names that still look like bare titles, OCR prefixes, or structural labels
    if any(
        f.startswith(pfx)
        for pfx in ("dri ", "or ", "dr ", "dt ", "kisim ", "bolum ", "madde ", "milletvekilleri", "sayin ")
    ):
        return False
    if any(
        f.endswith(" " + kw) or f == kw
        for kw in ("bakani", "baskani", "baskan", "katip", "milletvekili", "sozcusu", "komitesi", "bakanligi", "uyesi", "yok")
    ):
        return False
    # Reject 2-letter broken OCR last tokens like "Salih Co Ar"
    if len(words[-1].rstrip(".")) <= 2:
        return False
    return True


def canonicalize_speakers(speeches: list[dict]) -> dict:
    """Two-pass entity resolution: (1) within-document surname, (2) diacritic + period/global fuzzy match."""
    # Save original parsed name
    for s in speeches:
        s["speaker_name_parsed"] = s["speaker_name"]
        s["canon_method"] = "exact"

    # --- Pass 1: Within-document single-word surname resolution ---
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for s in speeches:
        by_doc[s["doc_id"]].append(s)

    doc_surname_merges = 0
    for doc_id, turns in by_doc.items():
        multi_names = Counter()
        for t in turns:
            if t["speaker_role"] == "milletvekili_bakan" and len(t["speaker_name"].split()) >= 2:
                multi_names[t["speaker_name"]] += 1
        if not multi_names:
            continue
        # Map folded surname -> list of candidate full names, and 2-word prefix -> 3-word full names
        surname_map: dict[str, list[str]] = defaultdict(list)
        prefix2_map: dict[str, list[str]] = defaultdict(list)
        for nm in multi_names:
            f_parts = fold_ascii(nm).split()
            if len(f_parts) >= 2:
                surname_map[f_parts[-1]].append(nm)
            if len(f_parts) >= 3:
                prefix2_map[f"{f_parts[0]} {f_parts[1]}"].append(nm)

        for t in turns:
            if t["speaker_role"] != "milletvekili_bakan":
                continue
            words = t["speaker_name"].split()
            if len(words) == 1 and len(words[0]) >= 4:
                fw = fold_ascii(words[0])
                cands = surname_map.get(fw, [])
                if len(cands) == 1:
                    t["speaker_name"] = cands[0]
                    t["canon_method"] = "doc_surname"
                    doc_surname_merges += 1
            elif len(words) == 2:
                fp2 = fold_ascii(t["speaker_name"])
                cands = prefix2_map.get(fp2, [])
                if len(cands) == 1:
                    t["speaker_name"] = cands[0]
                    t["canon_method"] = "doc_prefix"
                    doc_surname_merges += 1

    # --- Pass 2: Build canonical politician roster ---
    digital_counts = Counter()
    all_counts = Counter()
    period_counts: dict[tuple, Counter] = defaultdict(Counter)

    for s in speeches:
        nm = s["speaker_name"]
        if not is_clean_person_name(nm):
            continue
        all_counts[nm] += 1
        src = s.get("text_source") or ""
        if src in ("docx", "doc", "rtf", "pdf_digital"):
            digital_counts[nm] += 1
        pkey = (s.get("meclis"), s.get("donem"))
        period_counts[pkey][nm] += 1

    # Group clean candidates by folded ASCII key and choose best Turkish display spelling
    folded_groups: dict[str, list[str]] = defaultdict(list)
    for nm, cnt in all_counts.items():
        if digital_counts[nm] >= 3 or cnt >= 6:
            folded_groups[fold_ascii(nm)].append(nm)

    folded_to_canon: dict[str, str] = {}
    for fk, variants in folded_groups.items():
        # Filter out circumflex OCR variants (e.g. "Özkân Murat", "Raîf Denktaş") when clean variants exist
        clean_no_cfx = [v for v in variants if not any(c in _CIRCUMFLEX for c in v)]
        pool = clean_no_cfx if clean_no_cfx else variants
        max_dig = max(digital_counts[v] for v in pool)
        max_all = max(all_counts[v] for v in pool)
        # Only let a more-diacritic variant beat the highest-frequency spelling if it has >=20% support
        # (prevents 4 typos of "Salih Çoşar" overriding 22,000 occurrences of "Salih Coşar")
        def score_variant(v: str):
            has_strong_support = (
                (max_dig >= 10 and digital_counts[v] >= max_dig * 0.20)
                or (max_dig < 10 and all_counts[v] >= max_all * 0.20)
            )
            return (
                1 if has_strong_support else 0,
                diacritic_score(v) if has_strong_support else 0,
                digital_counts[v],
                all_counts[v],
            )
        folded_to_canon[fk] = max(pool, key=score_variant)

    # Track counts & periods per folded key for canonical roster self-deduplication
    fk_counts: Counter = Counter()
    fk_periods: dict[str, set] = defaultdict(set)
    for nm, cnt in all_counts.items():
        fk = fold_ascii(nm)
        if fk in folded_to_canon:
            fk_counts[fk] += cnt
    for pkey, cnts in period_counts.items():
        for nm, c in cnts.items():
            fk = fold_ascii(nm)
            if fk in folded_to_canon:
                fk_periods[fk].add(pkey)

    roster_dedup_merges: dict[str, list[str]] = defaultdict(list)

    def _remap_target(old_canon: str, new_canon: str, reason: str):
        if old_canon == new_canon:
            return
        for k, v in list(folded_to_canon.items()):
            if v == old_canon:
                folded_to_canon[k] = new_canon
        roster_dedup_merges[new_canon].append(f"{old_canon} [{reason}]")

    # 2.1 Spaceless exact consolidation inside canonical roster
    # (e.g. "Ayhan Hal İt Acarkan" -> "Ayhan Halit Acarkan", "Ferdisabitsoyer" -> "Ferdi Sabit Soyer")
    by_nospace: dict[str, list[str]] = defaultdict(list)
    for fk in folded_to_canon:
        ns = fk.replace(" ", "")
        if len(ns) >= 8:
            by_nospace[ns].append(fk)
    for ns, fks in by_nospace.items():
        if len(fks) < 2:
            continue
        winner = max(
            fks,
            key=lambda k: (
                1 if 2 <= len(k.split()) <= 3 else 0,
                fk_counts[k],
            ),
        )
        win_canon = folded_to_canon[winner]
        for k in fks:
            if k != winner:
                _remap_target(folded_to_canon[k], win_canon, "spaceless")

    # 2.2 Turkish dative/ablative suffix consolidation (chair handoffs without apostrophe)
    # (e.g. "Ahmet Savaşana" -> "Ahmet Savaşan", "Sunat Atuna" -> "Sunat Atun", "Vehbi Zeki Sertere" -> "Vehbi Zeki Serter")
    for fk in sorted(folded_to_canon.keys(), key=lambda k: fk_counts[k]):
        for sfx in ("dan", "den", "tan", "ten", "ya", "ye", "a", "e"):
            if fk.endswith(sfx) and len(fk) - len(sfx) >= 6:
                base = fk[: -len(sfx)]
                if base in folded_to_canon and folded_to_canon[base] != folded_to_canon[fk]:
                    if fk_counts[base] >= fk_counts[fk] * 3 and (fk_periods[fk] & fk_periods[base]):
                        _remap_target(folded_to_canon[fk], folded_to_canon[base], "suffix")
                        break

    # 2.3 Same-period high-confidence OCR typo consolidation inside canonical roster
    # (e.g. "Vehbi Zekir Serter" -> "Vehbi Zeki Serter", "Hüseyin Avkran Alanlı" -> "Hüseyin Avkıran Alanlı")
    canon_to_fk_agg: dict[str, str] = {}
    canon_total_cnt: Counter = Counter()
    canon_p_agg: dict[str, set] = defaultdict(set)
    for fk, cnm in folded_to_canon.items():
        canon_total_cnt[cnm] += fk_counts[fk]
        canon_p_agg[cnm].update(fk_periods[fk])
        if cnm not in canon_to_fk_agg or fk_counts[fk] > fk_counts[canon_to_fk_agg[cnm]]:
            canon_to_fk_agg[cnm] = fk

    survivors: list[tuple[str, str]] = []  # (rep_fk, canon_nm)
    for cnm, cnt in canon_total_cnt.most_common():
        rep_fk = canon_to_fk_agg[cnm]
        rep_words = rep_fk.split()
        merged_to = None
        for s_fk, s_cnm in survivors:
            if cnt >= canon_total_cnt[s_cnm] * 0.15:
                continue
            if not (canon_p_agg[cnm] & canon_p_agg[s_cnm]):
                continue
            s_words = s_fk.split()
            if len(rep_words) != len(s_words) or abs(len(rep_fk) - len(s_fk)) > 2:
                continue
            if rep_fk[0] != s_fk[0]:
                continue
            if SequenceMatcher(None, rep_fk, s_fk).ratio() >= 0.89:
                merged_to = s_cnm
                break
        if merged_to:
            _remap_target(cnm, merged_to, "roster_ocr")
            canon_total_cnt[merged_to] += cnt
            canon_p_agg[merged_to].update(canon_p_agg[cnm])
        else:
            survivors.append((rep_fk, cnm))

    # 2.4 Upgrade middle-initial canonicals (e.g. "Ferdi S. Soyer" -> "Ferdi Sabit Soyer",
    # "Vehbi Z. Serter" -> "Vehbi Zeki Serter", "Naci Ti Usar" -> "Naci Talat Usar")
    full3_by_init: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for fk, canon_nm in folded_to_canon.items():
        c_parts = fold_ascii(canon_nm).split()
        if len(c_parts) == 3 and len(c_parts[1]) >= 2 and c_parts[1] not in ("ro", "yo", "ti", "to"):
            full3_by_init[(c_parts[0], c_parts[1][0], c_parts[2])].add(canon_nm)

    _OCR_INIT_NOISE = {"ro", "yo", "ti", "to", "so", "zo", "no", "mo", "ko", "ho", "bo", "co", "do", "fo", "go", "lo", "po"}
    for fk in list(folded_to_canon.keys()):
        parts = fk.split()
        if len(parts) == 3 and (len(parts[1]) == 1 or parts[1] in _OCR_INIT_NOISE):
            matches = full3_by_init.get((parts[0], parts[1][0], parts[2]), set())
            if len(matches) == 1:
                target = next(iter(matches))
                _remap_target(folded_to_canon[fk], target, "middle_initial")

    # 2.5 Unambiguous same-period 2-word <-> 3-word subset consolidation
    # Keeps first name intact: (A==X and B==Z) OR (A==X and B==Y), NEVER drops first name (protects father/son like Rauf Raif Denktaş vs Raif Denktaş)
    canon_total_2: Counter = Counter()
    canon_p_2: dict[str, set] = defaultdict(set)
    for fk, cnm in folded_to_canon.items():
        canon_total_2[cnm] += fk_counts[fk]
        canon_p_2[cnm].update(fk_periods[fk])

    surv_2 = [c for c in canon_total_2 if len(fold_ascii(c).split()) == 2]
    surv_3 = [c for c in canon_total_2 if len(fold_ascii(c).split()) == 3]
    surv_3_parts = {c: fold_ascii(c).split() for c in surv_3}

    for c2 in surv_2:
        a, b = fold_ascii(c2).split()
        if len(a) < 3 or len(b) < 3:
            continue
        p2 = canon_p_2[c2]
        cands = []
        for c3 in surv_3:
            if not (p2 & canon_p_2[c3]):
                continue
            x, y, z = surv_3_parts[c3]
            if (a == x and b == z) or (a == x and b == y):
                cands.append(c3)
        if len(cands) == 1:
            c3 = cands[0]
            x, y, z = surv_3_parts[c3]
            tot = canon_total_2[c2] + canon_total_2[c3]
            # Prefer full 3-word legal name when middle token isn't a bare initial and 3-word form has solid support
            if len(y) >= 2 and canon_total_2[c3] >= 30 and canon_total_2[c3] >= tot * 0.15:
                winner, loser = c3, c2
            elif canon_total_2[c3] > canon_total_2[c2]:
                winner, loser = c3, c2
            else:
                winner, loser = c2, c3
            _remap_target(loser, winner, "subset_2_3")

    canon_set = set(folded_to_canon.values())

    # Refresh full3_by_init with final canonicals for runtime resolve_fuzzy()
    full3_by_init_final: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for canon_nm in canon_set:
        parts = fold_ascii(canon_nm).split()
        if len(parts) == 3 and len(parts[1]) >= 2:
            full3_by_init_final[(parts[0], parts[1][0], parts[2])].append(canon_nm)

    # Spaceless exact map for OCR concatenated names (e.g. "Hüseyinangolemli" -> "Hüseyin Angolemli")
    spaceless_to_canon: dict[str, list[str]] = defaultdict(list)
    for fk, canon_nm in folded_to_canon.items():
        if " " in fk and len(fk.replace(" ", "")) >= 8:
            spaceless_to_canon[fk.replace(" ", "")].append(canon_nm)
    spaceless_unique: dict[str, str] = {
        k: v[0] for k, v in spaceless_to_canon.items() if len(set(v)) == 1
    }

    # Active canonicals per (meclis, donem)
    period_canon_folded: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for pkey, cnts in period_counts.items():
        seen_f = set()
        for nm, c in cnts.items():
            fk = fold_ascii(nm)
            if fk in folded_to_canon and c >= 2 and fk not in seen_f:
                seen_f.add(fk)
                period_canon_folded[pkey].append((fk, folded_to_canon[fk]))

    global_canon_folded = list(folded_to_canon.items())

    # Cache fuzzy resolutions per (pkey, parsed_name)
    fuzzy_cache: dict[tuple, tuple[str, str]] = {}
    merge_examples: dict[str, Counter] = defaultdict(Counter)
    method_counts = Counter()

    def resolve_fuzzy(name: str, pkey: tuple) -> tuple[str, str]:
        ckey = (pkey, name)
        if ckey in fuzzy_cache:
            return fuzzy_cache[ckey]

        fk = fold_ascii(name)
        if not fk or fk in GENERIC_NON_PERSON:
            fuzzy_cache[ckey] = (name, "exact")
            return name, "exact"

        # 2a. Exact diacritic-folded match
        if fk in folded_to_canon:
            target = folded_to_canon[fk]
            method = "exact" if target == name else "diacritic_fold"
            fuzzy_cache[ckey] = (target, method)
            return target, method

        # 2a-spaceless. Exact spaceless match (e.g. "Hüseyinangolemli" -> "Hüseyin Angolemli")
        fk_nospace = fk.replace(" ", "")
        if fk_nospace in spaceless_unique:
            target = spaceless_unique[fk_nospace]
            fuzzy_cache[ckey] = (target, "spaceless_exact")
            return target, "spaceless_exact"

        words = fk.split()
        # 2a-bis. Middle-initial match against 3-word full canonical (e.g., "Vehbi Z. Serter" -> "Vehbi Zeki Serter")
        if len(words) == 3 and (len(words[1]) == 1 or words[1] in _OCR_INIT_NOISE):
            m3 = full3_by_init_final.get((words[0], words[1][0], words[2]), [])
            if len(m3) == 1:
                fuzzy_cache[ckey] = (m3[0], "middle_initial")
                return m3[0], "middle_initial"

        if not (2 <= len(words) <= 4):
            fuzzy_cache[ckey] = (name, "exact")
            return name, "exact"

        def best_match(candidates: list[tuple[str, str]], min_ratio: float) -> str | None:
            best_score = 0.0
            second_score = 0.0
            best_target = None
            for c_fk, c_name in candidates:
                c_words = c_fk.split()
                if abs(len(c_words) - len(words)) > 1:
                    continue
                if abs(len(c_fk) - len(fk)) > 4:
                    continue
                ratio = SequenceMatcher(None, fk, c_fk).ratio()
                if ratio > best_score:
                    second_score = best_score
                    best_score = ratio
                    best_target = c_name
                elif ratio > second_score:
                    second_score = ratio
            if best_target and best_score >= min_ratio:
                # Avoid ambiguous collisions between two equally close names
                if best_score - second_score >= 0.04 or best_score >= 0.93:
                    return best_target
            return None

        # 2b. Period-scoped fuzzy match
        p_cands = period_canon_folded.get(pkey, [])
        if p_cands:
            m_p = best_match(p_cands, min_ratio=0.82)
            if m_p:
                fuzzy_cache[ckey] = (m_p, "fuzzy_period")
                return m_p, "fuzzy_period"

        # 2c. Global strict fuzzy match
        m_g = best_match(global_canon_folded, min_ratio=0.89)
        if m_g:
            fuzzy_cache[ckey] = (m_g, "fuzzy_global")
            return m_g, "fuzzy_global"

        fuzzy_cache[ckey] = (name, "exact")
        return name, "exact"

    for s in speeches:
        if s["speaker_role"] in ("baskan", "gecici_baskan", "katip", "milletvekili_bakan"):
            pkey = (s.get("meclis"), s.get("donem"))
            new_nm, mth = resolve_fuzzy(s["speaker_name"], pkey)
            if new_nm != s["speaker_name"]:
                merge_examples[new_nm][s["speaker_name"]] += 1
                s["speaker_name"] = new_nm
                if s["canon_method"] == "exact":
                    s["canon_method"] = mth
        method_counts[s["canon_method"]] += 1

    report = {
        "canonical_roster_size": len(canon_set),
        "doc_surname_merges": doc_surname_merges,
        "method_counts": dict(method_counts),
        "roster_dedup_merges": dict(sorted(roster_dedup_merges.items())),
        "sample_merges": {
            k: dict(v.most_common(10))
            for k, v in sorted(merge_examples.items(), key=lambda kv: -sum(kv[1].values()))[:80]
        },
    }
    return report


def write_parquet(rows: list[dict], out_path: Path, schema: pa.Schema | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="zstd")
    return table.num_rows, out_path.stat().st_size


def build_sqlite_fts(speeches: list[dict], speakers_rows: list[dict]):
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")

    conn.executescript(
        """
        CREATE TABLE speeches (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id TEXT UNIQUE,
            doc_id TEXT,
            meclis TEXT,
            donem INTEGER,
            yasama_yili INTEGER,
            birlesim INTEGER,
            oturum INTEGER,
            tarih TEXT,
            turn_index INTEGER,
            speaker_name TEXT,
            speaker_name_parsed TEXT,
            speaker_raw TEXT,
            speaker_role TEXT,
            speaker_title TEXT,
            constituency TEXT,
            modifiers TEXT,
            canon_method TEXT,
            text TEXT,
            char_count INTEGER,
            word_count INTEGER,
            text_source TEXT,
            source_file TEXT,
            source_url TEXT
        );

        CREATE INDEX idx_speeches_speaker ON speeches(speaker_name);
        CREATE INDEX idx_speeches_meclis_donem ON speeches(meclis, donem);
        CREATE INDEX idx_speeches_tarih ON speeches(tarih);
        CREATE INDEX idx_speeches_role ON speeches(speaker_role);

        CREATE VIRTUAL TABLE speeches_fts USING fts5(
            speaker_name,
            speaker_title,
            constituency,
            text,
            content='speeches',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TABLE speakers (
            speaker_name TEXT PRIMARY KEY,
            speaker_folded TEXT,
            speech_count INTEGER,
            word_count INTEGER,
            doc_count INTEGER,
            first_date TEXT,
            last_date TEXT,
            meclisler TEXT,
            donemler TEXT,
            titles TEXT,
            constituencies TEXT
        );
        CREATE INDEX idx_speakers_folded ON speakers(speaker_folded);
        """
    )

    cols = [
        "speech_id", "doc_id", "meclis", "donem", "yasama_yili", "birlesim", "oturum",
        "tarih", "turn_index", "speaker_name", "speaker_name_parsed", "speaker_raw",
        "speaker_role", "speaker_title", "constituency", "modifiers", "canon_method",
        "text", "char_count", "word_count", "text_source", "source_file", "source_url",
    ]
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT INTO speeches ({','.join(cols)}) VALUES ({placeholders})"

    batch = []
    for s in speeches:
        batch.append(tuple(s.get(c) for c in cols))
        if len(batch) >= 10000:
            conn.executemany(sql, batch)
            batch.clear()
    if batch:
        conn.executemany(sql, batch)

    conn.execute("INSERT INTO speeches_fts(speeches_fts) VALUES('rebuild');")

    spk_batch = [
        (
            r["speaker_name"],
            fold_ascii(r["speaker_name"]),
            r["speech_count"],
            r["word_count"],
            r["doc_count"],
            r["first_date"],
            r["last_date"],
            ", ".join(r["meclisler"]),
            ", ".join(str(x) for x in r["donemler"]),
            " | ".join(r["titles"][:8]),
            ", ".join(r["constituencies"]),
        )
        for r in speakers_rows
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO speakers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        spk_batch,
    )
    conn.commit()
    conn.close()


def write_dataset_card(
    docs: list[dict],
    speeches: list[dict],
    speakers_rows: list[dict],
    canon_report: dict,
):
    total_chars = sum(s["char_count"] for s in speeches)
    total_words = sum(s["word_count"] for s in speeches)
    by_meclis = Counter(d["meclis"] for d in docs)
    sp_by_meclis = Counter(s["meclis"] for s in speeches)
    by_src = Counter(d["text_source"] for d in docs)
    dates = sorted(d["tarih"] for d in docs if d.get("tarih"))
    date_min = dates[0] if dates else "—"
    date_max = dates[-1] if dates else "—"
    dated_docs = [d for d in docs if d.get("tarih")]
    latest_doc = max(dated_docs, key=lambda d: (d["tarih"], d["doc_id"])) if dated_docs else None
    latest_note = (
        f"`{latest_doc['tarih']}` ({latest_doc['meclis']} Dönem {latest_doc['donem']} · {latest_doc['birlesim']}. Birleşim · `{latest_doc['doc_id']}`)"
        if latest_doc
        else f"`{date_max}`"
    )

    meclis_names = {
        "KKTC": "KKTC Cumhuriyet Meclisi (Dönem 1–10)",
        "KTFD": "Kıbrıs Türk Federe Meclisi",
        "KTY": "Kıbrıs Türk Yönetimi Meclisi",
        "KTFD_KM": "KTFD Kurucu Meclisi",
        "KKTC_KM": "KKTC Kurucu Meclisi",
        "KKTC_ARA": "KKTC Aradönem",
        "KC": "Kıbrıs Cumhuriyeti Dönemi (1960)",
        "TCM": "Türk Cemaat Meclisi (1963)",
    }

    rows_md = []
    for code, d_cnt in by_meclis.most_common():
        rows_md.append(
            f"| `{code}` | {meclis_names.get(code, code)} | {d_cnt:,} | {sp_by_meclis[code]:,} |"
        )

    top_spk_md = []
    for r in speakers_rows[:20]:
        top_spk_md.append(
            f"| {r['speaker_name']} | {r['speech_count']:,} | {r['word_count']:,} | {r['first_date'] or '—'} – {r['last_date'] or '—'} |"
        )

    card = f"""---
language:
- tr
license: cc-by-4.0
task_categories:
- text-retrieval
- text-generation
pretty_name: KKTC Cumhuriyet Meclisi Tutanakları (Konuşmacı Bazlı)
size_categories:
- 1M<n<10M
configs:
- config_name: speeches
  default: true
  data_files:
  - split: train
    path: data/speeches.parquet
- config_name: documents
  data_files:
  - split: train
    path: data/documents.parquet
- config_name: speakers
  data_files:
  - split: train
    path: data/speakers.parquet
---

# KKTC Cumhuriyet Meclisi Tutanakları (Konuşmacı Bazlı Veri Seti)

Kuzey Kıbrıs Türk Cumhuriyeti (KKTC) Cumhuriyet Meclisi ve öncül meclislerin (1960–günümüz) resmî tutanak dergilerinin **konuşmacı bazlı (speaker-turn)** yapılandırılmış Parquet veri seti.

* **Web Arama Uygulaması:** [KKTC Meclis Tutanak Arama (HF Space)](https://huggingface.co/spaces/stevevaius/KKTC-Meclis-Tutanak-Arama)
* **Kaynak:** [KKTC Cumhuriyet Meclisi Evrak Arşivi](https://evrakcm.gov.ct.tr/siteler/belgeler/tutanaklar/Shared%20Documents/Forms/Belge%20Grntleme.aspx)
* **En Son Eklenen Tutanak:** {latest_note}
* **Toplam Belge:** {len(docs):,}
* **Toplam Konuşma Bloğu (Turn):** {len(speeches):,}
* **Benzersiz Konuşmacı (Normalize):** {len(speakers_rows):,}
* **Toplam Kelime / Karakter:** {total_words:,} kelime · {total_chars:,} karakter
* **Tarih Aralığı (bulunanlar):** `{date_min}` – `{date_max}`

## Meclis Dağılımı

| Kod | Meclis | Belge | Konuşma Bloğu |
|---|---|---:|---:|
{chr(10).join(rows_md)}

## Metin Kaynağı Dağılımı

{", ".join(f"`{k}`: {v:,}" for k, v in by_src.most_common())}

## Alt Veri Setleri (Configs)

1. **`speeches` (varsayılan):** Her satır tek bir konuşmacı sözüdür (`speech_id`, `doc_id`, `meclis`, `donem`, `yasama_yili`, `birlesim`, `oturum`, `tarih`, `turn_index`, `speaker_name`, `speaker_name_parsed`, `speaker_raw`, `speaker_role`, `speaker_title`, `constituency`, `modifiers`, `canon_method`, `text`, `char_count`, `word_count`, `text_source`, `source_file`, `source_url`).
2. **`documents`:** Belge düzeyinde tam metin ve meta veriler.
3. **`speakers`:** Konuşmacı dizini (konuşma sayısı, kelime sayısı, aktif tarih aralığı, unvanlar ve seçim bölgeleri).

## En Çok Söz Alan 20 Konuşmacı

| Konuşmacı | Konuşma | Kelime | Tarih Aralığı |
|---|---:|---:|---|
{chr(10).join(top_spk_md)}

## Örnek Kullanım (Python)

```python
from datasets import load_dataset

ds = load_dataset("stevevaius/KKTC_Meclis_Tutanak", "speeches", split="train")
erhurman = ds.filter(lambda x: x["speaker_name"] == "Tufan Erhürman")
print(len(erhurman), erhurman[0]["tarih"], erhurman[0]["text"][:200])
```

## Lisans ve Atıf

Orijinal tutanak belgelerinin hakları **KKTC Cumhuriyet Meclisi**'ne aittir. Bu veri seti arşiv, araştırma ve konuşmacı bazlı arama amacıyla derlenmiştir; yapılandırma, ayrıştırma ve normalizasyon katmanı **CC BY 4.0** ile paylaşılmıştır. Atıf: *KKTC Cumhuriyet Meclisi Tutanak Dergisi, evrakcm.gov.ct.tr*.
"""
    (HF_DIR / "README.md").write_text(card, encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sqlite", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading parsed documents and speeches...")
    docs = [json.loads(l) for l in (PARSED_DIR / "documents.jsonl").open("r", encoding="utf-8") if l.strip()]
    speeches = [json.loads(l) for l in (PARSED_DIR / "speeches.jsonl").open("r", encoding="utf-8") if l.strip()]
    print(f"Loaded {len(docs):,} docs and {len(speeches):,} speeches.")

    print("Running 2-pass speaker canonicalization...")
    canon_report = canonicalize_speakers(speeches)
    (PARSED_DIR / "canonicalization_report.json").write_text(
        json.dumps(canon_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Canonical roster: {canon_report['canonical_roster_size']} | "
        f"methods={canon_report['method_counts']}"
    )

    # Refresh per-document speaker lists & build speaker profiles
    doc_speakers = defaultdict(set)
    spk_stats = defaultdict(
        lambda: {
            "speech_count": 0,
            "word_count": 0,
            "docs": set(),
            "dates": [],
            "meclisler": set(),
            "donemler": set(),
            "titles": Counter(),
            "constituencies": Counter(),
            "raw_variants": Counter(),
        }
    )

    for s in speeches:
        nm = s["speaker_name"]
        doc_speakers[s["doc_id"]].add(nm)
        st = spk_stats[nm]
        st["speech_count"] += 1
        st["word_count"] += s["word_count"]
        st["docs"].add(s["doc_id"])
        if s.get("tarih"):
            st["dates"].append(s["tarih"])
        if s.get("meclis"):
            st["meclisler"].add(s["meclis"])
        if s.get("donem") is not None:
            st["donemler"].add(int(s["donem"]))
        if s.get("speaker_title"):
            st["titles"][s["speaker_title"]] += 1
        if s.get("constituency"):
            st["constituencies"][s["constituency"]] += 1
        if s.get("speaker_name_parsed"):
            st["raw_variants"][s["speaker_name_parsed"]] += 1

    for d in docs:
        d["konusmacilar"] = sorted(doc_speakers.get(d["doc_id"], set()))

    speakers_rows = []
    for nm, st in spk_stats.items():
        dates_s = sorted(st["dates"])
        speakers_rows.append(
            {
                "speaker_name": nm,
                "speech_count": st["speech_count"],
                "word_count": st["word_count"],
                "doc_count": len(st["docs"]),
                "first_date": dates_s[0] if dates_s else None,
                "last_date": dates_s[-1] if dates_s else None,
                "meclisler": sorted(st["meclisler"]),
                "donemler": sorted(st["donemler"]),
                "titles": [t for t, _ in st["titles"].most_common(15)],
                "constituencies": [c for c, _ in st["constituencies"].most_common(5)],
                "raw_variants": [v for v, _ in st["raw_variants"].most_common(20)],
            }
        )
    speakers_rows.sort(key=lambda r: (-r["speech_count"], r["speaker_name"]))

    print("Writing Hugging Face Parquet files...")
    n_sp, sz_sp = write_parquet(speeches, HF_DATA_DIR / "speeches.parquet")
    n_doc, sz_doc = write_parquet(docs, HF_DATA_DIR / "documents.parquet")
    n_spk, sz_spk = write_parquet(speakers_rows, HF_DATA_DIR / "speakers.parquet")
    print(
        f"  speeches.parquet:  {n_sp:,} rows ({sz_sp / (1024*1024):.1f} MB)\n"
        f"  documents.parquet: {n_doc:,} rows ({sz_doc / (1024*1024):.1f} MB)\n"
        f"  speakers.parquet:  {n_spk:,} rows ({sz_spk / (1024*1024):.2f} MB)"
    )

    write_dataset_card(docs, speeches, speakers_rows, canon_report)
    print(f"Wrote Hugging Face Dataset Card: {HF_DIR / 'README.md'}")

    if not args.skip_sqlite:
        print("Building SQLite FTS5 search index...")
        build_sqlite_fts(speeches, speakers_rows)
        print(f"Built {DB_PATH} ({DB_PATH.stat().st_size / (1024*1024):.1f} MB) in {time.time() - t0:.1f}s total.")
    else:
        print(f"Skipped SQLite FTS5 build (--skip-sqlite). Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
