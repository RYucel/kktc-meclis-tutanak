"""Shared builder for the Static Space ``speakers_index.json`` speaker entries.

Both ``optimize_for_wasm.py`` (full rebuild) and ``sync_incremental.py`` (weekly sync) used to build
the entries independently; the weekly path silently dropped the folded search aliases. This module
is now the single source of truth and adds two compact flags consumed by ``space_static/index.html``:

* ``"v": 1`` — verified person (official roster member or high-confidence anchor identity)
* ``"r": 1`` — generic role label ("Meclis Başkanı", "Katip", "İkinci Başkan" ...)
* ``"p": 1`` — plausible person: a clean, rare name that matches no known identity
  ("Fazıl Küçük", "Emine Çolak"); listed normally, never badged as an OCR label

Entries without any flag are unmatched labels (mostly OCR fragments); the UI lists at most a few of
them, muted and badged, below the flagged matches. Older UIs simply ignore the extra keys.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from speaker_resolution import ROLE_TITLE_NAMES, SpeakerResolver, fold, load_rosters


def stats_from_parquet(con, sp_path: str) -> dict[str, dict]:
    """speaker_name -> {"count", "docs", "years": Counter} straight from speeches.parquet."""
    sp = Path(sp_path).as_posix()
    stats: dict[str, dict] = {}
    for nm, cnt, docs in con.execute(
        f"SELECT speaker_name, COUNT(*), COUNT(DISTINCT doc_id) FROM read_parquet('{sp}') "
        "WHERE speaker_name IS NOT NULL GROUP BY 1"
    ).fetchall():
        stats[nm] = {"count": int(cnt), "docs": int(docs), "years": Counter()}
    for nm, yr, cnt in con.execute(
        f"SELECT speaker_name, TRY_CAST(substr(tarih, 1, 4) AS INTEGER) AS yr, COUNT(*) FROM read_parquet('{sp}') "
        "WHERE speaker_name IS NOT NULL AND tarih IS NOT NULL GROUP BY 1, 2"
    ).fetchall():
        if yr is not None and nm in stats:
            stats[nm]["years"][int(yr)] = int(cnt)
    return stats


def core_alias(variant: str) -> str | None:
    core = SpeakerResolver.core_tokens(variant)
    return " ".join(core) if core else None


_DIRTY_TITLE_RE = re.compile(r"[§îâûÎÂÛ0-9*#£>^~|]")


def _rank_titles(raw_counts: list[tuple[str, int]], limit: int = 5) -> list[str]:
    """Merge punctuation/diacritic variants of titles, drop 1-turn OCR noise, return top-N by frequency."""
    if not raw_counts:
        return []
    groups: dict[str, dict] = {}
    for title, cnt in raw_counts:
        t = re.sub(r"\s+", " ", (title or "").strip())
        if not t or len(t) < 3 or _DIRTY_TITLE_RE.search(t):
            continue
        key = fold(t).replace(" ", "")
        if not key:
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = {"total": cnt, "best": t, "best_c": cnt}
        else:
            g["total"] += cnt
            # Prefer clean punctuation (no " ,") and higher raw count
            cand_score = (0 if " ," in t else 1, cnt)
            best_score = (0 if " ," in g["best"] else 1, g["best_c"])
            if cand_score > best_score:
                g["best"], g["best_c"] = t, cnt
    ranked = sorted(groups.values(), key=lambda g: (-g["total"], g["best"]))
    if not ranked:
        return []
    total_all = sum(g["total"] for g in ranked)
    min_cnt = 2 if total_all >= 10 else 1
    filtered = [g["best"] for g in ranked if g["total"] >= min_cnt]
    return (filtered or [ranked[0]["best"]])[:limit]


def _rank_constituencies(raw_counts: list[tuple[str, int]], limit: int = 3) -> list[str]:
    """Return speaker constituencies sorted by frequency, dropping stray 1-2 turn OCR/parse outliers."""
    if not raw_counts:
        return []
    ranked = sorted(raw_counts, key=lambda x: (-x[1], x[0]))
    top_cnt = ranked[0][1]
    min_cnt = max(3, int(top_cnt * 0.05)) if top_cnt >= 20 else 1
    return [c for c, cnt in ranked if cnt >= min_cnt][:limit] or [ranked[0][0]]


def build_speaker_entries(
    con,
    sp_path: str,
    spk_path: str,
    rosters: dict | None = None,
    extra_aliases: dict[str, set[str]] | None = None,
    max_aliases: int = 8,
) -> tuple[list[dict], dict]:
    """Return (entries, summary) for speakers_index.json."""
    sp = Path(sp_path).as_posix()
    spk = Path(spk_path).as_posix()
    rosters = load_rosters() if rosters is None else rosters
    # Roster spelling variants (e.g. Resmi Gazete Latin forms of 1960 Greek Cypriot members) stay searchable
    roster_aliases: dict[str, set[str]] = {}
    for term in rosters.values():
        for mem in term.get("members", []):
            for a in mem.get("aliases", []) or []:
                if fold(a):
                    roster_aliases.setdefault(mem["name"], set()).add(fold(a))
    if roster_aliases:
        merged = {k: set(v) for k, v in (extra_aliases or {}).items()}
        for k, v in roster_aliases.items():
            merged.setdefault(k, set()).update(v)
        extra_aliases = merged

    resolver = SpeakerResolver(stats_from_parquet(con, sp), rosters)
    verified = resolver.verified_names()

    timelines: dict[str, list[list]] = {}
    for nm, yr, cnt in con.execute(
        f"""
        SELECT speaker_name, substr(tarih, 1, 4) AS yr, COUNT(*) AS cnt
        FROM read_parquet('{sp}')
        WHERE tarih IS NOT NULL AND length(tarih) >= 4
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall():
        timelines.setdefault(nm, []).append([yr, int(cnt)])

    # Original labels per canonical speaker, most frequent first (speakers.parquet raw_variants is unordered)
    variants: dict[str, list[str]] = {}
    sp_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sp}')").fetchall()}
    if "speaker_name_parsed" in sp_cols:
        for nm, v, _c in con.execute(
            f"""
            SELECT speaker_name, speaker_name_parsed, COUNT(*) AS c FROM read_parquet('{sp}')
            WHERE speaker_name_parsed IS NOT NULL AND speaker_name_parsed <> speaker_name
            GROUP BY 1, 2 ORDER BY 1, c DESC, 2
            """
        ).fetchall():
            lst = variants.setdefault(nm, [])
            if len(lst) < 40:
                lst.append(v)

    # Frequency-ranked titles & constituencies directly from speeches.parquet
    # (speakers.parquet list_distinct() order is arbitrary and keeps punctuation duplicates / 1-turn typos)
    raw_titles: dict[str, list[tuple[str, int]]] = {}
    for nm, t, c in con.execute(
        f"""
        SELECT speaker_name, speaker_title, COUNT(*) AS c
        FROM read_parquet('{sp}')
        WHERE speaker_name IS NOT NULL AND speaker_title IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchall():
        raw_titles.setdefault(nm, []).append((t, int(c)))

    raw_consts: dict[str, list[tuple[str, int]]] = {}
    for nm, cst, c in con.execute(
        f"""
        SELECT speaker_name, constituency, COUNT(*) AS c
        FROM read_parquet('{sp}')
        WHERE speaker_name IS NOT NULL AND constituency IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchall():
        raw_consts.setdefault(nm, []).append((cst, int(c)))

    rows = con.execute(
        f"""
        SELECT speaker_name, speech_count, word_count, doc_count,
               first_date, last_date, titles, constituencies
        FROM read_parquet('{spk}')
        ORDER BY speech_count DESC, speaker_name ASC
        """
    ).fetchall()

    entries: list[dict] = []
    n_verified = n_role = 0
    turns_verified = turns_total = 0
    n_plausible = 0
    for nm, sc, wc, dc, fd, ld, titles, consts in rows:
        raw_variants = variants.get(nm, [])
        base_f = fold(nm)
        is_role = nm in ROLE_TITLE_NAMES
        is_person = nm in verified and not is_role
        aliases: list[str] = []
        if is_person:
            for a in sorted((extra_aliases or {}).get(nm, set())):
                if a and a != base_f and a not in aliases:
                    aliases.append(a)
            for v in raw_variants or []:
                a = core_alias(v)
                if a and a != base_f and a not in aliases:
                    aliases.append(a)
                if len(aliases) >= max_aliases:
                    break
        clean_consts = _rank_constituencies(raw_consts.get(nm, []), limit=3) if nm in raw_consts else (list(consts)[:3] if consts else [])
        clean_titles = _rank_titles(raw_titles.get(nm, []), limit=5) if nm in raw_titles else (list(titles)[:5] if titles else [])
        entry = {
            "n": nm,
            "f": base_f if not aliases else " | ".join([base_f] + aliases[:max_aliases]),
            "s": int(sc),
            "w": int(wc or 0),
            "d": int(dc or 0),
            "a": (fd or "")[:10],
            "b": (ld or "")[:10],
            "c": clean_consts,
            "t": clean_titles,
        }
        if is_person:
            entry["v"] = 1
            n_verified += 1
            turns_verified += int(sc)
        elif is_role:
            entry["r"] = 1
            n_role += 1
            turns_verified += int(sc)
        elif resolver.is_plausible_person(nm):
            entry["p"] = 1
            n_plausible += 1
        turns_total += int(sc)
        if int(sc) >= 5 and nm in timelines:
            entry["y"] = timelines[nm]
        entries.append(entry)

    summary = {
        "speakers_verified": n_verified,
        "speakers_role_labels": n_role,
        "speakers_plausible": n_plausible,
        "speakers_unverified": len(entries) - n_verified - n_role - n_plausible,
        "verified_turn_share": round(turns_verified / turns_total, 4) if turns_total else 0.0,
    }
    return entries, summary


def merge_alias_sets(*sources: dict[str, set[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for src in sources:
        for k, v in (src or {}).items():
            out.setdefault(k, set()).update(v)
    return out


def aliases_from_canon_report(report: dict) -> dict[str, set[str]]:
    """Folded aliases from build_dataset's canonicalization_report.json (roster_dedup_merges)."""
    out: dict[str, set[str]] = {}
    for canon_nm, items in (report or {}).get("roster_dedup_merges", {}).items():
        for raw_item in items:
            alias_nm = re.sub(r"\s*\[[^\]]+\]$", "", raw_item).strip()
            fa = fold(alias_nm)
            if fa:
                out.setdefault(canon_nm, set()).add(fa)
    return out
