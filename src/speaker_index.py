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
        entry = {
            "n": nm,
            "f": base_f if not aliases else " | ".join([base_f] + aliases[:max_aliases]),
            "s": int(sc),
            "w": int(wc or 0),
            "d": int(dc or 0),
            "a": (fd or "")[:10],
            "b": (ld or "")[:10],
            "c": list(consts)[:3] if consts else [],
            "t": list(titles)[:5] if titles else [],
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
