"""Re-pack speeches.parquet (sorted by speaker_name, tarih DESC) and documents.parquet (sorted by doc_id)
with small row groups for instant HTTP Range-Request zone-map pruning in DuckDB-WASM,
and build compact space_static/speakers_index.json for 0ms autocomplete + speaker dossiers.
"""

import json
import os
import re
import time
from pathlib import Path

import duckdb

HF_DATA = Path("data/hf_dataset/data")
STATIC_SPACE = Path("space_static")
STATIC_SPACE.mkdir(parents=True, exist_ok=True)

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


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

    print("3. Building compact space_static/speakers_index.json...")
    from datetime import datetime, timezone
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

    # Yearly timelines for speakers with >= 5 speeches
    tl_rows = con.execute(
        f"""
        SELECT speaker_name, substr(tarih, 1, 4) AS yr, COUNT(*) AS cnt
        FROM read_parquet('{sp_in}')
        WHERE tarih IS NOT NULL AND length(tarih) >= 4
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    timelines: dict[str, list[list]] = {}
    for nm, yr, cnt in tl_rows:
        timelines.setdefault(nm, []).append([yr, int(cnt)])

    spk_rows = con.execute(
        f"""
        SELECT speaker_name, speech_count, word_count, doc_count,
               first_date, last_date, titles, constituencies
        FROM read_parquet('{spk_in}')
        ORDER BY speech_count DESC
        """
    ).fetchall()

    speakers_list = []
    for r in spk_rows:
        nm, sc, wc, dc, fd, ld, titles, consts = r
        entry = {
            "n": nm,
            "f": fold_ascii(nm),
            "s": int(sc),
            "w": int(wc),
            "d": int(dc),
            "a": (fd or "")[:10],
            "b": (ld or "")[:10],
            "c": list(consts)[:3] if consts else [],
            "t": list(titles)[:5] if titles else [],
        }
        if int(sc) >= 5 and nm in timelines:
            entry["y"] = timelines[nm]
        speakers_list.append(entry)

    index_payload = {
        "stats": {
            "speeches": int(st_row[0]),
            "docs": int(st_row[1]),
            "total_docs": int(total_all_docs),
            "words": int(st_row[2] or 0),
            "min_date": st_row[3],
            "max_date": st_row[4],
            "speakers": len(speakers_list),
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


if __name__ == "__main__":
    main()
