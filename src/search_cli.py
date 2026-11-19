"""Fast CLI for speaker-based and full-text search over KKTC Meclisi minutes (SQLite FTS5)."""

import argparse
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/search.db")

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", s)).strip()


def cmd_find_speakers(conn: sqlite3.Connection, query: str, limit: int):
    fq = fold_ascii(query)
    rows = conn.execute(
        """
        SELECT speaker_name, speech_count, word_count, doc_count, first_date, last_date, meclisler, constituencies
        FROM speakers
        WHERE speaker_folded LIKE ?
        ORDER BY speech_count DESC
        LIMIT ?
        """,
        (f"%{fq}%", limit),
    ).fetchall()
    print(f"Found {len(rows)} matching speakers for '{query}':")
    for r in rows:
        print(
            f"  {r[0]:32s} | speeches={r[1]:5d} | words={r[2]:8d} | "
            f"{r[4] or '?'}..{r[5] or '?'} | {r[6]} | {r[7]}"
        )


def cmd_search(conn: sqlite3.Connection, args):
    where = []
    params = []
    use_fts = bool(args.query)

    if args.speaker:
        # Resolve matching canonical speaker names via folded substring
        fq = fold_ascii(args.speaker)
        spk_names = [
            r[0]
            for r in conn.execute(
                "SELECT speaker_name FROM speakers WHERE speaker_folded LIKE ?",
                (f"%{fq}%",),
            ).fetchall()
        ]
        if not spk_names:
            print(f"No speaker matching '{args.speaker}'.")
            return
        placeholders = ",".join("?" for _ in spk_names)
        where.append(f"s.speaker_name IN ({placeholders})")
        params.extend(spk_names)

    if args.meclis:
        where.append("s.meclis = ?")
        params.append(args.meclis.upper())
    if args.donem is not None:
        where.append("s.donem = ?")
        params.append(args.donem)
    if args.from_date:
        where.append("s.tarih >= ?")
        params.append(args.from_date)
    if args.to_date:
        where.append("s.tarih <= ?")
        params.append(args.to_date)

    where_sql = (" AND " + " AND ".join(where)) if where else ""

    if use_fts:
        sql = f"""
            SELECT s.tarih, s.meclis, s.donem, s.birlesim, s.speaker_name, s.speaker_title,
                   s.constituency, snippet(speeches_fts, 3, '[[', ']]', '...', 28) AS snip,
                   s.source_file
            FROM speeches_fts
            JOIN speeches s ON speeches_fts.rowid = s.rowid
            WHERE speeches_fts MATCH ? {where_sql}
            ORDER BY s.tarih DESC, s.turn_index ASC
            LIMIT ?
        """
        full_params = [args.query] + params + [args.limit]
    else:
        sql = f"""
            SELECT s.tarih, s.meclis, s.donem, s.birlesim, s.speaker_name, s.speaker_title,
                   s.constituency, substr(s.text, 1, 240) AS snip, s.source_file
            FROM speeches s
            WHERE 1=1 {where_sql}
            ORDER BY s.tarih DESC, s.turn_index ASC
            LIMIT ?
        """
        full_params = params + [args.limit]

    rows = conn.execute(sql, full_params).fetchall()
    print(f"Results ({len(rows)} shown):")
    for idx, r in enumerate(rows, start=1):
        tarih, meclis, donem, birlesim, spk, title, const, snip, src = r
        meta = f"{tarih or '?'} | {meclis} D{donem or '?'} B{birlesim or '?'} | {src}"
        who = spk + (f" ({title})" if title else "") + (f" [{const}]" if const else "")
        clean_snip = re.sub(r"\s+", " ", snip or "").strip()
        print(f"\n[{idx}] {who}\n    {meta}\n    {clean_snip}")


def main():
    ap = argparse.ArgumentParser(description="Search KKTC Meclisi speeches by speaker and/or keywords.")
    ap.add_argument("--speaker", "-s", help="Speaker name (substring, Turkish diacritic-insensitive)")
    ap.add_argument("--query", "-q", help="FTS5 keyword query (e.g. 'kibris elektrik' or '\"su projesi\"')")
    ap.add_argument("--list-speakers", help="List matching speakers and counts")
    ap.add_argument("--meclis", help="Filter by meclis code (KKTC, KTFD, KTY, ...)")
    ap.add_argument("--donem", type=int, help="Filter by donem number")
    ap.add_argument("--from-date", help="Start date YYYY-MM-DD")
    ap.add_argument("--to-date", help="End date YYYY-MM-DD")
    ap.add_argument("--limit", "-n", type=int, default=15)
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found. Run src/build_dataset.py first.")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        if args.list_speakers:
            cmd_find_speakers(conn, args.list_speakers, args.limit)
        else:
            cmd_search(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
