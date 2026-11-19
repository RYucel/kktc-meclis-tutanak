"""FastAPI + DuckDB backend for KKTC Cumhuriyet Meclisi speaker-based parliamentary archive search."""

import csv
import html
import io
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download

DATASET_REPO = os.environ.get("KKTC_DATASET_REPO", "stevevaius/KKTC_Meclis_Tutanak")
LOCAL_DATA_DIR = Path(os.environ.get("KKTC_LOCAL_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data/hf_dataset/data")))
STATIC_DIR = Path(__file__).resolve().parent / "static"

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)
_TR_LOWER = str.maketrans({"İ": "i", "I": "ı", "Â": "â", "Î": "î", "Û": "û"})


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


def tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER).lower().replace("\u0307", "")


class ArchiveEngine:
    def __init__(self):
        self.con: duckdb.DuckDBPyConnection | None = None
        self.ready: bool = False
        self.status: str = "Başlatılıyor..."
        self.error: str | None = None
        self.stats: dict[str, Any] = {}
        self.load_seconds: float = 0.0

    def _resolve_parquet_paths(self) -> tuple[str, str]:
        local_sp = LOCAL_DATA_DIR / "speeches.parquet"
        local_spk = LOCAL_DATA_DIR / "speakers.parquet"
        if local_sp.exists() and local_spk.exists():
            return str(local_sp), str(local_spk)

        self.status = "Hugging Face Hub'dan tutanak arşivi indiriliyor (~241 MB)..."
        sp_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="data/speeches.parquet",
            repo_type="dataset",
        )
        spk_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="data/speakers.parquet",
            repo_type="dataset",
        )
        return sp_path, spk_path

    def load(self):
        t0 = time.perf_counter()
        try:
            sp_path, spk_path = self._resolve_parquet_paths()
            self.status = "1,27 milyon konuşma DuckDB belleğine indeksleniyor..."

            con = duckdb.connect(":memory:")
            threads = max(2, min(4, os.cpu_count() or 2))
            con.execute(f"SET threads={threads};")

            # Load sorted by speaker_name, tarih DESC for ultra-fast zone-map pruning
            con.execute(
                """
                CREATE TABLE speeches AS
                SELECT speech_id, doc_id, meclis, donem, yasama_yili, birlesim, oturum,
                       tarih, turn_index, speaker_name, speaker_title, constituency,
                       modifiers, speaker_role, text, word_count, text_source,
                       source_file, source_url
                FROM read_parquet(?)
                ORDER BY speaker_name, tarih DESC, turn_index ASC
                """,
                [sp_path],
            )

            con.execute(
                """
                CREATE TABLE speakers AS
                SELECT speaker_name, speech_count, word_count, doc_count,
                       first_date, last_date, meclisler, donemler, titles, constituencies,
                       lower(
                         replace(replace(replace(replace(replace(replace(replace(replace(replace(
                           speaker_name,
                           'İ','i'),'I','i'),'ı','i'),'Ç','c'),'ç','c'),'Ğ','g'),'ğ','g'),'Ö','o'),'ö','o')
                       ) AS _f1
                FROM read_parquet(?)
                """,
                [spk_path],
            )
            con.execute(
                """
                ALTER TABLE speakers ADD COLUMN speaker_folded VARCHAR;
                UPDATE speakers SET speaker_folded = replace(replace(replace(replace(replace(replace(replace(_f1,
                    'Ş','s'),'ş','s'),'Ü','u'),'ü','u'),'â','a'),'î','i'),'û','u');
                """
            )

            row = con.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT doc_id), SUM(word_count), MIN(tarih), MAX(tarih)
                FROM speeches
                """
            ).fetchone()
            spk_cnt = con.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
            meclis_rows = con.execute(
                "SELECT meclis, COUNT(*) AS c FROM speeches GROUP BY 1 ORDER BY c DESC"
            ).fetchall()

            self.stats = {
                "speeches": int(row[0]),
                "docs": int(row[1]),
                "words": int(row[2] or 0),
                "min_date": row[3],
                "max_date": row[4],
                "speakers": int(spk_cnt),
                "by_meclis": [{"meclis": m, "count": int(c)} for m, c in meclis_rows],
                "dataset_repo": DATASET_REPO,
            }
            self.con = con
            self.load_seconds = round(time.perf_counter() - t0, 2)
            self.status = "Hazır"
            self.ready = True
            print(f"[ArchiveEngine] Ready in {self.load_seconds}s: {self.stats['speeches']:,} speeches", flush=True)
        except Exception as e:
            self.error = str(e)
            self.status = f"Hata: {e}"
            print(f"[ArchiveEngine] FAILED: {e}", flush=True)

    def cursor(self) -> duckdb.DuckDBPyConnection:
        if not self.ready or self.con is None:
            raise HTTPException(status_code=503, detail=self.status)
        return self.con.cursor()


engine = ArchiveEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    th = threading.Thread(target=engine.load, daemon=True)
    th.start()
    yield


app = FastAPI(title="KKTC Cumhuriyet Meclisi Tutanak Arama", lifespan=lifespan)


def parse_query_terms(q: str) -> list[str]:
    """Split search query into quoted phrases and single terms (max 6 terms)."""
    q = q.strip()
    if not q:
        return []
    tokens = re.findall(r'"([^"]+)"|(\S+)', q)
    terms = [a or b for a, b in tokens if (a or b).strip()]
    return [t.strip() for t in terms if len(t.strip()) >= 2][:6]


def build_keyword_clause(terms: list[str]) -> tuple[str, list[str]]:
    clauses = []
    params: list[str] = []
    for term in terms:
        t_tr = tr_lower(term)
        # Also match ASCII i/ı variant so both 'kibris'/'kıbrıs' and 'KIBRIS' match cleanly
        t_i = t_tr.replace("ı", "i")
        t_dotless = t_tr.replace("i", "ı")
        variants = sorted({t_tr, t_i, t_dotless})
        sub = " OR ".join("contains(lower(text), ?)" for _ in variants)
        clauses.append(f"({sub})")
        params.extend(variants)
    return " AND ".join(clauses), params


def highlight_snippet(text: str, terms: list[str], max_len: int = 440) -> str:
    if not text:
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if not terms:
        out = clean[:max_len] + ("..." if len(clean) > max_len else "")
        return html.escape(out)

    # Build Turkish-aware highlight regex
    alts = []
    for term in terms:
        pat_chars = []
        for ch in term:
            cl = tr_lower(ch)
            if cl in ("i", "ı"):
                pat_chars.append("[iİıI]")
            elif cl == "c":
                pat_chars.append("[cCçÇ]")
            elif cl == "ç":
                pat_chars.append("[çÇcC]")
            elif cl == "g":
                pat_chars.append("[gGğĞ]")
            elif cl == "ğ":
                pat_chars.append("[ğĞgG]")
            elif cl == "o":
                pat_chars.append("[oOöÖ]")
            elif cl == "ö":
                pat_chars.append("[öÖoO]")
            elif cl == "s":
                pat_chars.append("[sSşŞ]")
            elif cl == "ş":
                pat_chars.append("[şŞsS]")
            elif cl == "u":
                pat_chars.append("[uUüÜ]")
            elif cl == "ü":
                pat_chars.append("[üÜuU]")
            else:
                pat_chars.append(re.escape(ch))
        alts.append("".join(pat_chars))
    hl_re = re.compile("(" + "|".join(alts) + ")", re.I)

    m = hl_re.search(clean)
    if m and len(clean) > max_len:
        start = max(0, m.start() - 130)
        end = min(len(clean), start + max_len)
        # Snap to word boundaries
        if start > 0:
            sp = clean.find(" ", start)
            if 0 < sp - start < 20:
                start = sp + 1
        if end < len(clean):
            sp = clean.rfind(" ", start, end)
            if sp > start + max_len // 2:
                end = sp
        window = ("..." if start > 0 else "") + clean[start:end] + ("..." if end < len(clean) else "")
    else:
        window = clean[:max_len] + ("..." if len(clean) > max_len else "")

    parts = hl_re.split(window)
    out = []
    for i, part in enumerate(parts):
        esc = html.escape(part)
        if i % 2 == 1:
            out.append(f"<mark>{esc}</mark>")
        else:
            out.append(esc)
    return "".join(out)


def resolve_speaker_names(cur: duckdb.DuckDBPyConnection, speaker_query: str) -> list[str]:
    sq = speaker_query.strip()
    if not sq:
        return []
    # Check exact match first
    exact = cur.execute("SELECT speaker_name FROM speakers WHERE speaker_name = ? LIMIT 1", [sq]).fetchone()
    if exact:
        return [exact[0]]
    fq = fold_ascii(sq)
    rows = cur.execute(
        """
        SELECT speaker_name FROM speakers
        WHERE speaker_folded LIKE ?
        ORDER BY speech_count DESC
        LIMIT 30
        """,
        [f"%{fq}%"],
    ).fetchall()
    return [r[0] for r in rows]


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def api_health():
    return {
        "ready": engine.ready,
        "status": engine.status,
        "error": engine.error,
        "load_seconds": engine.load_seconds,
    }


@app.get("/api/stats")
def api_stats():
    if not engine.ready:
        return JSONResponse({"ready": False, "status": engine.status}, status_code=200)
    return {"ready": True, **engine.stats}


@app.get("/api/speakers")
def api_speakers(q: str = Query("", max_length=80), limit: int = Query(18, ge=1, le=50)):
    cur = engine.cursor()
    try:
        fq = fold_ascii(q)
        if fq:
            rows = cur.execute(
                """
                SELECT speaker_name, speech_count, word_count, doc_count, first_date, last_date, constituencies
                FROM speakers
                WHERE speaker_folded LIKE ?
                ORDER BY speech_count DESC
                LIMIT ?
                """,
                [f"%{fq}%", limit],
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT speaker_name, speech_count, word_count, doc_count, first_date, last_date, constituencies
                FROM speakers
                WHERE speaker_name NOT IN ('Meclis Başkanı', 'Katip', 'Geçici Başkan', 'Geçici Katip', 'Sesler')
                ORDER BY speech_count DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        return [
            {
                "speaker_name": r[0],
                "speech_count": int(r[1]),
                "word_count": int(r[2]),
                "doc_count": int(r[3]),
                "first_date": r[4],
                "last_date": r[5],
                "constituencies": list(r[6]) if r[6] else [],
            }
            for r in rows
        ]
    finally:
        cur.close()


@app.get("/api/speaker_profile")
def api_speaker_profile(name: str = Query(..., min_length=2, max_length=100)):
    cur = engine.cursor()
    try:
        matched = resolve_speaker_names(cur, name)
        if not matched:
            raise HTTPException(status_code=404, detail="Konuşmacı bulunamadı.")
        canonical = matched[0]
        row = cur.execute(
            """
            SELECT speaker_name, speech_count, word_count, doc_count,
                   first_date, last_date, meclisler, donemler, titles, constituencies
            FROM speakers
            WHERE speaker_name = ?
            """,
            [canonical],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Konuşmacı bulunamadı.")

        timeline = cur.execute(
            """
            SELECT substr(tarih, 1, 4) AS yr, COUNT(*) AS cnt, SUM(word_count) AS words
            FROM speeches
            WHERE speaker_name = ? AND tarih IS NOT NULL AND length(tarih) >= 4
            GROUP BY 1
            ORDER BY 1 ASC
            """,
            [canonical],
        ).fetchall()

        return {
            "speaker_name": row[0],
            "speech_count": int(row[1]),
            "word_count": int(row[2]),
            "doc_count": int(row[3]),
            "first_date": row[4],
            "last_date": row[5],
            "meclisler": list(row[6]) if row[6] else [],
            "donemler": [int(x) for x in (row[7] or [])],
            "titles": list(row[8])[:8] if row[8] else [],
            "constituencies": list(row[9]) if row[9] else [],
            "timeline": [{"year": r[0], "speeches": int(r[1]), "words": int(r[2] or 0)} for r in timeline],
        }
    finally:
        cur.close()


def _build_where(
    cur: duckdb.DuckDBPyConnection,
    speaker: str,
    q: str,
    meclis: str,
    donem: int | None,
    from_year: int | None,
    to_year: int | None,
    role: str,
) -> tuple[str, list[Any], list[str], list[str]]:
    where: list[str] = []
    params: list[Any] = []
    resolved_speakers: list[str] = []

    if speaker.strip():
        resolved_speakers = resolve_speaker_names(cur, speaker)
        if not resolved_speakers:
            return "1=0", [], [], []
        if len(resolved_speakers) == 1:
            where.append("speaker_name = ?")
            params.append(resolved_speakers[0])
        else:
            ph = ",".join("?" for _ in resolved_speakers)
            where.append(f"speaker_name IN ({ph})")
            params.extend(resolved_speakers)

    if meclis.strip():
        where.append("meclis = ?")
        params.append(meclis.strip().upper())

    if donem is not None:
        where.append("donem = ?")
        params.append(donem)

    if from_year is not None:
        where.append("tarih >= ?")
        params.append(f"{from_year:04d}-01-01")

    if to_year is not None:
        where.append("tarih <= ?")
        params.append(f"{to_year:04d}-12-31")

    if role.strip():
        where.append("speaker_role = ?")
        params.append(role.strip())

    terms = parse_query_terms(q)
    if terms:
        kw_sql, kw_params = build_keyword_clause(terms)
        where.append(kw_sql)
        params.extend(kw_params)

    where_sql = " AND ".join(where) if where else "1=1"
    return where_sql, params, terms, resolved_speakers


@app.get("/api/search")
def api_search(
    speaker: str = Query("", max_length=100),
    q: str = Query("", max_length=160),
    meclis: str = Query("", max_length=20),
    donem: int | None = Query(None, ge=1, le=20),
    from_year: int | None = Query(None, ge=1950, le=2030),
    to_year: int | None = Query(None, ge=1950, le=2030),
    role: str = Query("", max_length=30),
    sort: str = Query("newest", pattern="^(newest|oldest|longest)$"),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0, le=50000),
):
    t0 = time.perf_counter()
    cur = engine.cursor()
    try:
        where_sql, params, terms, resolved_speakers = _build_where(
            cur, speaker, q, meclis, donem, from_year, to_year, role
        )
        if where_sql == "1=0":
            return {"total": 0, "ms": 0.0, "resolved_speakers": [], "items": []}

        order_sql = {
            "newest": "tarih DESC NULLS LAST, turn_index ASC",
            "oldest": "tarih ASC NULLS LAST, turn_index ASC",
            "longest": "word_count DESC, tarih DESC NULLS LAST",
        }[sort]

        total = int(cur.execute(f"SELECT COUNT(*) FROM speeches WHERE {where_sql}", params).fetchone()[0])

        rows = cur.execute(
            f"""
            SELECT speech_id, doc_id, meclis, donem, birlesim, oturum, tarih, turn_index,
                   speaker_name, speaker_title, constituency, modifiers, speaker_role,
                   text, word_count, text_source, source_file, source_url
            FROM speeches
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        items = []
        for r in rows:
            items.append(
                {
                    "speech_id": r[0],
                    "doc_id": r[1],
                    "meclis": r[2],
                    "donem": r[3],
                    "birlesim": r[4],
                    "oturum": r[5],
                    "tarih": r[6],
                    "turn_index": r[7],
                    "speaker_name": r[8],
                    "speaker_title": r[9],
                    "constituency": r[10],
                    "modifiers": r[11],
                    "speaker_role": r[12],
                    "snippet_html": highlight_snippet(r[13], terms),
                    "word_count": int(r[14] or 0),
                    "text_source": r[15],
                    "source_file": r[16],
                    "source_url": r[17],
                }
            )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "total": total,
            "ms": ms,
            "resolved_speakers": resolved_speakers[:10],
            "items": items,
        }
    finally:
        cur.close()


@app.get("/api/context")
def api_context(
    doc_id: str = Query(..., min_length=3, max_length=250),
    turn_index: int = Query(..., ge=0),
    window: int = Query(3, ge=0, le=8),
):
    cur = engine.cursor()
    try:
        rows = cur.execute(
            """
            SELECT speech_id, turn_index, oturum, tarih, meclis, donem, birlesim,
                   speaker_name, speaker_title, constituency, modifiers, speaker_role,
                   text, word_count, source_file, source_url
            FROM speeches
            WHERE doc_id = ? AND turn_index BETWEEN ? AND ?
            ORDER BY turn_index ASC
            """,
            [doc_id, max(0, turn_index - window), turn_index + window],
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Konuşma bağlamı bulunamadı.")
        return {
            "doc_id": doc_id,
            "focus_turn": turn_index,
            "source_file": rows[0][14],
            "source_url": rows[0][15],
            "tarih": rows[0][3],
            "meclis": rows[0][4],
            "donem": rows[0][5],
            "birlesim": rows[0][6],
            "turns": [
                {
                    "speech_id": r[0],
                    "turn_index": int(r[1]),
                    "oturum": int(r[2] or 0),
                    "speaker_name": r[7],
                    "speaker_title": r[8],
                    "constituency": r[9],
                    "modifiers": r[10],
                    "speaker_role": r[11],
                    "text": r[12],
                    "word_count": int(r[13] or 0),
                    "is_focus": int(r[1]) == turn_index,
                }
                for r in rows
            ],
        }
    finally:
        cur.close()


@app.get("/api/export")
def api_export(
    speaker: str = Query("", max_length=100),
    q: str = Query("", max_length=160),
    meclis: str = Query("", max_length=20),
    donem: int | None = Query(None, ge=1, le=20),
    from_year: int | None = Query(None, ge=1950, le=2030),
    to_year: int | None = Query(None, ge=1950, le=2030),
    limit: int = Query(500, ge=1, le=2000),
):
    cur = engine.cursor()
    try:
        where_sql, params, _, _ = _build_where(cur, speaker, q, meclis, donem, from_year, to_year, "")
        if where_sql == "1=0":
            rows = []
        else:
            rows = cur.execute(
                f"""
                SELECT tarih, meclis, donem, birlesim, oturum, turn_index,
                       speaker_name, speaker_title, constituency, word_count, text, source_url
                FROM speeches
                WHERE {where_sql}
                ORDER BY tarih DESC NULLS LAST, turn_index ASC
                LIMIT ?
                """,
                params + [limit],
            ).fetchall()

        buf = io.StringIO()
        buf.write("\ufeff")  # UTF-8 BOM for clean Turkish Excel opening
        writer = csv.writer(buf)
        writer.writerow(
            ["tarih", "meclis", "donem", "birlesim", "oturum", "turn_index", "konusmaci", "unvan", "bolge", "kelime_sayisi", "metin", "kaynak_url"]
        )
        for r in rows:
            writer.writerow(r)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="kktc_tutanak_sonuclar.csv"'},
        )
    finally:
        cur.close()
