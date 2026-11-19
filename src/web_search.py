"""Zero-dependency local web UI for speaker-based search over KKTC Meclisi minutes."""

import argparse
import html
import json
import re
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


INDEX_HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KKTC Cumhuriyet Meclisi Tutanak Arama</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#f1f5f9; --acc:#38bdf8; --border:#334155; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:20px 24px; border-bottom:1px solid var(--border); background:#0b1220; }
  header h1 { margin:0 0 6px; font-size:20px; }
  header .sub { color:var(--muted); font-size:13px; }
  main { max-width:1100px; margin:20px auto; padding:0 16px 40px; }
  .search-box { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; display:grid; grid-template-columns:1.2fr 1.6fr 0.7fr 0.6fr auto; gap:10px; align-items:end; }
  @media (max-width:850px){ .search-box { grid-template-columns:1fr; } }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
  input, select { width:100%; padding:10px 12px; border-radius:8px; border:1px solid var(--border); background:#0f172a; color:var(--text); font-size:14px; }
  button { padding:10px 18px; border-radius:8px; border:0; background:var(--acc); color:#0b1220; font-weight:700; cursor:pointer; font-size:14px; }
  button:hover { opacity:0.9; }
  .meta-bar { display:flex; justify-content:space-between; align-items:center; margin:14px 2px; color:var(--muted); font-size:13px; flex-wrap:wrap; gap:8px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:10px; }
  .card-head { display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
  .speaker { font-weight:700; color:var(--acc); font-size:15px; cursor:pointer; }
  .badge { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; background:#0f172a; border:1px solid var(--border); color:#cbd5e1; margin-left:6px; }
  .snippet { line-height:1.6; font-size:14px; color:#e2e8f0; white-space:pre-wrap; }
  mark { background:#fde047; color:#0f172a; padding:0 2px; border-radius:3px; }
  .src-link { color:var(--muted); font-size:12px; text-decoration:none; }
  .src-link:hover { color:var(--acc); text-decoration:underline; }
  .speaker-wrap { position:relative; }
  .ac-list { position:absolute; top:100%; left:0; right:0; background:#0b1220; border:1px solid var(--border); border-radius:8px; max-height:240px; overflow:auto; z-index:20; display:none; }
  .ac-item { padding:8px 10px; cursor:pointer; font-size:13px; display:flex; justify-content:space-between; border-bottom:1px solid #1e293b; }
  .ac-item:hover { background:#1e293b; }
</style>
</head>
<body>
<header>
  <h1>KKTC Cumhuriyet Meclisi Tutanak Arama (1960–Günümüz)</h1>
  <div class="sub" id="stats">Yükleniyor...</div>
</header>
<main>
  <form class="search-box" id="form">
    <div class="speaker-wrap">
      <label for="speaker">Konuşmacı</label>
      <input id="speaker" placeholder="Örn: Tufan Erhürman, Derviş Eroğlu..." autocomplete="off">
      <div class="ac-list" id="ac"></div>
    </div>
    <div>
      <label for="q">Konuşma Metni (FTS5)</label>
      <input id="q" placeholder="Örn: anayasa, elektrik, su projesi, bütçe...">
    </div>
    <div>
      <label for="meclis">Meclis</label>
      <select id="meclis">
        <option value="">Tümü</option>
        <option value="KKTC">KKTC</option>
        <option value="KTFD">KTFD</option>
        <option value="KTY">KTY</option>
        <option value="KKTC_KM">KKTC Kurucu</option>
        <option value="KTFD_KM">KTFD Kurucu</option>
        <option value="KC">Kıbrıs Cumh.</option>
        <option value="TCM">Türk Cemaat</option>
      </select>
    </div>
    <div>
      <label for="from_year">Yıl (Baş.)</label>
      <input id="from_year" type="number" min="1960" max="2030" placeholder="1960">
    </div>
    <div>
      <button type="submit">Ara</button>
    </div>
  </form>
  <div class="meta-bar" id="meta"></div>
  <div id="results"></div>
</main>
<script>
const $ = id => document.getElementById(id);
fetch('/api/stats').then(r=>r.json()).then(s=>{
  $('stats').textContent = `${s.speeches.toLocaleString('tr-TR')} konuşma bloğu · ${s.speakers.toLocaleString('tr-TR')} konuşmacı · ${s.docs.toLocaleString('tr-TR')} tutanak belgesi (${s.min_date || '?'} – ${s.max_date || '?'})`;
});

let acTimer = null;
$('speaker').addEventListener('input', ()=>{
  clearTimeout(acTimer);
  const v = $('speaker').value.trim();
  if (v.length < 2) { $('ac').style.display='none'; return; }
  acTimer = setTimeout(async ()=>{
    const res = await fetch('/api/speakers?q=' + encodeURIComponent(v));
    const items = await res.json();
    if (!items.length) { $('ac').style.display='none'; return; }
    $('ac').innerHTML = items.map(it =>
      `<div class="ac-item" data-name="${it.speaker_name.replace(/"/g,'&quot;')}"><span>${it.speaker_name}</span><span style="color:#94a3b8">${it.speech_count} söz</span></div>`
    ).join('');
    $('ac').style.display='block';
  }, 150);
});
$('ac').addEventListener('click', e=>{
  const item = e.target.closest('.ac-item');
  if (!item) return;
  $('speaker').value = item.dataset.name;
  $('ac').style.display='none';
  runSearch();
});
document.addEventListener('click', e=>{ if (!e.target.closest('.speaker-wrap')) $('ac').style.display='none'; });

async function runSearch(offset=0){
  const params = new URLSearchParams({
    speaker: $('speaker').value.trim(),
    q: $('q').value.trim(),
    meclis: $('meclis').value,
    from_year: $('from_year').value.trim(),
    limit: '30',
    offset: String(offset),
  });
  $('meta').textContent = 'Aranıyor...';
  const r = await fetch('/api/search?' + params.toString());
  const data = await r.json();
  if (data.error) { $('meta').textContent = 'Hata: ' + data.error; return; }
  $('meta').innerHTML = `<span>${data.total.toLocaleString('tr-TR')} eşleşme (${data.ms} ms) — gösterilen: ${offset+1}–${Math.min(offset+data.items.length, data.total)}</span>` +
    `<span>${offset > 0 ? `<button type="button" onclick="runSearch(${Math.max(0, offset-30)})">← Önceki</button> ` : ''}${offset + 30 < data.total ? `<button type="button" onclick="runSearch(${offset+30})">Sonraki →</button>` : ''}</span>`;
  $('results').innerHTML = data.items.map(it => `
    <div class="card">
      <div class="card-head">
        <div>
          <span class="speaker" onclick="document.getElementById('speaker').value=this.textContent;runSearch();">${it.speaker_name}</span>
          ${it.speaker_title ? `<span class="badge">${it.speaker_title}</span>` : ''}
          ${it.constituency ? `<span class="badge">${it.constituency}</span>` : ''}
        </div>
        <div>
          <span class="badge">${it.tarih || 'Tarihsiz'}</span>
          <span class="badge">${it.meclis} ${it.donem ? 'D'+it.donem : ''} ${it.birlesim ? 'B'+it.birlesim : ''}</span>
        </div>
      </div>
      <div class="snippet">${it.snippet}</div>
      <div style="margin-top:8px"><a class="src-link" href="${it.source_url}" target="_blank" rel="noopener">Kaynak: ${it.source_file}</a></div>
    </div>
  `).join('') || '<div class="card">Sonuç bulunamadı.</div>';
}

$('form').addEventListener('submit', e=>{ e.preventDefault(); runSearch(0); });
runSearch(0);
</script>
</body>
</html>
"""


class SearchHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        conn = get_conn()
        try:
            if parsed.path == "/api/stats":
                row = conn.execute(
                    "SELECT COUNT(*) AS sp, COUNT(DISTINCT doc_id) AS docs, MIN(tarih) AS mn, MAX(tarih) AS mx FROM speeches"
                ).fetchone()
                spk_cnt = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
                self._send_json(
                    {
                        "speeches": row["sp"],
                        "docs": row["docs"],
                        "speakers": spk_cnt,
                        "min_date": row["mn"],
                        "max_date": row["mx"],
                    }
                )
                return

            if parsed.path == "/api/speakers":
                q = (qs.get("q", [""])[0]).strip()
                fq = fold_ascii(q)
                rows = conn.execute(
                    """
                    SELECT speaker_name, speech_count, word_count, first_date, last_date
                    FROM speakers
                    WHERE speaker_folded LIKE ?
                    ORDER BY speech_count DESC
                    LIMIT 20
                    """,
                    (f"%{fq}%",),
                ).fetchall()
                self._send_json([dict(r) for r in rows])
                return

            if parsed.path == "/api/search":
                import time

                t0 = time.perf_counter()
                speaker = (qs.get("speaker", [""])[0]).strip()
                q = (qs.get("q", [""])[0]).strip()
                meclis = (qs.get("meclis", [""])[0]).strip().upper()
                from_year = (qs.get("from_year", [""])[0]).strip()
                limit = min(100, max(1, int(qs.get("limit", ["30"])[0])))
                offset = max(0, int(qs.get("offset", ["0"])[0]))

                where = []
                params = []

                if speaker:
                    fq = fold_ascii(speaker)
                    spks = [
                        r[0]
                        for r in conn.execute(
                            "SELECT speaker_name FROM speakers WHERE speaker_folded LIKE ? LIMIT 50",
                            (f"%{fq}%",),
                        ).fetchall()
                    ]
                    if not spks:
                        self._send_json({"total": 0, "ms": 0, "items": []})
                        return
                    where.append("s.speaker_name IN (" + ",".join("?" for _ in spks) + ")")
                    params.extend(spks)

                if meclis:
                    where.append("s.meclis = ?")
                    params.append(meclis)
                if from_year and from_year.isdigit():
                    where.append("s.tarih >= ?")
                    params.append(f"{int(from_year):04d}-01-01")

                where_sql = (" AND " + " AND ".join(where)) if where else ""

                try:
                    if q:
                        total = conn.execute(
                            f"""
                            SELECT COUNT(*) FROM speeches_fts
                            JOIN speeches s ON speeches_fts.rowid = s.rowid
                            WHERE speeches_fts MATCH ? {where_sql}
                            """,
                            [q] + params,
                        ).fetchone()[0]
                        rows = conn.execute(
                            f"""
                            SELECT s.speaker_name, s.speaker_title, s.constituency, s.tarih,
                                   s.meclis, s.donem, s.birlesim, s.source_file, s.source_url,
                                   snippet(speeches_fts, 3, '<<<M>>>', '<<</M>>>', '...', 45) AS snip
                            FROM speeches_fts
                            JOIN speeches s ON speeches_fts.rowid = s.rowid
                            WHERE speeches_fts MATCH ? {where_sql}
                            ORDER BY s.tarih DESC, s.turn_index ASC
                            LIMIT ? OFFSET ?
                            """,
                            [q] + params + [limit, offset],
                        ).fetchall()
                    else:
                        total = conn.execute(
                            f"SELECT COUNT(*) FROM speeches s WHERE 1=1 {where_sql}",
                            params,
                        ).fetchone()[0]
                        rows = conn.execute(
                            f"""
                            SELECT s.speaker_name, s.speaker_title, s.constituency, s.tarih,
                                   s.meclis, s.donem, s.birlesim, s.source_file, s.source_url,
                                   substr(s.text, 1, 420) AS snip
                            FROM speeches s
                            WHERE 1=1 {where_sql}
                            ORDER BY s.tarih DESC, s.turn_index ASC
                            LIMIT ? OFFSET ?
                            """,
                            params + [limit, offset],
                        ).fetchall()
                except sqlite3.OperationalError as e:
                    self._send_json({"error": str(e)}, code=400)
                    return

                items = []
                for r in rows:
                    snip_esc = html.escape(r["snip"] or "")
                    snip_esc = snip_esc.replace("&lt;&lt;&lt;M&gt;&gt;&gt;", "<mark>").replace(
                        "&lt;&lt;&lt;/M&gt;&gt;&gt;", "</mark>"
                    )
                    items.append(
                        {
                            "speaker_name": r["speaker_name"],
                            "speaker_title": r["speaker_title"],
                            "constituency": r["constituency"],
                            "tarih": r["tarih"],
                            "meclis": r["meclis"],
                            "donem": r["donem"],
                            "birlesim": r["birlesim"],
                            "source_file": r["source_file"],
                            "source_url": r["source_url"],
                            "snippet": snip_esc,
                        }
                    )

                ms = round((time.perf_counter() - t0) * 1000, 1)
                self._send_json({"total": total, "ms": ms, "items": items})
                return

            self.send_error(404)
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found. Run src/build_dataset.py first.")
    srv = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    print(f"KKTC Meclis Tutanak Arama UI running at http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
