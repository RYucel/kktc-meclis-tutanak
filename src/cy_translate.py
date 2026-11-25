"""Translate 1960-1964 Republic of Cyprus (KC) Greek/mixed speech turns to Turkish via NVIDIA NIM.

Features:
- Splits long turns (>3,000 chars) on paragraph/line boundaries so output never truncates.
- Content-addressed JSONL chunk cache (`data/cy_translations/cache.jsonl`) for zero-waste resume.
- Small multi-chunk batches (<=5 chunks, <=2,400 chars) with strict `<<<ID>>>` tags and automatic
  binary-split fallback down to single-chunk translation when any chunk fails validation.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
from pathlib import Path

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_HOST = "integrate.api.nvidia.com"
API_PATH = "/v1/chat/completions"

_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_TAG_SPLIT_RE = re.compile(r"(?m)^\s*<<<(\d+)>>>\s*$")

SYSTEM_PROMPT = (
    "Sen 1960-1964 Kıbrıs Cumhuriyeti Temsilciler Meclisi resmî tutanak çevirmenisin. "
    "Verilen Yunanca (Katharevousa) meclis konuşmalarını anlam kaybı olmadan, tam, doğru ve doğal resmî meclis Türkçesine çevir. "
    "Eski OCR kaynaklı küçük Yunanca harf hatalarını bağlamdan düzelt. "
    "Metin içinde zaten Türkçe olan cümle veya bölümler varsa bunların kelimelerini değiştirmeden "
    "(ör. eski Türkçe 'müstacel', 'tasvip', 'müzakere' gibi ifadeleri bozmadan) aynen koru; yalnızca Yunanca kısımları Türkçeye çevir. "
    "Çıktıda hiçbir Yunanca kelime veya harf bırakma; kişi ve yer adlarını Türkçe yazılışlarıyla aktar "
    "(Πάφος = Baf, Λεμεσός = Leymosun/Limasol, Αμμόχωστος = Mağusa, Λάρναξ = Larnaka, Λευκωσία = Lefkoşa, Κυρήνεια = Girne).\n"
    "Terminoloji: Βουλὴ τῶν Ἀντιπροσώπων = Temsilciler Meclisi, Πρόεδρος = Başkan, "
    "Ἀντιπρόεδρος = Başkan Vekili, Γραμματεύς = Kâtip, Ἑλληνικὴ Κοινότης = Rum Cemaati, "
    "Τουρκικὴ Κοινότης = Türk Cemaati, Ἕλληνες Βουλευταί = Rum Milletvekilleri, "
    "Τοῦρκοι Βουλευταί = Türk Milletvekilleri, Νομοσχέδιον = Yasa Tasarısı, "
    "Πρότασις Νόμου = Yasa Önerisi, Ἡμερησία Διάταξις = Gündem, "
    "Ἐπιτροπὴ Ἐπιλογῆς = Seçim Komitesi, Κοινοβουλευτικὴ Ὁμάς = Meclis Grubu, "
    "ἐγκριθείς / ἐνεκρίθη = onaylanan / tasvip edilen, κατεπείγων = ivedi / müstacel."
)

MULTI_INSTRUCTION = (
    "Girdideki her <<<ID>>> satırı için çıktıda tam olarak aynı <<<ID>>> satırını yaz ve hemen altına yalnızca o konuşmanın Türkçe çevirisini yaz.\n"
    "Hiçbir <<<ID>>> bloğunu atlama veya birleştirme; açıklama, yorum veya dipnot ekleme."
)

SINGLE_INSTRUCTION = (
    "Yalnızca verilen Yunanca metnin tam ve doğru Türkçe çevirisini yaz; ön söz, açıklama veya ek etiket ekleme."
)

_lock = threading.Lock()
_pace_lock = threading.Lock()
_next_req_at = 0.0
_cooldown_until = 0.0
_min_interval = 60.0 / 32.0
_max_tokens = 4096
_http_timeout = 90
_tls = threading.local()
_conns: list[http.client.HTTPSConnection] = []


def configure_rate_limit(rpm: float) -> None:
    global _min_interval
    _min_interval = (60.0 / rpm) if rpm > 0 else 0.0


def _wait_pace_slot() -> None:
    global _next_req_at
    if _min_interval <= 0 and _cooldown_until <= 0:
        return
    with _pace_lock:
        now = time.monotonic()
        wait_until = max(now, _next_req_at, _cooldown_until)
        _next_req_at = wait_until + _min_interval
    delay = wait_until - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _trigger_cooldown(seconds: float) -> None:
    global _cooldown_until, _next_req_at
    with _pace_lock:
        target = time.monotonic() + seconds
        if target > _cooldown_until:
            _cooldown_until = target
        if target > _next_req_at:
            _next_req_at = target


def load_nvidia_api_key() -> str:
    """Load NVIDIA_API_KEY from environment or ~/.env or ./.env without printing it."""
    val = os.environ.get("NVIDIA_API_KEY", "").strip()
    if val:
        return val
    for env_path in (Path(".env"), Path.home() / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("NVIDIA_API_KEY="):
                cand = s.split("=", 1)[1].strip().strip('"').strip("'")
                if cand:
                    return cand
    raise SystemExit("NVIDIA_API_KEY bulunamadi (ortam degiskeni veya ~/.env icine ekleyin).")


def has_greek(text: str) -> bool:
    """Return True when text contains real Greek content rather than stray OCR characters in Turkish text."""
    s = text or ""
    g = len(_GREEK_RE.findall(s))
    if g == 0:
        return False
    alpha = sum(1 for c in s if c.isalpha()) or 1
    return g >= 30 or (g / alpha) >= 0.30


def chunk_hash(chunk: str) -> str:
    return hashlib.sha1(chunk.strip().encode("utf-8")).hexdigest()[:20]


def _pack_units(units: list[str], sep: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if len(u) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            if sep == "\n\n":
                out.extend(_pack_units([ln.strip() for ln in u.splitlines() if ln.strip()], "\n", max_chars))
            elif sep == "\n":
                out.extend(_pack_units(u.split(), " ", max_chars))
            else:
                for i in range(0, len(u), max_chars):
                    out.append(u[i : i + max_chars])
            continue
        cand = f"{cur}{sep}{u}" if cur else u
        if len(cand) <= max_chars:
            cur = cand
        else:
            out.append(cur)
            cur = u
    if cur:
        out.append(cur)
    return out


def split_greek_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split long Greek turns along paragraph/line boundaries so no chunk exceeds `max_chars`."""
    s = (text or "").strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    paras = [p.strip() for p in re.split(r"\n{2,}", s) if p.strip()]
    return _pack_units(paras, "\n\n", max_chars)


def load_cache(cache_path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h, tr = rec.get("h"), rec.get("tr")
                if h and tr:
                    cache[h] = tr
            except json.JSONDecodeError:
                continue
    return cache


def lookup_translation(text: str, cache: dict[str, str], max_chars: int = 3000) -> str | None:
    """Return full Turkish translation if text has Greek and every Greek chunk is cached, else None."""
    if not has_greek(text):
        return None
    chunks = split_greek_chunks(text, max_chars=max_chars)
    if not chunks:
        return None
    parts: list[str] = []
    for ch in chunks:
        if not has_greek(ch):
            parts.append(ch)
            continue
        tr = cache.get(chunk_hash(ch))
        if not tr:
            return None
        parts.append(tr)
    return "\n\n".join(parts)


def is_valid_turkish_translation(src: str, tr: str) -> bool:
    s = (tr or "").strip()
    if not s:
        return False
    if not has_greek(src):
        return True
    tr_greek = len(_GREEK_RE.findall(s))
    if tr_greek == 0:
        return True
    alpha = sum(1 for c in s if c.isalpha()) or 1
    return tr_greek < 15 and (tr_greek / alpha) < 0.05


def _get_conn() -> http.client.HTTPSConnection:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = http.client.HTTPSConnection(NVIDIA_HOST, timeout=_http_timeout)
        _tls.conn = conn
        with _lock:
            _conns.append(conn)
    return conn


def _reset_conn() -> None:
    conn = getattr(_tls, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _tls.conn = None


def _call_nim(api_key: str, model: str, instruction: str, user_text: str, retries: int = 7) -> str:
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": _max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{instruction}"},
            {"role": "user", "content": user_text},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            _wait_pace_slot()
            conn = _get_conn()
            conn.request("POST", API_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status == 200:
                data = json.loads(raw.decode("utf-8"))
                return (data["choices"][0]["message"].get("content") or "").strip()
            msg = raw.decode("utf-8", errors="replace")[:250]
            if resp.status in (401, 403):
                raise SystemExit(f"NVIDIA NIM auth error HTTP {resp.status}: {msg}")
            if resp.status == 402:
                raise SystemExit(f"NVIDIA NIM quota/billing HTTP 402: {msg}")
            last_err = RuntimeError(f"HTTP {resp.status}: {msg}")
            if resp.status == 429:
                wait = min(12.0 * (1.5**attempt) + random.random() * 3.0, 65.0)
                _trigger_cooldown(wait)
                time.sleep(wait)
                continue
            if resp.status >= 500:
                time.sleep(min(2.0 * (1.6**attempt) + random.random() * 1.5, 45.0))
                continue
            raise last_err
        except SystemExit:
            raise
        except Exception as exc:
            last_err = exc
            _reset_conn()
            if attempt + 1 < retries:
                time.sleep(min(2.5 * (1.6**attempt) + random.random(), 30.0))
    raise RuntimeError(f"NIM request failed after {retries} attempts: {last_err}")


def translate_batch(items: list[tuple[str, str]], api_key: str, model: str) -> dict[str, str]:
    """Translate list of (hash, greek_text) items, recursively binary-splitting on any alignment mismatch."""
    if not items:
        return {}
    if len(items) == 1:
        h, src = items[0]
        raw = _call_nim(api_key, model, SINGLE_INSTRUCTION, src)
        cleaned = re.sub(r"^\s*<<<0>>>\s*", "", raw).strip()
        if is_valid_turkish_translation(src, cleaned):
            return {h: cleaned}
        retry_inst = (
            SINGLE_INSTRUCTION
            + " UYARI: Çıktıda hiçbir Yunanca cümle veya Yunanca harf kalmamalıdır; metnin tamamını eksiksiz Türkçeye çevir."
        )
        raw = _call_nim(api_key, model, retry_inst, src)
        cleaned = re.sub(r"^\s*<<<0>>>\s*", "", raw).strip()
        if is_valid_turkish_translation(src, cleaned):
            return {h: cleaned}
        raise RuntimeError(f"Single-chunk translation invalid/untranslated after retry (hash={h})")

    user_msg = "\n\n".join(f"<<<{i}>>>\n{txt}" for i, (_h, txt) in enumerate(items))
    raw = _call_nim(api_key, model, MULTI_INSTRUCTION, user_msg)
    parts = _TAG_SPLIT_RE.split(raw)
    parsed: dict[int, str] = {}
    for pos in range(1, len(parts) - 1, 2):
        idx = int(parts[pos])
        parsed[idx] = parts[pos + 1].strip()

    if len(parsed) == len(items) and all(
        i in parsed and is_valid_turkish_translation(items[i][1], parsed[i]) for i in range(len(items))
    ):
        return {items[i][0]: parsed[i] for i in range(len(items))}

    # Binary split guarantees 1-to-1 alignment with zero merged or dropped turns
    mid = len(items) // 2
    out = translate_batch(items[:mid], api_key, model)
    out.update(translate_batch(items[mid:], api_key, model))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeches", default="data/cy_build/speeches.jsonl")
    ap.add_argument("--cache", default="data/cy_translations/cache.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=32.0, help="Global max requests/min across workers (0 = unlimited)")
    ap.add_argument("--max-batch-items", type=int, default=6)
    ap.add_argument("--max-batch-chars", type=int, default=3400)
    ap.add_argument("--vol", nargs="*", default=None, help="Optional volume letters filter, e.g. D A")
    ap.add_argument("--limit-docs", type=int, default=0, help="Only process first N matching documents (0 = all)")
    ap.add_argument("--budget", type=float, default=0, help="Stop cleanly after N seconds (cached chunks kept)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Completion token cap per request")
    ap.add_argument("--timeout", type=int, default=90, help="HTTP read timeout (s)")
    args = ap.parse_args()
    configure_rate_limit(args.rpm)
    global _max_tokens, _http_timeout
    _max_tokens, _http_timeout = args.max_tokens, args.timeout

    api_key = load_nvidia_api_key()
    sp_path = Path(args.speeches)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)

    vol_order = (
        {f"Kıbrıs Cumhuriyeti Temsilciler Meclisi/Synodos {v} -": idx for idx, v in enumerate(args.vol)}
        if args.vol
        else None
    )
    raw_speeches = [json.loads(ln) for ln in sp_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if vol_order is not None:
        def _vol_key(rec: dict) -> int | None:
            did = rec["doc_id"]
            for pref, idx in vol_order.items():
                if did.startswith(pref):
                    return idx
            return None
        matched_speeches = [(k, rec) for rec in raw_speeches if (k := _vol_key(rec)) is not None]
        matched_speeches.sort(key=lambda pair: pair[0])
        iter_speeches = [rec for _, rec in matched_speeches]
    else:
        iter_speeches = raw_speeches

    seen_docs: list[str] = []
    seen_doc_set: set[str] = set()
    needed: dict[str, str] = {}
    total_el_turns = 0

    for s in iter_speeches:
        did = s["doc_id"]
        if did not in seen_doc_set:
            if args.limit_docs and len(seen_docs) >= args.limit_docs:
                continue
            seen_doc_set.add(did)
            seen_docs.append(did)
        if not has_greek(s.get("text") or ""):
            continue
        total_el_turns += 1
        for ch in split_greek_chunks(s["text"]):
            if not has_greek(ch):
                continue
            h = chunk_hash(ch)
            if h not in cache and h not in needed:
                needed[h] = ch

    items_todo = list(needed.items())
    # Pack chunks into batches <= 5 items and <= 2,400 chars
    batches: list[list[tuple[str, str]]] = []
    cur_b: list[tuple[str, str]] = []
    cur_chars = 0
    for item in items_todo:
        txt_len = len(item[1])
        if cur_b and (len(cur_b) >= args.max_batch_items or cur_chars + txt_len > args.max_batch_chars):
            batches.append(cur_b)
            cur_b, cur_chars = [], 0
        cur_b.append(item)
        cur_chars += txt_len
    if cur_b:
        batches.append(cur_b)

    print(
        f"Docs matched: {len(seen_docs)}, Greek/mixed turns: {total_el_turns:,}, "
        f"cached chunks: {len(cache):,}, remaining unique chunks: {len(items_todo):,} in {len(batches):,} batches "
        f"(model={args.model}, workers={args.workers})",
        flush=True,
    )
    if not batches:
        return

    t0 = time.perf_counter()
    stop = threading.Event()
    stats = {"batches": 0, "chunks": 0, "errors": 0}

    def _work(batch: list[tuple[str, str]]) -> int:
        if stop.is_set():
            return 0
        if args.budget and (time.perf_counter() - t0) >= args.budget:
            stop.set()
            return 0
        res = translate_batch(batch, api_key, args.model)
        with _lock:
            with cache_path.open("a", encoding="utf-8") as cf_fh:
                for h, tr in res.items():
                    if h not in cache and tr:
                        cache[h] = tr
                        cf_fh.write(json.dumps({"h": h, "tr": tr}, ensure_ascii=False) + "\n")
            stats["batches"] += 1
            stats["chunks"] += len(res)
            if stats["batches"] % 20 == 0 or stats["batches"] == len(batches):
                dt = max(time.perf_counter() - t0, 0.1)
                rate = stats["chunks"] / dt
                eta = (len(items_todo) - stats["chunks"]) / max(rate, 0.01)
                print(
                    f"  [batches {stats['batches']:,}/{len(batches):,}] chunks {stats['chunks']:,}/{len(items_todo):,} "
                    f"({rate * 60:.0f} ch/min, elapsed {dt:.0f}s, eta {eta/60:.1f}m, errs={stats['errors']})",
                    flush=True,
                )
        return len(res)

    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work, b) for b in batches]
            for f in cf.as_completed(futs):
                try:
                    f.result()
                except Exception as exc:
                    with _lock:
                        stats["errors"] += 1
                    print(f"  [WARN] batch failed: {exc}", flush=True)
    finally:
        for c in _conns:
            try:
                c.close()
            except Exception:
                pass
        _conns.clear()

    print(
        f"Finished: +{stats['chunks']:,} chunks cached (total cache {len(cache):,}), errs={stats['errors']}, "
        f"elapsed {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
