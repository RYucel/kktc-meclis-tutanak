"""Re-read scanned PDF pages with a vision-language model (Gemini) to replace noisy OCR text layers.

The model sees the page image, so it transcribes what is printed instead of guessing from garbled text.
Every page is cached as data/reocr/<doc-hash>/pNNNN.txt (+ usage in usage.jsonl), so runs are resumable and
nothing is paid for twice. reprocess_docs.py --reocr-dir uses a document's pages only when ALL are cached.

Pilot (a few pages from each listed document, with a before/after comparison):
  set GEMINI_API_KEY=...
  python src/reocr_vlm.py --docs data/ocr_quality/noisy_docs.txt --limit-docs 20 --pages-per-doc 3 --pilot
Full run:
  python src/reocr_vlm.py --docs data/ocr_quality/noisy_docs.txt --workers 8
Pages are cached per model under data/reocr/<model>/, so two models can be compared on the same pages.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import html as html_lib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROMPT = (
    "Bu görüntü, Kıbrıs Türk meclis tutanaklarından taranmış tek bir sayfadır. "
    "Sayfadaki metni basıldığı gibi, harfi harfine Türkçe olarak yaz.\n"
    "Kurallar:\n"
    "- Hiçbir kelimeyi düzeltme, sadeleştirme, özetleme ya da yorum ekleme. Eski yazım ve imlayı koru.\n"
    "- Satır düzenini koru: her basılı satır ayrı satır olsun; satır sonundaki tireyle bölünmüş kelimeleri olduğu gibi bırak.\n"
    "- Konuşmacı başlıklarını (ör. 'BAŞKAN -', 'HÜSEYİN ANGOLEMLİ (Lefkoşa) -') aynen yaz.\n"
    "- Sayfa numaralarını ve başlıkları da yaz.\n"
    "- Okuyamadığın yerlere [okunamadı] yaz; tahmin etme.\n"
    "- Sadece sayfa metnini ver; açıklama, markdown veya kod bloğu ekleme."
)
PROMPT_MOGAN = "Sayfayı markdown olarak yaz."
_lock = threading.Lock()
DEFAULT_MODEL = "gemini-3.8-flash"


def clean_mogan_markdown(md: str) -> str:
    """Convert MoganOCR Markdown/HTML output to clean plain text compatible with parse_speakers.py."""
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    # Remove <figure/> placeholders and unwrap <header>/<footer> tags
    s = re.sub(r"<figure\b[^>]*/?>.*?</figure>", "", s, flags=re.S | re.I)
    s = re.sub(r"<figure\b[^>]*/>", "", s, flags=re.I)
    s = re.sub(r"</?(?:header|footer)>", "", s, flags=re.I)

    # Flatten HTML tables into pipe-separated rows (matching extract_docx)
    def _table_to_lines(m: re.Match) -> str:
        rows = []
        for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", m.group(0), flags=re.S | re.I):
            cells = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_lib.unescape(c))).strip()
                for c in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    s = re.sub(r"<table\b[^>]*>.*?</table>", _table_to_lines, s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = html_lib.unescape(s)

    # Strip Markdown heading prefixes and bold/italic wrappers so speaker headers stay plain text
    s = re.sub(r"(?m)^#{1,6}\s+", "", s)
    s = re.sub(r"\*\*([^\n*]+?)\*\*", r"\1", s)
    s = re.sub(r"__([^\n_]+?)__", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def transcribe_openai(base_url: str, model: str, png: bytes, prompt: str, clean_md: bool = False, retries: int = 4) -> tuple[str, dict]:
    """Call a local OpenAI-compatible VLM server (llama-server, vLLM, Ollama)."""
    url = base_url.rstrip("/") + "/chat/completions"
    b64 = base64.b64encode(png).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[-1].strip()
            if clean_md:
                text = clean_mogan_markdown(text)
            u = data.get("usage") or {}
            usage = {
                "in": int(u.get("prompt_tokens", 0) or 0),
                "out": int(u.get("completion_tokens", 0) or 0),
                "think": 0,
            }
            return text, usage
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise SystemExit(f"Local VLM HTTP {e.code}: {err_body}")
            if attempt + 1 >= retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
        except Exception:
            if attempt + 1 >= retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("transcribe_openai exhausted retries")
# USD per 1M tokens (input, output), standard tier, from ai.google.dev pricing (Nov 2026); batch is half
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def doc_key(doc_id: str) -> str:
    return hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:16]


def page_cache(out: Path, doc_id: str, pno: int) -> Path:
    return out / doc_key(doc_id) / f"p{pno + 1:04d}.txt"


def cached_pages(out: Path, doc_id: str, page_count: int) -> list[str] | None:
    """All pages of a document if every page has been re-read, else None."""
    paths = [page_cache(out, doc_id, i) for i in range(page_count)]
    if not all(p.exists() for p in paths):
        return None
    return [p.read_text(encoding="utf-8") for p in paths]


def render_png(page, dpi: int) -> bytes:
    return page.get_pixmap(dpi=dpi, colorspace="gray").tobytes("png")


def make_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (Google AI Studio key).")
    return genai.Client(api_key=key)


_think_choice: dict[str, int] = {}


def _thinking_configs(types):
    """Transcription needs no reasoning: try the lowest thinking setting the model accepts, then none."""
    out = []
    for kw in ({"thinking_level": "minimal"}, {"thinking_level": "low"}, {"thinking_budget": 0}):
        try:
            out.append(types.ThinkingConfig(**kw))
        except Exception:  # older SDKs lack thinking_level
            pass
    out.append(None)
    return out


def transcribe(client, model: str, png: bytes, retries: int = 8) -> tuple[str, dict]:
    import random

    from google.genai import types

    thinks = _thinking_configs(types)
    delay = 10.0
    attempt = 0
    while True:
        idx = _think_choice.get(model, 0)
        cfg = types.GenerateContentConfig(temperature=0.0, thinking_config=thinks[idx]) if thinks[idx] else types.GenerateContentConfig(temperature=0.0)
        try:
            r = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=cfg,
            )
            u = r.usage_metadata
            usage = {
                "in": getattr(u, "prompt_token_count", 0) or 0,
                "out": (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
                "think": getattr(u, "thoughts_token_count", 0) or 0,
            }
            text = (r.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[-1]
            return text, usage
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 400 and idx < len(thinks) - 1 and "think" in str(e).lower():
                with _lock:  # model rejects this thinking setting: fall back to the next one
                    _think_choice[model] = max(_think_choice.get(model, 0), idx + 1)
                continue
            if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                raise SystemExit(f"Gemini API error {code} (not retried): {str(e)[:300]}")
            attempt += 1
            if attempt >= retries:
                raise
            wait = delay * (1 + random.random() * 0.5)
            print(f"      retry in {wait:.0f}s: {str(e)[:100]}")
            time.sleep(wait)
            delay = min(delay * 2, 120)


def pick_pages(n: int, k: int) -> list[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    # skip the cover page when possible; spread the rest evenly
    lo = 1 if n > k else 0
    step = (n - lo) / k
    return sorted({lo + int(i * step + step / 2) for i in range(k)})


def main():
    import pymupdf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default=None, help="Cache dir (default data/reocr/<model>)")
    ap.add_argument("--model", default=os.environ.get("KKTC_VLM_MODEL", DEFAULT_MODEL),
                    help=f"Model id (default {DEFAULT_MODEL}; local example: moganocr-tr-1.5)")
    ap.add_argument("--base-url", default=os.environ.get("KKTC_VLM_BASE_URL", ""),
                    help="OpenAI-compatible server base URL (e.g. http://127.0.0.1:8080/v1 for llama-server / vLLM)")
    ap.add_argument("--preset", choices=["auto", "gemini", "mogan"], default="auto",
                    help="Prompt & post-processing preset (auto picks 'mogan' when model name contains 'mogan')")
    ap.add_argument("--list-models", action="store_true", help="Print the model ids your API key can use and exit")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--pages-per-doc", type=int, default=0, help="Sample this many pages per document (0 = all)")
    ap.add_argument("--max-pages", type=int, default=0, help="Hard stop after this many new pages (cost guard)")
    ap.add_argument("--max-usd", type=float, default=1.0,
                    help="Stop when the estimated spend of this run reaches this many USD (default 1). Raise it deliberately for a full run.")
    ap.add_argument("--pilot", action="store_true", help="Write pilot_report.json + pilot_compare.md (before/after)")
    ap.add_argument("--in-price", type=float, default=0.0, help="USD per 1M input tokens (for the cost estimate)")
    ap.add_argument("--out-price", type=float, default=0.0, help="USD per 1M output tokens (for the cost estimate)")
    args = ap.parse_args()
    if args.list_models:
        for m in make_client().models.list():
            if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]):
                print(m.name.removeprefix("models/"))
        return
    if not (args.in_price or args.out_price) and args.model in PRICES:
        args.in_price, args.out_price = PRICES[args.model]

    use_local = bool(args.base_url)
    if use_local and args.model == DEFAULT_MODEL:
        args.model = "moganocr-tr-1.5"
    preset = ("mogan" if "mogan" in args.model.lower() else "gemini") if args.preset == "auto" else args.preset
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_")
    out = Path(args.out or f"data/reocr/{model_slug}")
    out.mkdir(parents=True, exist_ok=True)
    doc_ids = [ln.strip() for ln in Path(args.docs).read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    if args.limit_docs:
        doc_ids = doc_ids[: args.limit_docs]

    jobs = []
    total_pages = 0
    for did in doc_ids:
        src = Path(args.raw_dir) / Path(*did.split("/"))
        if not src.exists():
            print(f"   [skip] raw file missing: {did}")
            continue
        with pymupdf.open(str(src)) as d:
            n = len(d)
        total_pages += n
        (out / doc_key(did)).mkdir(exist_ok=True)
        (out / doc_key(did) / "doc_id.txt").write_text(did, encoding="utf-8")
        for pno in pick_pages(n, args.pages_per_doc):
            if not page_cache(out, did, pno).exists():
                jobs.append((did, src, pno))
    if args.max_pages:
        jobs = jobs[: args.max_pages]
    print(f"{len(doc_ids)} documents, {total_pages:,} pages in total; {len(jobs):,} pages to read with {args.model}"
          + (f" via {args.base_url} (preset={preset})" if use_local else ""))

    client = None if use_local else make_client()
    usage_log = out / "usage.jsonl"
    done = {"n": 0, "in": 0, "out": 0, "think": 0}
    t0 = time.perf_counter()

    stop = threading.Event()

    def spent() -> float:
        return (done["in"] * args.in_price + done["out"] * args.out_price) / 1e6

    if not use_local and args.max_usd and not (args.in_price or args.out_price):
        raise SystemExit(f"No price known for {args.model}: pass --in-price/--out-price so --max-usd can be enforced.")

    def work(job):
        if stop.is_set():
            return
        did, src, pno = job
        with pymupdf.open(str(src)) as d:
            png = render_png(d[pno], args.dpi)
        if use_local:
            prompt = PROMPT_MOGAN if preset == "mogan" else PROMPT
            text, usage = transcribe_openai(args.base_url, args.model, png, prompt=prompt, clean_md=(preset == "mogan"))
        else:
            text, usage = transcribe(client, args.model, png)
        cp = page_cache(out, did, pno)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, cp)
        with _lock:
            done["n"] += 1
            done["in"] += usage["in"]
            done["out"] += usage["out"]
            done["think"] += usage.get("think", 0)
            with usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": did, "page": pno + 1, "model": args.model, **usage}, ensure_ascii=False) + "\n")
            if not use_local and args.max_usd and spent() >= args.max_usd and not stop.is_set():
                stop.set()
                print(f"   COST CAP: ~${spent():.2f} spent (--max-usd {args.max_usd}); stopping. Cached pages are kept.")
            if done["n"] % 10 == 0 or done["n"] == len(jobs):
                elapsed = max(time.perf_counter() - t0, 0.1)
                cost_s = "free/local" if use_local else f"~${spent():.2f}"
                print(f"   {done['n']:,}/{len(jobs):,} pages ({done['n']/elapsed:.2f} pg/s)  tokens in={done['in']:,} out={done['out']:,}  {cost_s}  {elapsed:.0f}s", flush=True)

    errors = []
    ex = cf.ThreadPoolExecutor(max_workers=args.workers)
    futs = {ex.submit(work, j): j for j in jobs}
    pending = set(futs)
    try:
        # short waits keep the main thread responsive so Ctrl+C works on Windows too
        while pending:
            finished, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                try:
                    f.result()
                except Exception as e:
                    errors.append(f"{futs[f][0]} p{futs[f][2] + 1}: {e}")
    except KeyboardInterrupt:
        stop.set()
        print(f"\n   Stopped by user after {done['n']} pages (~${spent():.2f}). Cached pages are kept.")
        os._exit(130)  # do not wait for in-flight requests
    ex.shutdown(wait=True)
    if errors:
        print(f"   {len(errors)} pages failed (re-run to retry): {errors[:3]}")

    if done["n"]:
        per_in, per_out = done["in"] / done["n"], done["out"] / done["n"]
        est = {"pages_read": done["n"], "avg_in_tokens": round(per_in), "avg_out_tokens": round(per_out), "corpus_pages": total_pages}
        if args.in_price or args.out_price:
            per_page = (per_in * args.in_price + per_out * args.out_price) / 1e6
            est.update({"usd_per_page": round(per_page, 5), "usd_all_pages": round(per_page * total_pages, 2)})
        print(f"   estimate: {est}")
        (out / "last_run_estimate.json").write_text(json.dumps(est, indent=1), encoding="utf-8")

    if args.pilot:
        write_pilot_report(out, doc_ids, Path(args.raw_dir))


def write_pilot_report(out: Path, doc_ids: list[str], raw_dir: Path):
    """Compare the old text layer with the VLM transcription page by page."""
    import pyarrow.parquet as pq
    import pymupdf

    from extract_text import looks_letter_spaced, rebuild_letter_spaced_page
    from ocr_quality import score_text

    vp = Path("data/ocr_quality/vocab.parquet")
    vocab = set(pq.read_table(vp, columns=["word"]).column(0).to_pylist()) if vp.exists() else set()
    rows, md = [], ["# Görsel model ile yeniden okuma — pilot karşılaştırma\n"]
    for did in doc_ids:
        d_dir = out / doc_key(did)
        pages = sorted(d_dir.glob("p*.txt")) if d_dir.exists() else []
        if not pages:
            continue
        with pymupdf.open(str(raw_dir / Path(*did.split("/")))) as d:
            for p in pages:
                pno = int(p.stem[1:]) - 1
                old = d[pno].get_text("text") or ""
                if looks_letter_spaced(old):
                    old = rebuild_letter_spaced_page(d[pno])
                new = p.read_text(encoding="utf-8")
                b, a = score_text(old, vocab), score_text(new, vocab)
                rows.append({"doc_id": did, "page": pno + 1, "score_before": b["score"], "score_after": a["score"],
                             "words_before": b["words"], "words_after": a["words"], "illegible": new.count("[okunamadı]")})
                if len(md) < 1 + 3 * 8:
                    md.append(f"\n## {did} — sayfa {pno + 1} (skor {b['score']:.2f} → {a['score']:.2f})\n")
                    md.append("**Eski OCR:**\n\n```\n" + old.strip()[:1200] + "\n```\n")
                    md.append("**Görsel model:**\n\n```\n" + new.strip()[:1200] + "\n```\n")
    if not rows:
        return
    import statistics as st

    summary = {
        "n_pages": len(rows),
        "median_score_before": st.median(r["score_before"] for r in rows),
        "median_score_after": st.median(r["score_after"] for r in rows),
        "pages_improved": sum(1 for r in rows if r["score_after"] < r["score_before"]),
        "illegible_marks": sum(r["illegible"] for r in rows),
        "pages": rows,
    }
    (out / "pilot_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "pilot_compare.md").write_text("\n".join(md), encoding="utf-8")
    print(f"   pilot: median score {summary['median_score_before']:.3f} -> {summary['median_score_after']:.3f}, "
          f"improved {summary['pages_improved']}/{len(rows)} pages -> {out / 'pilot_compare.md'}")


if __name__ == "__main__":
    main()
