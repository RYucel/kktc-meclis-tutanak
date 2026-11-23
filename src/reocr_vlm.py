"""Re-read scanned PDF pages with a vision-language model (Gemini) to replace noisy OCR text layers.

The model sees the page image, so it transcribes what is printed instead of guessing from garbled text.
Every page is cached as data/reocr/<doc-hash>/pNNNN.txt (+ usage in usage.jsonl), so runs are resumable and
nothing is paid for twice. reprocess_docs.py --reocr-dir uses a document's pages only when ALL are cached.

Pilot (a few pages from each listed document, with a before/after comparison):
  set GEMINI_API_KEY=...
  python src/reocr_vlm.py --docs data/ocr_quality/noisy_docs.txt --limit-docs 20 --pages-per-doc 3 --pilot
Full run:
  python src/reocr_vlm.py --docs data/ocr_quality/noisy_docs.txt --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
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
_lock = threading.Lock()


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


def transcribe(client, model: str, png: bytes, retries: int = 5) -> tuple[str, dict]:
    from google.genai import types

    delay = 4.0
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            u = r.usage_metadata
            usage = {
                "in": getattr(u, "prompt_token_count", 0) or 0,
                "out": (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
            }
            text = (r.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[-1]
            return text, usage
        except Exception as e:  # rate limits / transient errors
            if attempt == retries - 1:
                raise
            print(f"      retry in {delay:.0f}s: {str(e)[:120]}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


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
    ap.add_argument("--out", default="data/reocr")
    ap.add_argument("--model", default=os.environ.get("KKTC_VLM_MODEL", "gemini-flash-latest"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--pages-per-doc", type=int, default=0, help="Sample this many pages per document (0 = all)")
    ap.add_argument("--max-pages", type=int, default=0, help="Hard stop after this many new pages (cost guard)")
    ap.add_argument("--pilot", action="store_true", help="Write pilot_report.json + pilot_compare.md (before/after)")
    ap.add_argument("--in-price", type=float, default=0.0, help="USD per 1M input tokens (for the cost estimate)")
    ap.add_argument("--out-price", type=float, default=0.0, help="USD per 1M output tokens (for the cost estimate)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc_ids = [ln.strip() for ln in Path(args.docs).read_text(encoding="utf-8").splitlines() if ln.strip()]
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
    print(f"{len(doc_ids)} documents, {total_pages:,} pages in total; {len(jobs):,} pages to read with {args.model}")

    client = make_client()
    usage_log = out / "usage.jsonl"
    done = {"n": 0, "in": 0, "out": 0}
    t0 = time.perf_counter()

    def work(job):
        did, src, pno = job
        with pymupdf.open(str(src)) as d:
            png = render_png(d[pno], args.dpi)
        text, usage = transcribe(client, args.model, png)
        cp = page_cache(out, did, pno)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, cp)
        with _lock:
            done["n"] += 1
            done["in"] += usage["in"]
            done["out"] += usage["out"]
            with usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": did, "page": pno + 1, "model": args.model, **usage}, ensure_ascii=False) + "\n")
            if done["n"] % 25 == 0 or done["n"] == len(jobs):
                print(f"   {done['n']:,}/{len(jobs):,} pages  tokens in={done['in']:,} out={done['out']:,}  {time.perf_counter() - t0:.0f}s")

    errors = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errors.append(f"{futs[f][0]} p{futs[f][2] + 1}: {e}")
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
