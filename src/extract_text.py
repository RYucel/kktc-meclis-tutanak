"""Extract clean text and structural metadata from KKTC Meclisi .docx, .doc, .rtf, and .pdf files."""

import argparse
import concurrent.futures
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")
MANIFEST_PATH = Path("data/manifest.jsonl")
DOWNLOAD_INDEX_PATH = Path("data/download_index.jsonl")

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_PUA_CHAR_MAP = {
    "\uf06c": "•",
    "\uf0b7": "•",
    "\uf0a7": "•",
    "\uf02d": "-",
    "\uf0be": "—",
}
_PUA_STRIP_RE = re.compile(r"[\ue000-\uf8ff\ufffd]")


def normalize_turkish_text(s: str) -> str:
    """Normalize decomposed NFD Turkish letters to NFC and repair MS Word Symbol-font PUA glyphs."""
    if not s:
        return ""
    if any("\ue000" <= c <= "\uf8ff" or c == "\ufffd" for c in s):
        out = []
        for ch in s:
            if ch in _PUA_CHAR_MAP:
                out.append(_PUA_CHAR_MAP[ch])
            elif "\uf020" <= ch <= "\uf07e":
                out.append(chr(ord(ch) - 0xF000))
            elif ("\ue000" <= ch <= "\uf8ff") or ch == "\ufffd":
                continue
            else:
                out.append(ch)
        s = "".join(out)
    s = s.replace("i\u0307", "i")
    return unicodedata.normalize("NFC", s)


def detect_magic(path: Path) -> str:
    with path.open("rb") as f:
        hdr = f.read(256)
    if hdr.startswith(b"%PDF"):
        return "pdf"
    if hdr.startswith(b"PK\x03\x04"):
        return "zip_docx"
    if hdr.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole_doc"
    if hdr.startswith(b"{\\rtf"):
        return "rtf"
    return "other"


def extract_paragraph_xml(p_elem: ET.Element) -> str:
    parts = []
    for node in p_elem.iter():
        tag = node.tag
        if tag == f"{W_NS}t" and node.text:
            parts.append(node.text)
        elif tag == f"{W_NS}tab":
            parts.append("\t")
        elif tag in (f"{W_NS}br", f"{W_NS}cr"):
            parts.append("\n")
    return "".join(parts)


def extract_docx(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(f"{W_NS}body")
    lines = []
    if body is not None:
        for child in body:
            if child.tag == f"{W_NS}p":
                txt = extract_paragraph_xml(child).strip()
                if txt:
                    lines.append(txt)
            elif child.tag == f"{W_NS}tbl":
                for tr in child.findall(f".//{W_NS}tr"):
                    cells = []
                    for tc in tr.findall(f"{W_NS}tc"):
                        c_parts = [
                            extract_paragraph_xml(p).strip()
                            for p in tc.findall(f".//{W_NS}p")
                        ]
                        c_txt = " ".join(x for x in c_parts if x)
                        if c_txt:
                            cells.append(c_txt)
                    if cells:
                        lines.append(" | ".join(cells))
    full_text = normalize_turkish_text("\n".join(lines))
    return {
        "text": full_text,
        "pages": None,
        "page_count": None,
        "text_source": "docx",
        "total_chars": len(full_text),
        "empty_pages": 0,
        "image_pages": 0,
    }


_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")


def looks_letter_spaced(text: str) -> bool:
    """True when a page's text layer is letter-spaced ('h e r h a n g i b i r'): most alphabetic tokens are 1 char."""
    toks = _ALPHA_TOKEN_RE.findall(text or "")
    if len(toks) < 40:
        return False
    return sum(1 for t in toks if len(t) == 1) / len(toks) > 0.4


def _join_line_tokens(ln: list) -> str:
    """Join PyMuPDF word boxes of one visual line; glue letters whose gap is in the 'letter gap' cluster."""
    toks = [w[4] for w in ln]
    if len(toks) < 3 or sum(1 for t in toks if len(t) == 1) / len(toks) < 0.5:
        return " ".join(toks)  # tokens are already real words: every gap is a word gap
    hs = sorted(w[3] - w[1] for w in ln)
    h = max(hs[len(hs) // 2], 1.0)
    gaps = [(b[0] - a[2]) / h for a, b in zip(ln, ln[1:])]
    srt = sorted(gaps)
    thr, best = 0.6, 0.0
    for x, y in zip(srt, srt[1:]):
        jump = (y + 0.05) / (x + 0.05)
        if jump > best and 0.1 <= (x + y) / 2 <= 1.5:
            best, thr = jump, (x + y) / 2
    if best < 1.6:
        thr = 0.6
    out = toks[0]
    for g, t in zip(gaps, toks[1:]):
        out += ("" if g < thr else " ") + t
    return out


def rebuild_letter_spaced_page(page) -> str:
    """Rebuild a letter-spaced page from glyph geometry: cluster word boxes by baseline, then join by gap size."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    if not words:
        return ""
    lines, cur, cur_y, cur_h = [], [], 0.0, 1.0
    for w in sorted(words, key=lambda w: w[3]):
        if cur and abs(w[3] - cur_y) <= 0.45 * cur_h:
            cur.append(w)
        else:
            if cur:
                lines.append(cur)
            cur, cur_y, cur_h = [w], w[3], max(w[3] - w[1], 1.0)
    if cur:
        lines.append(cur)
    return "\n".join(_join_line_tokens(sorted(ln, key=lambda w: w[0])) for ln in lines)


def extract_pdf(path: Path) -> dict:
    import fitz

    doc = fitz.open(str(path))
    try:
        pages = []
        image_pages = 0
        empty_pages = 0
        spaced_pages_fixed = 0
        for page in doc:
            t = normalize_turkish_text(page.get_text("text") or "")
            if looks_letter_spaced(t):
                rebuilt = normalize_turkish_text(rebuild_letter_spaced_page(page))
                if rebuilt.strip():
                    t = rebuilt
                    spaced_pages_fixed += 1
            pages.append(t)
            if len(t.strip()) < 30:
                empty_pages += 1
            if len(page.get_images()) > 0:
                image_pages += 1
        page_count = len(doc)
    finally:
        doc.close()

    full_text = "\n\n".join(p.strip() for p in pages if p.strip())
    if page_count == 0 or empty_pages >= max(1, page_count * 0.5):
        text_source = "pdf_needs_ocr"
    elif image_pages > 0:
        text_source = "pdf_embedded_ocr"
    else:
        text_source = "pdf_digital"

    return {
        "text": full_text,
        "pages": pages,
        "page_count": page_count,
        "text_source": text_source,
        "total_chars": len(full_text),
        "empty_pages": empty_pages,
        "image_pages": image_pages,
        "spaced_pages_fixed": spaced_pages_fixed,
    }


def clean_word_text(raw: str) -> str:
    # Word COM returns \r for paragraphs, \x07 for table cell/row markers, \x0b soft newline, \x0c page break
    s = normalize_turkish_text(raw)
    s = s.replace("\r\x07", "\n").replace("\x07", " | ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x0b", "\n").replace("\x0c", "\n\n")
    s = re.sub(r"[\x00-\x08\x0e-\x1f]", "", s)
    lines = [line.strip() for line in s.split("\n")]
    # Collapse excessive blank lines while preserving paragraph boundaries
    out = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip()


class WordComExtractor:
    def __init__(self, recycle_every: int = 150):
        self.recycle_every = recycle_every
        self.word = None
        self.count = 0

    def _start(self):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self.word = win32com.client.DispatchEx("Word.Application")
        self.word.Visible = False
        self.word.DisplayAlerts = 0  # wdAlertsNone
        try:
            self.word.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        except Exception:
            pass
        try:
            self.word.Options.ConfirmConversions = False
        except Exception:
            pass
        self.count = 0

    def close(self):
        if self.word is not None:
            try:
                self.word.Quit(SaveChanges=0)
            except Exception:
                pass
            self.word = None

    def extract(self, path: Path, source_label: str = "doc") -> dict:
        if self.word is None or self.count >= self.recycle_every:
            self.close()
            self._start()

        abs_path = str(path.resolve())
        doc = None
        try:
            doc = self.word.Documents.Open(
                abs_path,
                ConfirmConversions=False,
                ReadOnly=True,
                AddToRecentFiles=False,
                PasswordDocument="",
                Revert=False,
                OpenAndRepair=False,
                NoEncodingDialog=True,
            )
            raw_text = doc.Content.Text
            doc.Close(SaveChanges=False)
            doc = None
            self.count += 1
            cleaned = clean_word_text(raw_text)
            return {
                "text": cleaned,
                "pages": None,
                "page_count": None,
                "text_source": source_label,
                "total_chars": len(cleaned),
                "empty_pages": 0,
                "image_pages": 0,
            }
        except Exception:
            if doc is not None:
                try:
                    doc.Close(SaveChanges=False)
                except Exception:
                    pass
            # Restart Word instance on failure so next file starts clean
            self.close()
            raise


def out_path_for(local_rel_path: str) -> Path:
    return EXTRACTED_DIR / Path(*local_rel_path.split("/")).with_suffix(
        Path(local_rel_path).suffix + ".json"
    )


def save_extracted(rec: dict, extracted: dict):
    out = out_path_for(rec["local_rel_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "local_rel_path": rec["local_rel_path"],
        "filename": rec["filename"],
        "ext": rec["ext"],
        "meclis": rec["meclis"],
        "donem": rec["donem"],
        "yasama_yili": rec["yasama_yili"],
        "top_folder": rec["top_folder"],
        "download_url": rec["download_url"],
        "sha256": rec.get("sha256"),
        "size_bytes": rec.get("size_bytes"),
        **extracted,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)


def process_fast_one(rec: dict) -> tuple[str, str, int]:
    """Process one PDF or DOCX in a worker process."""
    rel = rec["local_rel_path"]
    src = RAW_DIR / Path(*rel.split("/"))
    magic = detect_magic(src)
    if magic == "pdf":
        res = extract_pdf(src)
    elif magic == "zip_docx":
        res = extract_docx(src)
    else:
        return ("defer_word", rel, 0)
    save_extracted(rec, res)
    return (res["text_source"], rel, res["total_chars"])


def load_records() -> list[dict]:
    src = DOWNLOAD_INDEX_PATH if DOWNLOAD_INDEX_PATH.exists() else MANIFEST_PATH
    recs = [json.loads(line) for line in src.open("r", encoding="utf-8") if line.strip()]
    # Only include files that exist on disk
    ready = []
    for r in recs:
        p = RAW_DIR / Path(*r["local_rel_path"].split("/"))
        if p.exists() and p.stat().st_size > 0:
            ready.append(r)
    return ready


def run_word_chunk(worker_id: int, chunk: list[tuple[dict, str]]) -> tuple[int, list[dict]]:
    """Run an isolated MS Word COM instance on a shard of .doc/.rtf files."""
    extractor = WordComExtractor(recycle_every=120)
    ok = 0
    errs = []
    t0 = time.time()
    try:
        for idx, (r, magic) in enumerate(chunk, start=1):
            src = RAW_DIR / Path(*r["local_rel_path"].split("/"))
            label = "rtf" if (magic == "rtf" or r["ext"].lower() == "rtf") else "doc"
            try:
                res = extractor.extract(src, source_label=label)
                save_extracted(r, res)
                ok += 1
            except Exception as e:
                # Fallback to pure-Python OLE2 Word 6/95 & Word 97-2003 reader
                if magic == "ole_doc":
                    try:
                        from ole_doc_fallback import extract_ole_doc_text

                        raw_t = extract_ole_doc_text(src)
                        cleaned = clean_word_text(raw_t)
                        if len(cleaned) >= 20:
                            save_extracted(
                                r,
                                {
                                    "text": cleaned,
                                    "pages": None,
                                    "page_count": None,
                                    "text_source": "doc_ole_fallback",
                                    "total_chars": len(cleaned),
                                    "empty_pages": 0,
                                    "image_pages": 0,
                                },
                            )
                            ok += 1
                            continue
                    except Exception as e2:
                        e = RuntimeError(f"WordCOM={e} | OLEFallback={e2}")
                errs.append({"file": r["local_rel_path"], "error": str(e)})
                print(f"  [W{worker_id} ERR] {r['local_rel_path']}: {e}", flush=True)
            if idx % 50 == 0 or idx == len(chunk):
                rate = idx / max(time.time() - t0, 0.1)
                print(
                    f"  [W{worker_id} {idx}/{len(chunk)}] {rate:.1f} files/s err={len(errs)}",
                    flush=True,
                )
    finally:
        extractor.close()
    return ok, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "pdf", "docx", "doc"], default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--word-workers", type=int, default=4)
    args = ap.parse_args()

    records = load_records()
    print(f"Found {len(records)} downloaded files on disk.")

    todo = []
    for r in records:
        if not args.force and out_path_for(r["local_rel_path"]).exists():
            continue
        ext = r["ext"].lower()
        if args.only == "pdf" and ext != "pdf":
            continue
        if args.only == "docx" and ext != "docx":
            continue
        if args.only == "doc" and ext not in ("doc", "rtf"):
            continue
        todo.append(r)

    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"Pending extraction: {len(todo)} files.")
    if not todo:
        return

    fast_jobs = []
    word_jobs = []
    for r in todo:
        p = RAW_DIR / Path(*r["local_rel_path"].split("/"))
        magic = detect_magic(p)
        if magic in ("pdf", "zip_docx"):
            fast_jobs.append(r)
        else:
            word_jobs.append((r, magic))

    print(f"Plan: {len(fast_jobs)} fast (PDF/DOCX) + {len(word_jobs)} Word COM (DOC/RTF/other)")
    t0 = time.time()
    errors = []

    # 1. Fast parallel extraction for PDF & DOCX
    if fast_jobs:
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(process_fast_one, r): r for r in fast_jobs}
            for fut in concurrent.futures.as_completed(fut_map):
                r = fut_map[fut]
                done += 1
                try:
                    status, rel, chars = fut.result()
                    if status == "defer_word":
                        word_jobs.append((r, "other"))
                except Exception as e:
                    # Try Word COM fallback for broken docx
                    if r["ext"].lower() == "docx":
                        word_jobs.append((r, "docx_fallback"))
                    else:
                        errors.append({"file": r["local_rel_path"], "error": str(e)})
                        print(f"  [ERR] {r['local_rel_path']}: {e}", flush=True)
                if done % 100 == 0 or done == len(fast_jobs):
                    print(
                        f"  [PDF/DOCX {done}/{len(fast_jobs)}] elapsed={time.time() - t0:.1f}s err={len(errors)}",
                        flush=True,
                    )

    # 2. Parallel MS Word COM extraction for .doc / .rtf / fallbacks
    if word_jobs:
        t_word = time.time()
        n_w = max(1, min(args.word_workers, len(word_jobs)))
        shards = [word_jobs[i::n_w] for i in range(n_w)]
        if n_w == 1:
            _, errs = run_word_chunk(0, shards[0])
            errors.extend(errs)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_w) as ex:
                futs = [ex.submit(run_word_chunk, wid, sh) for wid, sh in enumerate(shards)]
                for fut in concurrent.futures.as_completed(futs):
                    _, errs = fut.result()
                    errors.extend(errs)
        print(
            f"  [DOC/RTF DONE] {len(word_jobs)} files in {time.time() - t_word:.1f}s err={len(errors)}",
            flush=True,
        )

    if errors:
        Path("data/extract_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Extraction finished in {time.time() - t0:.1f}s | errors={len(errors)}")


if __name__ == "__main__":
    main()
