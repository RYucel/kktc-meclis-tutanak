"""Resumable, polite concurrent downloader for KKTC Meclisi minutes."""

import concurrent.futures
import hashlib
import json
import os
import ssl
import threading
import time
import urllib.request
from pathlib import Path

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.jsonl")
LOG_PATH = Path("data/download_index.jsonl")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_magic(header: bytes) -> str:
    if header.startswith(b"%PDF"):
        return "pdf"
    if header.startswith(b"PK\x03\x04"):
        return "zip_docx"
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole_doc"
    if header.startswith(b"{\\rtf"):
        return "rtf"
    if b"<html" in header[:256].lower() or b"<!doctype" in header[:256].lower():
        return "html_error"
    return "other"


def download_one(rec: dict, retries: int = 4) -> dict:
    rel = rec["local_rel_path"]
    dest = RAW_DIR / Path(*rel.split("/"))
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        with dest.open("rb") as f:
            hdr = f.read(256)
        magic = detect_magic(hdr)
        if magic != "html_error":
            return {
                **rec,
                "status": "cached",
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "magic": magic,
            }

    url = rec["download_url"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KKTC-Tutanak-Archiver/1.0"},
            )
            h = hashlib.sha256()
            total = 0
            first_chunk = b""
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp:
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(1 << 17)
                        if not chunk:
                            break
                        if not first_chunk:
                            first_chunk = chunk[:256]
                        out.write(chunk)
                        h.update(chunk)
                        total += len(chunk)

            magic = detect_magic(first_chunk)
            if total == 0 or magic == "html_error":
                tmp.unlink(missing_ok=True)
                raise ValueError(f"Invalid payload (bytes={total}, magic={magic})")

            os.replace(tmp, dest)
            return {
                **rec,
                "status": "downloaded",
                "size_bytes": total,
                "sha256": h.hexdigest(),
                "magic": magic,
            }
        except Exception as e:
            last_err = str(e)
            tmp.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))

    return {
        **rec,
        "status": "error",
        "error": last_err,
        "size_bytes": 0,
        "sha256": None,
        "magic": None,
    }


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit("Run src/crawl_manifest.py first.")

    records = [json.loads(line) for line in MANIFEST_PATH.open("r", encoding="utf-8") if line.strip()]
    total = len(records)
    print(f"Downloading {total} files into {RAW_DIR.resolve()} ...")

    results = []
    errors = 0
    cached = 0
    downloaded = 0
    total_bytes = 0
    t0 = time.time()
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(download_one, r) for r in records]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            res = fut.result()
            with lock:
                results.append(res)
                total_bytes += res.get("size_bytes", 0)
                st = res["status"]
                if st == "downloaded":
                    downloaded += 1
                elif st == "cached":
                    cached += 1
                else:
                    errors += 1
                    print(f"  [ERR] {res['local_rel_path']}: {res.get('error')}", flush=True)

                if idx % 100 == 0 or idx == total:
                    elapsed = max(time.time() - t0, 0.1)
                    mb = total_bytes / (1024 * 1024)
                    rate = idx / elapsed
                    eta = (total - idx) / rate if rate > 0 else 0
                    print(
                        f"[{idx}/{total}] dl={downloaded} cached={cached} err={errors} "
                        f"total={mb:.1f} MB ({rate:.1f} files/s, ETA {eta:.0f}s)",
                        flush=True,
                    )

    results.sort(key=lambda r: r["local_rel_path"].lower())
    with LOG_PATH.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\nDONE in {time.time() - t0:.1f}s | downloaded={downloaded}, cached={cached}, "
        f"errors={errors}, size={total_bytes / (1024**3):.2f} GB"
    )


if __name__ == "__main__":
    main()
