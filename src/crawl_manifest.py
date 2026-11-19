"""Crawl KKTC Cumhuriyet Meclisi SharePoint archive and build complete file manifest."""

import concurrent.futures
import html as html_lib
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://evrakcm.gov.ct.tr"
ROOT_REL = "/siteler/belgeler/tutanaklar/Shared Documents"
VIEW_URL = f"{BASE}/siteler/belgeler/tutanaklar/Shared%20Documents/Forms/Belge%20Grntleme.aspx"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_html(url: str, retries: int = 4) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KKTC-Tutanak-Indexer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def get_next_page_url(page_html: str) -> str | None:
    m = re.search(r'id="bottomPagingCellWPQ1".*?</tr></table></td>', page_html, re.S)
    if not m:
        return None
    block = m.group(0)
    for a in re.finditer(r"<a\s[^>]*>.*?</a>", block, re.S):
        tag = a.group(0)
        if "next.gif" in tag or "Sonraki" in tag:
            m2 = re.search(r"RefreshPageTo\(event,\s*&quot;(.*?)&quot;\)", tag)
            if m2:
                u = html_lib.unescape(m2.group(1))
                return BASE + u if u.startswith("/") else u
    return None


def parse_items_on_page(page_html: str, folder_rel: str) -> tuple[list[str], list[str]]:
    """Return (subfolders_rel, files_rel) directly inside folder_rel from one HTML page."""
    # Find subfolder links via RootFolder=... in table rows
    subfolders = []
    files = []

    # Direct links starting with folder_rel + "/"
    prefix = folder_rel + "/"
    # Links in SharePoint list table have href="<prefix><child>" (HTML escaped)
    raw_hrefs = re.findall(r'href="([^"]+)"', page_html)
    seen = set()
    for raw in raw_hrefs:
        h = html_lib.unescape(raw)
        if not h.startswith(prefix):
            continue
        rest = h[len(prefix) :]
        if not rest or "/" in rest or rest.startswith("Forms"):
            continue
        if h in seen:
            continue
        seen.add(h)
        # Check if this item also appears as a RootFolder link (meaning it's a folder)
        enc_full = urllib.parse.quote(h, safe="")
        is_folder = (f"RootFolder={enc_full}" in page_html) or (
            "." not in rest
        )
        if is_folder:
            subfolders.append(h)
        else:
            files.append(h)

    return subfolders, files


def list_folder_all_pages(folder_rel: str) -> tuple[list[str], list[str]]:
    if folder_rel == ROOT_REL:
        url = VIEW_URL
    else:
        url = f"{VIEW_URL}?RootFolder={urllib.parse.quote(folder_rel)}"

    all_subfolders: list[str] = []
    all_files: list[str] = []
    seen_sub = set()
    seen_files = set()
    seen_urls = set()

    page = 0
    while url and url not in seen_urls and page < 100:
        seen_urls.add(url)
        page += 1
        page_html = fetch_html(url)
        subs, fls = parse_items_on_page(page_html, folder_rel)
        for s in subs:
            if s not in seen_sub:
                seen_sub.add(s)
                all_subfolders.append(s)
        for f in fls:
            if f not in seen_files:
                seen_files.add(f)
                all_files.append(f)
        url = get_next_page_url(page_html)

    return all_subfolders, all_files


def classify_folder(rel_path: str) -> dict:
    """Infer meclis, donem, yasama_yili from SharePoint relative path."""
    parts = [p for p in rel_path[len(ROOT_REL) :].strip("/").split("/") if p]
    top = parts[0] if parts else ""
    meclis = "KKTC"
    donem = None
    yasama_yili = None

    tl = top.lower()
    if "türk cemaat" in tl or "turk cemaat" in tl:
        meclis = "TCM"
    elif "kıbrıs cumhuriyeti" in tl or "kibris cumhuriyeti" in tl:
        meclis = "KC"
    elif "kty" in tl:
        meclis = "KTY"
    elif "ktfd kurucu" in tl:
        meclis = "KTFD_KM"
    elif "ktfd" in tl:
        meclis = "KTFD"
    elif "kktc kurucu" in tl:
        meclis = "KKTC_KM"
    elif "aradönem" in tl or "aradonem" in tl:
        meclis = "KKTC_ARA"
    else:
        meclis = "KKTC"

    m_d = re.search(r"d[öo]nem\s*(\d+|I{1,3})\b", top, re.I)
    if m_d:
        val = m_d.group(1).upper()
        roman = {"I": 1, "II": 2, "III": 3}
        donem = roman.get(val, int(val) if val.isdigit() else None)
    m_y = re.search(r"y[ıi]l\s*(\d+)\b", top, re.I)
    if m_y:
        yasama_yili = int(m_y.group(1))

    return {
        "top_folder": top,
        "subpath": "/".join(parts),
        "meclis": meclis,
        "donem": donem,
        "yasama_yili": yasama_yili,
    }


def crawl_subtree(folder_rel: str) -> list[dict]:
    records = []
    queue = [folder_rel]
    visited = set()
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        subs, fls = list_folder_all_pages(cur)
        for s in subs:
            if s not in visited:
                queue.append(s)
        meta = classify_folder(cur)
        for f in fls:
            filename = f.rsplit("/", 1)[-1]
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            records.append(
                {
                    "rel_path": f,
                    "local_rel_path": f[len(ROOT_REL) :].lstrip("/"),
                    "download_url": BASE + urllib.parse.quote(f, safe="/"),
                    "filename": filename,
                    "ext": ext,
                    **meta,
                }
            )
    return records


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Listing top-level folders...")
    top_folders, root_files = list_folder_all_pages(ROOT_REL)
    print(f"Found {len(top_folders)} top-level folders, {len(root_files)} root files.")

    all_records: list[dict] = []
    for f in root_files:
        filename = f.rsplit("/", 1)[-1]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        all_records.append(
            {
                "rel_path": f,
                "local_rel_path": filename,
                "download_url": BASE + urllib.parse.quote(f, safe="/"),
                "filename": filename,
                "ext": ext,
                "top_folder": "",
                "subpath": "",
                "meclis": "UNKNOWN",
                "donem": None,
                "yasama_yili": None,
            }
        )

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        fut_map = {ex.submit(crawl_subtree, tf): tf for tf in top_folders}
        done = 0
        for fut in concurrent.futures.as_completed(fut_map):
            tf = fut_map[fut]
            done += 1
            try:
                recs = fut.result()
                all_records.extend(recs)
                print(f"[{done}/{len(top_folders)}] {tf.rsplit('/', 1)[-1]} -> {len(recs)} files")
            except Exception as e:
                errors.append({"folder": tf, "error": str(e)})
                print(f"[{done}/{len(top_folders)}] ERROR {tf}: {e}")

    # Sort deterministically
    all_records.sort(key=lambda r: r["local_rel_path"].lower())

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ext_counts = Counter(r["ext"] for r in all_records)
    meclis_counts = Counter(r["meclis"] for r in all_records)
    folder_counts = Counter(r["top_folder"] for r in all_records)

    summary = {
        "total_top_folders": len(top_folders),
        "total_files": len(all_records),
        "by_ext": dict(ext_counts.most_common()),
        "by_meclis": dict(meclis_counts.most_common()),
        "by_folder": dict(sorted(folder_counts.items())),
        "errors": errors,
    }
    summary_path = out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== CRAWL SUMMARY ===")
    print(f"Total files: {len(all_records)}")
    print("By extension:", dict(ext_counts.most_common()))
    print("By meclis:", dict(meclis_counts.most_common()))
    print(f"Errors: {len(errors)}")
    print(f"Saved: {manifest_path} and {summary_path}")


if __name__ == "__main__":
    main()
