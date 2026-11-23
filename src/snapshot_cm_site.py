"""Save raw HTML snapshots of cm.gov.ct.tr pages about former MPs and past Speakers.

cm.gov.ct.tr only answers requests from inside the region, so this runs on the sync machine.
It saves the seed pages plus every same-site page they link to that looks like a term / member /
speaker listing, so the roster parser can be written against the real markup:

    python src/snapshot_cm_site.py            # -> data/cm_snapshot/*.html + index.json

Nothing is parsed or uploaded here; commit or share data/cm_snapshot/ (a few hundred KB).
"""

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://cm.gov.ct.tr/"
SEEDS = ["GecmisMiletvekilleri", "GecmisMeclisBaskan", "Milletvekillerimiz1.aspx"]
FOLLOW_RE = re.compile(r"(gecmis|donem|milletvekil|baskan|meclis|uye)", re.I)
OUT = Path("data/cm_snapshot")
MAX_PAGES = 150

# Same relaxed TLS context the crawler already uses for the gov.ct.tr sites (incomplete cert chain)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kktc-meclis-tutanak snapshot)"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as r:
        raw = r.read()
        enc = r.headers.get_content_charset() or "utf-8"
    return raw.decode(enc, errors="replace")


def slug(url: str) -> str:
    p = urllib.parse.urlparse(url)
    s = (p.path.strip("/") or "index") + ("_" + p.query if p.query else "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:150] + ".html"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue = [urllib.parse.urljoin(BASE, s) for s in SEEDS]
    seen, index = set(), []
    while queue and len(seen) < MAX_PAGES:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch(url)
        except Exception as e:
            print(f"[ERR] {url}: {e}")
            index.append({"url": url, "error": str(e)})
            continue
        fn = slug(url)
        (OUT / fn).write_text(html, encoding="utf-8")
        index.append({"url": url, "file": fn, "bytes": len(html)})
        print(f"[OK ] {url} -> {fn} ({len(html):,} chars)")
        for href in re.findall(r'href=["\']([^"\'#]+)', html, flags=re.I):
            nxt = urllib.parse.urljoin(url, href)
            if urllib.parse.urlparse(nxt).netloc.endswith("cm.gov.ct.tr") and FOLLOW_RE.search(nxt) and nxt not in seen:
                if not re.search(r"\.(jpg|jpeg|png|gif|pdf|doc|docx|css|js)$", nxt, re.I):
                    queue.append(nxt)
        time.sleep(0.5)  # be polite to the parliament's server
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {sum(1 for i in index if 'file' in i)} pages to {OUT}/")


if __name__ == "__main__":
    main()
