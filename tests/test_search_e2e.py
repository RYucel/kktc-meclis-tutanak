"""End-to-end closed-loop verification of Parquet dataset, SQLite FTS5, and local Web Search API."""

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


def main():
    hf_data = Path("data/hf_dataset/data")
    for name in ("speeches.parquet", "documents.parquet", "speakers.parquet"):
        p = hf_data / name
        assert p.exists(), f"Missing {p}"
        pf = pq.ParquetFile(p)
        print(f"Parquet OK: {name:20s} rows={pf.metadata.num_rows:>9,} size={p.stat().st_size / (1024*1024):6.2f} MB")
        assert pf.metadata.num_rows > 0

    # Start web_search.py on an ephemeral port and test all API endpoints
    port = 18765
    proc = subprocess.Popen(
        [sys.executable, "src/web_search.py", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for server startup
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats", timeout=2) as r:
                    stats = json.loads(r.read().decode("utf-8"))
                    print("API /api/stats OK:", stats)
                    assert stats["speeches"] > 500_000
                    assert stats["docs"] > 3_000
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("web_search.py failed to start")

        # Test HTML UI loads
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            html_body = r.read().decode("utf-8")
            assert r.status == 200 and "KKTC Cumhuriyet Meclisi" in html_body
            print("UI / OK:", len(html_body), "bytes")

        # Test speaker autocomplete (Turkish diacritic-insensitive)
        for q in ("erhurman", "eroglu", "denktas", "ozgur"):
            url = f"http://127.0.0.1:{port}/api/speakers?q={urllib.parse.quote(q)}"
            with urllib.request.urlopen(url, timeout=5) as r:
                items = json.loads(r.read().decode("utf-8"))
                assert len(items) > 0, f"No speakers found for {q}"
                print(f"API /api/speakers?q={q}: top={items[0]['speaker_name']} ({items[0]['speech_count']} speeches)")

        # Test speaker + FTS5 keyword search
        tests = [
            ("Tufan Erhürman", "elektrik"),
            ("Derviş Eroğlu", "anayasa"),
            ("Rauf Raif Denktaş", "kıbrıs"),
            ("Özker Özgür", "bütçe"),
        ]
        for spk, kw in tests:
            qs = urllib.parse.urlencode({"speaker": spk, "q": kw, "limit": 3})
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/search?{qs}", timeout=10) as r:
                res = json.loads(r.read().decode("utf-8"))
                print(
                    f"API /api/search speaker={spk!r} q={kw!r} -> "
                    f"total={res['total']:,} ({res['ms']} ms) first_date={res['items'][0]['tarih'] if res['items'] else None}"
                )
                assert res["total"] > 0, f"Expected matches for {spk} + {kw}"

        print("\nALL END-TO-END VERIFICATION CHECKS PASSED!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
