"""Closed-loop end-to-end test for the Hugging Face Space FastAPI + DuckDB app."""

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=15) as r:
        assert r.status == 200
        return json.loads(r.read().decode("utf-8"))


def main():
    port = 17860
    space_dir = Path("space").resolve()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(space_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        t0 = time.perf_counter()
        ready = False
        for _ in range(60):
            try:
                h = get_json(f"{base}/api/health")
                if h.get("ready"):
                    print(f"Space READY in {time.perf_counter() - t0:.2f}s (engine={h['load_seconds']}s)")
                    ready = True
                    break
                if h.get("error"):
                    raise RuntimeError(f"Engine error: {h['error']}")
            except urllib.error.URLError:
                pass
            time.sleep(0.5)
        assert ready, "Space did not become ready within 30s"

        # 1. HTML index
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            body = r.read().decode("utf-8")
            assert r.status == 200 and "Cumhuriyet Meclisi Konuşma Arşivi" in body
            print("GET / OK:", len(body), "bytes")

        # 2. Stats
        stats = get_json(f"{base}/api/stats")
        assert stats["ready"] and stats["speeches"] == 1272408
        print("GET /api/stats OK:", stats["speeches"], "speeches,", stats["docs"], "docs,", stats["speakers"], "speakers")

        # 3. Speaker autocomplete
        for q, expected in [("erhurman", "Tufan Erhürman"), ("eroglu", "Derviş Eroğlu"), ("ozker", "Özker Özgür")]:
            res = get_json(f"{base}/api/speakers?q={urllib.parse.quote(q)}")
            assert res and res[0]["speaker_name"] == expected, f"Got {res[:2]} for {q}"
            print(f"GET /api/speakers?q={q} OK -> {res[0]['speaker_name']} ({res[0]['speech_count']})")

        # 4. Speaker profile + timeline
        prof = get_json(f"{base}/api/speaker_profile?name={urllib.parse.quote('Derviş Eroğlu')}")
        assert prof["speaker_name"] == "Derviş Eroğlu" and len(prof["timeline"]) > 20
        print(f"GET /api/speaker_profile OK -> {prof['speaker_name']}, {len(prof['timeline'])} active years")

        # 5. Search (speaker + keyword + highlighting)
        qs = urllib.parse.urlencode({"speaker": "Tufan Erhürman", "q": "elektrik", "limit": 5})
        sr = get_json(f"{base}/api/search?{qs}")
        assert sr["total"] > 100 and "<mark>" in sr["items"][0]["snippet_html"]
        print(f"GET /api/search (Erhürman + elektrik) OK -> total={sr['total']} in {sr['ms']} ms")

        # 6. Turkish diacritic-insensitive keyword search (kibris vs kıbrıs)
        qs2 = urllib.parse.urlencode({"speaker": "Rauf Raif Denktaş", "q": "kibris", "limit": 5})
        sr2 = get_json(f"{base}/api/search?{qs2}")
        assert sr2["total"] > 50
        print(f"GET /api/search (Denktaş + kibris) OK -> total={sr2['total']} in {sr2['ms']} ms")

        # 7. Session context drawer
        first = sr["items"][0]
        ctx_qs = urllib.parse.urlencode({"doc_id": first["doc_id"], "turn_index": first["turn_index"], "window": 2})
        ctx = get_json(f"{base}/api/context?{ctx_qs}")
        assert any(t["is_focus"] for t in ctx["turns"])
        print(f"GET /api/context OK -> {len(ctx['turns'])} surrounding turns from {ctx['doc_id']}")

        # 8. CSV Export
        exp_qs = urllib.parse.urlencode({"speaker": "Tufan Erhürman", "q": "elektrik", "limit": 5})
        with urllib.request.urlopen(f"{base}/api/export?{exp_qs}", timeout=10) as r:
            csv_bytes = r.read()
            assert csv_bytes.startswith(b"\xef\xbb\xbf") and b"Tufan Erh" in csv_bytes
            print(f"GET /api/export OK -> {len(csv_bytes)} bytes UTF-8 BOM CSV")

        print("\nALL SPACE E2E TESTS PASSED!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
