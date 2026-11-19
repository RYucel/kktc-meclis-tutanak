"""Real Chromium Playwright E2E test against the LIVE deployed Hugging Face Static Space."""

import time
from playwright.sync_api import sync_playwright

LIVE_URL = "https://stevevaius-kktc-meclis-tutanak-arama.static.hf.space/"


def main():
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        t0 = time.perf_counter()
        print("1. Opening live Space:", LIVE_URL)
        page.goto(LIVE_URL, wait_until="domcontentloaded", timeout=30000)

        # Wait for speakers_index.json directory cards
        page.wait_for_selector(".dir-card", timeout=15000)
        dir_cnt = page.locator(".dir-card").count()
        print(f"   Landing directory rendered: {dir_cnt} featured speakers in {time.perf_counter()-t0:.2f}s")
        assert dir_cnt == 24

        # Wait for DuckDB-WASM to become ready
        page.wait_for_function("document.getElementById('engine-dot').classList.contains('ok')", timeout=30000)
        print(f"2. DuckDB-WASM READY in {time.perf_counter()-t0:.2f}s -> {page.locator('#engine-text').inner_text()}")

        # Test speaker autocomplete
        page.fill("#speaker", "erhurman")
        page.wait_for_selector(".ac-item", timeout=5000)
        first_ac = page.locator(".ac-item .ac-name").first.inner_text()
        print("3. Autocomplete 'erhurman' ->", first_ac)
        assert first_ac == "Tufan Erhürman"
        page.locator(".ac-item").first.click()

        # Wait for dossier + speech cards from real HTTP Range Parquet query!
        t_q = time.perf_counter()
        page.wait_for_selector("#dossier .dossier-name", timeout=20000)
        page.wait_for_selector(".speech-card", timeout=45000)
        print(
            f"4. Live speaker query ('Tufan Erhürman') returned in {time.perf_counter()-t_q:.2f}s | "
            f"dossier={page.locator('#dossier .dossier-name').inner_text()} | "
            f"count_bar={page.locator('#results-count').inner_text()}"
        )
        assert page.locator(".speech-card").count() > 0
        assert page.locator(".t-col").count() >= 10

        # Add keyword 'elektrik' and search
        t_q2 = time.perf_counter()
        page.fill("#q", "elektrik")
        page.click("#btn-submit")
        page.wait_for_selector(".speech-card mark", timeout=45000)
        cnt_text = page.locator("#results-count").inner_text()
        first_snip = page.locator(".speech-card .sc-body").first.inner_text()[:140]
        print(f"5. Live speaker+keyword ('Tufan Erhürman' + 'elektrik') in {time.perf_counter()-t_q2:.2f}s | {cnt_text}")
        print(f"   First hit: {first_snip}")

        # Open full speech drawer
        page.locator(".speech-card .read-btn").first.click()
        page.wait_for_selector("#drawer-backdrop.open .ctx-text", timeout=5000)
        full_len = len(page.locator("#drawer-body .ctx-text").inner_text())
        print(f"6. Full speech drawer opened OK ({full_len} chars)")
        page.click("#drawer-close")

        # Quick-pick Derviş Eroğlu + anayasa
        t_q3 = time.perf_counter()
        page.fill("#q", "anayasa")
        page.locator(".chip", has_text="Derviş Eroğlu").click()
        page.wait_for_function(
            "document.querySelector('#dossier .dossier-name')?.textContent === 'Derviş Eroğlu' && document.querySelectorAll('.speech-card mark').length > 0",
            timeout=45000,
        )
        print(f"7. Quick-pick ('Derviş Eroğlu' + 'anayasa') in {time.perf_counter()-t_q3:.2f}s | {page.locator('#results-count').inner_text()}")

        browser.close()

    print(f"\nConsole errors: {len(console_errors)} | Page errors: {len(page_errors)}")
    for e in console_errors + page_errors:
        print("  ERR:", e)
    assert len(page_errors) == 0, f"Uncaught page errors: {page_errors}"
    assert len(console_errors) == 0, f"Console errors: {console_errors}"
    print("ALL LIVE PLAYWRIGHT BROWSER TESTS PASSED 100%!")


if __name__ == "__main__":
    main()
