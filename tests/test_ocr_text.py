import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_text import _join_line_tokens, looks_letter_spaced  # noqa: E402
from ocr_quality import score_text  # noqa: E402


def _boxes(spec, h=10.0, letter_gap=2.0, word_gap=11.0, w=5.0):
    """spec: list of words; each letter becomes its own box like a letter-spaced PDF text layer."""
    out, x = [], 0.0
    for wi, word in enumerate(spec):
        for li, ch in enumerate(word):
            out.append((x, 0.0, x + w, h, ch))
            x += w + (letter_gap if li < len(word) - 1 else word_gap)
    return out


def test_letter_spaced_detection():
    assert looks_letter_spaced("h e r h a n g i b i r ç ö z ü m " * 10)
    assert not looks_letter_spaced("Sayın Başkan, sayın milletvekilleri bugün bütçeyi görüşüyoruz. " * 10)


def test_join_line_rebuilds_words():
    assert _join_line_tokens(_boxes(["Halkımızın", "Anavatana", "duyduğu"])) == "Halkımızın Anavatana duyduğu"


def test_join_line_keeps_normal_words():
    boxes = [(0, 0, 40, 10, "politikamızın"), (42, 0, 70, 10, "temel"), (72, 0, 90, 10, "hedefi")]
    assert _join_line_tokens(boxes) == "politikamızın temel hedefi"


def test_score_text_flags_garbage():
    vocab = {"sayın", "başkan", "milletvekilleri", "bütçeyi", "görüşüyoruz", "bugün"}
    good = score_text("Sayın Başkan, sayın milletvekilleri bugün bütçeyi görüşüyoruz.", vocab)
    bad = score_text("Sayın mi 1 letveki 1 leri K^TC KK1lC*nin 4din b u g ü n", vocab)
    assert good["score"] < 0.05 < bad["score"]


def test_pick_pages_spreads_and_skips_cover():
    from reocr_vlm import pick_pages

    assert pick_pages(5, 0) == [0, 1, 2, 3, 4]
    picked = pick_pages(100, 3)
    assert len(picked) == 3 and 0 not in picked and picked == sorted(picked)


def test_cached_pages_requires_every_page(tmp_path):
    from reocr_vlm import cached_pages, page_cache

    for i in (0, 2):
        p = page_cache(tmp_path, "D/x.pdf", i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"sayfa {i}", encoding="utf-8")
    assert cached_pages(tmp_path, "D/x.pdf", 3) is None
    page_cache(tmp_path, "D/x.pdf", 1).write_text("sayfa 1", encoding="utf-8")
    assert cached_pages(tmp_path, "D/x.pdf", 3) == ["sayfa 0", "sayfa 1", "sayfa 2"]


def test_clean_mogan_markdown():
    from reocr_vlm import clean_mogan_markdown

    md = (
        "<header>12</header>\n"
        "## BİRİNCİ OTURUM\n\n"
        "**BAŞKAN -** Sayın milletvekilleri, toplantıyı açıyorum.\n"
        "<table><tr><th>Ad</th><th>Oy</th></tr><tr><td>Ali</td><td>Evet</td></tr></table>\n"
        "<figure/>"
        "<footer>Tutanak</footer>"
    )
    out = clean_mogan_markdown(md)
    assert "##" not in out and "**" not in out and "<header>" not in out and "<figure" not in out
    assert "BİRİNCİ OTURUM" in out
    assert "BAŞKAN - Sayın milletvekilleri, toplantıyı açıyorum." in out
    assert "Ali | Evet" in out
