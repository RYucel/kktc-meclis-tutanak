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


def test_stream_replace_documents_removes_replaces_and_appends(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reprocess_docs import stream_replace_documents

    old = tmp_path / "docs.parquet"
    pq.write_table(pa.Table.from_pylist([{"doc_id": f"d{i}", "text": f"t{i}"} for i in range(5)]), old)
    out = tmp_path / "out.parquet"
    n = stream_replace_documents(str(old), [{"doc_id": "d1", "text": "NEW"}, {"doc_id": "z9", "text": "added"}], out, remove={"d3"})
    rows = pq.read_table(out).to_pylist()
    assert n == 5 and [r["doc_id"] for r in rows] == ["d0", "d1", "d2", "d4", "z9"]
    assert rows[1]["text"] == "NEW" and rows[-1]["text"] == "added"


def test_cy_parse_labels_and_dates():
    from cy_parse import find_date, greek_header, is_label, resolve_label

    assert resolve_label("TPAMMATEYZ :")[0] == "Katip"           # ΓΡΑΜΜΑΤΕΥΣ OCR'd in Latin look-alikes
    assert resolve_label("TIPOEAPOZ")[0] == "Glafkos Klerides"    # ΠΡΟΕΔΡΟΣ with Π read as 'TI'
    assert resolve_label("B. AYZZAPIAHZ")[0] == "Vasos Lisaridis"
    assert resolve_label("Η. Α. ΕΙΖΑ")[0] == "Halit Ali Rıza"     # Turkish member in Greek script
    assert resolve_label("H.A.RİZA")[0] == "Halit Ali Rıza"
    assert resolve_label("SOTUSU")[1] == "metin"                     # 'SORUSU' heading, not a speaker
    assert is_label("GÜNDEM :") is None and is_label("ΠΡΟΕΔΡΟΣ :") == "ΠΡΟΕΔΡΟΣ"
    assert find_date("29 KASIM 196o TARİHLİ TOPLANTI.", (1960, 1961)) == "1960-11-29"
    assert find_date("ΣΥΝΕΔΡΙΑΣΙΣ ΤΗΣ 29ης ΑΥΓΟΥΣΤΟΥ 1960", (1960, 1961)) == "1960-08-29"
    assert greek_header("ΣΥΝΕΔΡΙΑΣΙΣ ΤΗΣ Ί8ης ΙΑΝΟΥΑΡΙΟΥ 1961", (1960, 1961)) == "1961-01"


def test_normalize_turkish_text():
    from extract_text import normalize_turkish_text

    assert normalize_turkish_text("askeri is\u0327 birlig\u0306i") == "askeri iş birliği"
    assert normalize_turkish_text("KÂ\uf054İP İLKAY KÂ\uf04dİL") == "KÂTİP İLKAY KÂMİL"
    assert normalize_turkish_text("\uf06c Madde\ufffd") == "• Madde"


def test_clean_mogan_markdown_strips_inline_html_and_repetitions():
    from reocr_vlm import clean_mogan_markdown

    md = "Sayın<sup>H</sup> Başkan,\n" + " ".join(["ἡ"] * 12)
    out = clean_mogan_markdown(md)
    assert "<sup>" not in out
    assert out == "SayınH Başkan,\nἡ"


def test_render_page_image_blank_and_crop():
    import pymupdf
    from reocr_vlm import render_page_image

    doc = pymupdf.open()
    blank = doc.new_page(width=842, height=1190)
    assert render_page_image(blank, dpi=200, as_jpeg=True)[0] is None

    boxed = doc.new_page(width=842, height=1190)
    boxed.insert_text((320, 500), "BAŞKAN - Sayın milletvekilleri, toplantıyı açıyorum.", fontsize=14)
    img_bytes, mime = render_page_image(boxed, dpi=200, as_jpeg=True)
    assert mime == "image/jpeg" and img_bytes is not None
    cropped_pix = pymupdf.Pixmap(img_bytes)
    full_pix = boxed.get_pixmap(matrix=pymupdf.Matrix(2350 / 1190, 2350 / 1190), colorspace="gray")
    assert cropped_pix.width * cropped_pix.height < 0.5 * full_pix.width * full_pix.height
    doc.close()
