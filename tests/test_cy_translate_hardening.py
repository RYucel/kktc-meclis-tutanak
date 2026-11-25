"""Unit tests for Greek detection, OCR noise filtering, and strict cache validation in cy_translate."""
from __future__ import annotations

import pytest

import src.cy_translate as ct


def test_has_greek_ignores_stray_ocr_in_turkish() -> None:
    # Real bug from Synodos A: '(o)' OCR'd as Greek omicron '(ο)' inside a purely Turkish speech
    turkish_with_ocr_bullet = (
        "(ο) Ay Nikola (Esentepe)-Baf yolunun tamiri.\n\n"
        "Umumiyetle yol işleriyle ilgili olarak tasvip edilen ve müstacel olan her projenin "
        "erken bir vakitte tatbik edilmesi."
    )
    assert ct.has_greek(turkish_with_ocr_bullet) is False

    # Stray 'νε' for 've' in short Turkish parenthetical
    assert ct.has_greek("(Üçüncü νε son defa tasarıyı okudu.)") is False
    # Stray Greek initial 'Ἑ.' in Turkish question header
    assert ct.has_greek("Lefkoşa Temsilcisi Sayın Ἑ. Papaioannu'nun 1 Eylül 1960 tarihli sorusu") is False


def test_has_greek_accepts_real_greek_and_merged_turns() -> None:
    assert ct.has_greek("Μάλιστα.") is True
    assert ct.has_greek("Ὄχι.") is True
    assert ct.has_greek("Ὁ κ. Ποταμίτης.") is True
    # Merged Greek Chair intro + long Turkish MP speech (>=30 Greek chars)
    merged = (
        "Ὑπάρχουν ἄλλοι ὁμιληταὶ ἐπὶ τοῦ θέματος ;\n\nὉ κ. Τζεμίλ.\n\n"
        "κ. CEMİL:\nPek Sayın Meclis Reisi, Reis Vekili ve Üyeler; kendi kanaatımca "
        + ("bu tasarı fevkalâde mühimdir. " * 25)
    )
    assert ct.has_greek(merged) is True


def test_is_valid_turkish_translation_rejects_leftover_greek() -> None:
    src = "Δὲν εἶναι εὔκολον νὰ θέσω τὸ ζήτημα εἰς τὰ Ἑλληνικὰ μέλη διὰ νὰ συμφωνήσουν."
    assert ct.is_valid_turkish_translation(src, src) is False
    # Partial Greek header left untranslated (>=15 Greek chars)
    partial = "Ἐρώτησις ὑποβληθεῖσα τὴν 21ην Σεπτεμβρίου 1960:\n«Mümkün müdür?»"
    assert ct.is_valid_turkish_translation(src, partial) is False
    # Clean Turkish output
    clean = "Konuyu Rum üyelere sunup uzlaşmalarını sağlamak kolay değildir."
    assert ct.is_valid_turkish_translation(src, clean) is True


def test_single_chunk_invalid_retry_raises_instead_of_polluting_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    src = "Συμφώνως τῶν προνοιῶν τοῦ ἄρθρου 76 τοῦ Συντάγματος κηρύσσω τὴν ἔναρξιν τῆς συνεδριάσεως."
    monkeypatch.setattr(ct, "_call_nim", lambda *a, **kw: src)
    with pytest.raises(RuntimeError, match="invalid/untranslated"):
        ct.translate_batch([("deadbeef", src)], api_key="dummy", model="dummy")


def test_lookup_translation_preserves_turkish_chunk_without_cache() -> None:
    el_para = "Παρακαλῶ ὅπως ἡ εἰσηγητικὴ ἔκθεσις τῶν δύο Ἐπιτροπῶν ἀναγνωσθῇ εἰς τὴν Τουρκικὴν γλῶσσαν."
    tr_para = "(ο) Umumiyetle yol işleriyle ilgili olarak tasvip edilen ve müstacel olan her projenin tatbiki. " * 35
    combined = f"{el_para}\n\n{tr_para}"
    cache = {ct.chunk_hash(el_para): "İki Komitenin gerekçeli raporunun Türkçe okunmasını rica ederim."}
    out = ct.lookup_translation(combined, cache, max_chars=2500)
    assert out is not None
    assert "İki Komitenin gerekçeli raporunun Türkçe okunmasını rica ederim." in out
    assert "müstacel olan her projenin tatbiki" in out


def test_aligned_batch_preserves_valid_items_and_retries_only_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [(f"h{i}", f"Συμφώνως τοῦ ἄρθρου {i} τοῦ Συντάγματος.") for i in range(12)]
    calls: list[str] = []

    def fake_call(api_key: str, model: str, instruction: str, user_text: str) -> str:
        calls.append(user_text)
        if len(calls) == 1:
            blocks = []
            for i in range(12):
                if i == 5:
                    # Item 5 invalid: >15 Greek chars left untranslated
                    blocks.append(f"<<<{i}>>>\nἘρώτησις ὑποβληθεῖσα τὴν 21ην Σεπτεμβρίου 1960.")
                else:
                    blocks.append(f"<<<{i}>>>\nAnayasa'nın {i}. maddesi uyarınca.")
            return "\n\n".join(blocks)
        # Subsequent single-item retries for h5 fail with HTTP error / untranslated
        return items[5][1]

    monkeypatch.setattr(ct, "_call_nim", fake_call)
    out = ct.translate_batch(items, api_key="dummy", model="dummy")
    # 1 initial batch call + 2 single-item retries for h5 only (never binary-splitting the 11 good items)
    assert len(calls) == 3
    assert len(out) == 11
    for i in range(12):
        if i != 5:
            assert out[f"h{i}"] == f"Anayasa'nın {i}. maddesi uyarınca."


def test_binary_split_preserves_left_half_when_right_half_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        ("h0", "Μάλιστα, κύριε Πρόεδρε."),
        ("h1", "Συμφωνῶ ἀπολύτως."),
        ("h2", "Τὸ ἄρθρον 76 τοῦ Συντάγματος."),
        ("h3", "Δὲν ὑπάρχει ἀντίρρησις."),
    ]

    def fake_call(api_key: str, model: str, instruction: str, user_text: str) -> str:
        if "<<<3>>>" in user_text:
            # Tag mismatch on initial 4-item batch
            return "<<<0>>>\nEvet.\n<<<1>>>\nKatılıyorum."
        if "Μάλιστα" in user_text and "Συμφωνῶ" in user_text:
            return "<<<0>>>\nEvet, Sayın Başkan.\n<<<1>>>\nTamamen katılıyorum."
        raise RuntimeError("NIM request failed after 14 attempts: HTTP 429")

    monkeypatch.setattr(ct, "_call_nim", fake_call)
    out = ct.translate_batch(items, api_key="dummy", model="dummy")
    assert out == {"h0": "Evet, Sayın Başkan.", "h1": "Tamamen katılıyorum."}

