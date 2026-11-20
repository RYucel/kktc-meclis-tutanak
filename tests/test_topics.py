"""Regression tests for synonym unification & word-boundary false-positive prevention in analyze_topics.py."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_topics import UNIFIED_CONCEPTS, build_re2_pattern, norm_tr_i


def _matches(concept_label: str, text: str) -> bool:
    pats = next(p for lbl, p, _ in UNIFIED_CONCEPTS if lbl == concept_label)
    rx = build_re2_pattern(pats)
    return bool(re.search(rx, norm_tr_i(text)))


def test_kibtek_synonyms_unified():
    for sample in [
        "KIB-TEK yönetim kurulu karar aldı.",
        "KIBTEK borçları ödenmelidir.",
        "KIPTEK santralinde arıza çıktı.",
        "KIB TEK çalışanları grevde.",
        "Kıbrıs Türk Elektrik Kurumu bütçesi görüşülüyor.",
        "Elektrik Kurumu akaryakıt ihalesi.",
    ]:
        assert _matches("KIB-TEK / Elektrik Kurumu", sample), f"Missed KIB-TEK variant: {sample}"


def test_no_substring_false_positives():
    # Aksa must NOT match maksat / baksak / aksaklık / olacaksak
    assert _matches("Aksa", "Aksa Enerji sözleşmesi uzatıldı.")
    assert _matches("Aksa", "Aksa'ya ödenen tutar artmıştır.")
    for neg in [
        "Bu yasanın maksadı açıktır.",
        "Baksak göreceğiz ki durum ciddi.",
        "İdarede aksaklık yaşanmıştır.",
        "Yarın görüşecek olacaksak hazır olalım.",
    ]:
        assert not _matches("Aksa", neg), f"False positive Aksa on: {neg}"

    # çevre / kirlilik must NOT match demokratik / pratik / Lefkoşa çevresi
    assert _matches("çevre / kirlilik", "Çevre kirliliği ve atık su sorunu büyüyor.")
    assert _matches("çevre / kirlilik", "Çevre Koruma Dairesi raporu.")
    for neg in [
        "Demokratik bir sistemde pratik çözümler gerekir.",
        "Lefkoşa çevresi temsilcisi söz istedi.",
        "Bürokratik engelleri aşmalıyız.",
    ]:
        assert not _matches("çevre / kirlilik", neg), f"False positive çevre/atık on: {neg}"

    # tarım must NOT match tasarım / aktarım
    assert _matches("tarım / çiftçi", "Tarım politikaları ve çiftçiler desteklenmeli.")
    assert not _matches("tarım / çiftçi", "Bu tasarım ve veri aktarımı tamamlandı.")

    # orman must NOT match performans / Nazif Borman
    assert _matches("orman / yangın", "Orman yangınlarına karşı tedbir alınmalı.")
    assert not _matches("orman / yangın", "Sayın Nazif Borman yüksek performans gösterdi.")

    # iskan must NOT match kıskanmak
    assert _matches("iskan", "İskan ve rehabilitasyon yasası.")
    assert not _matches("iskan", "Kimseyi kıskanmak doğru değildir.")

    # kredi must NOT match akreditasyon
    assert _matches("kredi / faiz", "Çiftçilere düşük faizli kredi sağlandı.")
    assert not _matches("kredi / faiz", "Üniversitelerin akreditasyon süreçleri tamamlandı.")


if __name__ == "__main__":
    test_kibtek_synonyms_unified()
    test_no_substring_false_positives()
    print("ALL TOPIC & SYNONYM UNIT TESTS PASSED!")
