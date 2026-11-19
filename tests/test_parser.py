"""Unit tests for KKTC Meclisi speaker parser and Turkish canonicalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parse_speakers import parse_document, tr_lower, tr_title, tr_upper
from build_dataset import canonicalize_speakers


def test_turkish_case():
    assert tr_lower("İRSEN KÜÇÜK") == "irsen küçük"
    assert tr_lower("IŞIK") == "ışık"
    assert tr_upper("irsen küçük") == "İRSEN KÜÇÜK"
    assert tr_title("FAİZ SUCUOĞLU") == "Faiz Sucuoğlu"
    assert tr_title("BAYINDIRLIK VE ULAŞTIRMA BAKANI") == "Bayındırlık ve Ulaştırma Bakanı"


def test_modern_docx_chair_handoff_and_turns():
    sample = {
        "local_rel_path": "Dönem 10 Yıl 1(d10y01)/B04.docx",
        "filename": "B04.docx",
        "meclis": "KKTC",
        "donem": 10,
        "yasama_yili": 1,
        "text_source": "docx",
        "download_url": "https://example.test/B04.docx",
        "text": "\n".join([
            "DÖNEM : X | YASAMA YILI: 2022/1",
            "4’üncü Birleşim",
            "25 Şubat 2022, Cuma",
            "BİRİNCİ OTURUM",
            "BAŞKAN : Geçici Başkan Zorlu TÖRE",
            "KATİP: Geçici Katip Ongun TALAT",
            "BAŞKAN – Sayın Milletvekilleri, birleşimi açıyorum.",
            "KATİP – Toplantı yeter sayısı vardır Sayın Başkan.",
            "KUDRET ÖZERSAY (Lefkoşa) - Teşekkür ederim Sayın Başkan.",
            "DIŞİŞLERİ BAKANI HASAN TAÇOY (Yerinden) – Ekip kurulmuştur.",
            "(Sayın Fazilet Özdenefe Başkanlık Kürsüsünü Sayın Salahi Şahiner’e devreder)",
            "BAŞKAN – Teşekkürler Sayın Özersay.",
            "ASIM AKANSOY (Gazimağusa) (Yerinden)(Devamla) – Hayır yapılmadı.",
        ]),
    }
    doc_rec, turns = parse_document(sample)
    assert doc_rec["tarih"] == "2022-02-25"
    assert doc_rec["birlesim"] == 4
    assert len(turns) == 6
    assert turns[0]["speaker_name"] == "Zorlu Töre"
    assert turns[1]["speaker_name"] == "Ongun Talat"
    assert turns[2]["speaker_name"] == "Kudret Özersay"
    assert turns[2]["constituency"] == "Lefkoşa"
    assert turns[3]["speaker_name"] == "Hasan Taçoy"
    assert turns[3]["speaker_title"] == "Dışişleri Bakanı"
    assert turns[3]["modifiers"] == "Yerinden"
    assert turns[4]["speaker_name"] == "Salahi Şahiner"  # Chair handoff resolved!
    assert turns[5]["speaker_name"] == "Asım Akansoy"
    assert turns[5]["constituency"] == "Gazimağusa"


def test_old_pdf_two_line_chair_and_academic_strip():
    sample = {
        "local_rel_path": "Dönem 1 Yıl 1(d01y01)/12B.pdf",
        "filename": "12B.pdf",
        "meclis": "KKTC",
        "donem": 1,
        "yasama_yili": 1,
        "text_source": "pdf_embedded_ocr",
        "download_url": "https://example.test/12B.pdf",
        "text": "\n".join([
            "12. Birleşim",
            "15 Ağustos 1985, Perşembe",
            "- BİRİNCİ OTURUM -",
            "Açılış Saati: 10.07",
            "Başkan",
            ": Başkan Yardımcısı Dt. Sait Güven",
            "Kâtipler: Feridun Onsav",
            "BAŞKAN - Değerli milletvekilleri; birleşimi açıyorum.",
            "KATİP - Nisabımız vardır Sayın Başkan.",
            "BAŞBAKAN Dr. DERVİŞ EROĞLU (Yerinden) - Evet Sayın Başkan.",
            "SERAKINCI (Devamla) - Devam ediyorum.",
            "EŞBER SERAKINCI (Gazi Mağusa) - İlk sözüm budur.",
        ]),
    }
    doc_rec, turns = parse_document(sample)
    assert doc_rec["tarih"] == "1985-08-15"
    assert doc_rec["birlesim"] == 12
    assert turns[0]["speaker_name"] == "Sait Güven"
    assert turns[1]["speaker_name"] == "Feridun Onsav"
    assert turns[2]["speaker_name"] == "Derviş Eroğlu"
    assert turns[2]["speaker_title"] == "Başbakan"

    # Test canonicalization (within-doc surname resolution)
    canonicalize_speakers(turns)
    assert turns[3]["speaker_name"] == "Eşber Serakıncı"
    assert turns[3]["canon_method"] == "doc_surname"


def test_pdf_spaced_wrapped_mojibake_and_edge_cases():
    sample = {
        "local_rel_path": "Dönem 3 Yıl 1(d03y01)/B-43.pdf",
        "filename": "B-43.pdf",
        "meclis": "KKTC",
        "donem": 3,
        "yasama_yili": 1,
        "text_source": "pdf_embedded_ocr",
        "download_url": "https://example.test/B-43.pdf",
        "text": "\n".join([
            "43. Birleşim",
            "10 Mart 1994, Perşembe",
            "BİRİNCİ OTURUM",
            "Başkan",
            ": Ayhan Halit Acarkan",
            "B A Ş K A N - Oturumu açıyorum.",
            "EAŞKAN - Söz sırası Sayın Murat'ta.",
            "ÇALIŞMA VE SOSYAL GÜVENLİK BAKANI ÖZKAN MURAT",
            "(Yerinden) - Teşekkür ederim.",
            "İDARİ VE SOSYAL İŞLER KOMİTESİ BAŞKANI KENAN",
            "AKIN - Katılıyorum.",
            "V E H B İ ZEKİ S E R T E R ( L e f k o ş a ) - Söz istiyorum.",
            "SALÝH COÞAR (Devamla) - Devam ediyorum.",
            "OZKER OZGUR -.Sayın Başkan, 75. madde hakkında.",
        ]),
    }
    doc_rec, turns = parse_document(sample)
    assert doc_rec["tarih"] == "1994-03-10"
    assert len(turns) == 7, f"Expected 7 turns, got {len(turns)}: {[t['speaker_name'] for t in turns]}"
    assert turns[0]["speaker_name"] == "Ayhan Halit Acarkan"
    assert turns[1]["speaker_name"] == "Ayhan Halit Acarkan"
    assert turns[2]["speaker_name"] == "Özkan Murat"
    assert turns[3]["speaker_name"] == "Kenan Akın"
    assert turns[4]["speaker_name"] == "Vehbi Zeki Serter"
    assert turns[4]["constituency"] == "Lefkoşa"
    assert turns[5]["speaker_name"] == "Salih Coşar"
    assert turns[6]["speaker_name"] == "Ozker Ozgur"


if __name__ == "__main__":
    test_turkish_case()
    test_modern_docx_chair_handoff_and_turns()
    test_old_pdf_two_line_chair_and_academic_strip()
    test_pdf_spaced_wrapped_mojibake_and_edge_cases()
    print("ALL UNIT TESTS PASSED!")
