"""Unit tests for modern & 1977-79 parliament.cy parser, ToUnicode CMap repair, and speaker resolution."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cy_modern as cm  # noqa: E402


def test_transliterate_and_overrides() -> None:
    assert cm.greek_fullname_to_turkish("Χριστόφιας Δημήτρης") == "Dimitris Hristofyas"
    assert cm.greek_fullname_to_turkish("Παπαδόπουλος Τάσσος") == "Tasos Papadopulos"
    assert cm.greek_fullname_to_turkish("Αναστασιάδης Νίκος") == "Nikos Anastasiadis"
    assert cm.greek_fullname_to_turkish("Νεοφύτου Αβέρωφ") == "Averof Neofitu"
    assert cm.greek_fullname_to_turkish("Κυπριανού Άντρος") == "Andros Kiprianu"
    assert cm.greek_fullname_to_turkish("Κυπριανού Ανδρέας") == "Andreas Kiprianu"
    assert cm.greek_fullname_to_turkish("Χατζηγιάννης Κυριάκος") == "Kiriakos Hacıyannis"
    assert cm.greek_fullname_to_turkish("Στεφάνου Στέφανος") == "Stefanos Stefanu"


def test_period_boundary_2016() -> None:
    assert cm.determine_donem("denktas_makarios_1977_1979", "1978-05-10") == 3
    assert cm.determine_donem("annan_2002_2004", "2004-04-22") == 8
    assert cm.determine_donem("crans_montana_2015_2017", "2016-04-14") == 10
    assert cm.determine_donem("crans_montana_2015_2017", "2016-06-02") == 11
    assert cm.determine_donem("crans_montana_2015_2017", "2017-07-14") == 11


def test_parse_sitting_chair_and_disambiguation() -> None:
    resolver = cm.PeriodSpeakerResolver(
        10,
        ["ΟΜΗΡΟΥ ΓΙΑΝΝΑΚΗΣ", "ΦΥΤΤΗΣ ΣΟΦΟΚΛΗΣ", "ΚΥΠΡΙΑΝΟΥ ΑΝΤΡΟΣ", "ΚΥΠΡΙΑΝΟΥ ΑΝΔΡΕΑΣ", "ΠΕΡΔΙΚΗΣ ΓΙΩΡΓΟΣ"],
    )
    pages = [
        """Πρακτικά της Βουλής των Αντιπροσώπων
Ι΄ ΒΟΥΛΕΥΤΙΚΗ ΠΕΡΙΟΔΟΣ - ΣΥΝΟΔΟΣ Ε΄
Συνεδρίαση 15ης Ιανουαρίου 2015
(Ώρα έναρξης: 4.20 μ.μ.)
(Αρ. 17)
ΠΡΟΕΔΡΟΣ:
(Γ. ΟΜΗΡΟΥ)
Κηρύσσω την έναρξη της σημερινής συνεδρίας.
ΓΡΑΜΜΑΤΕΑΣ:
(Α. ΜΙΧΑΗΛΙΔΗΣ)
Υπάρχει απαρτία, έντιμε κύριε Πρόεδρε.
ΑΝΤΡΟΣ ΚΥΠΡΙΑΝΟΥ:
Κύριε Πρόεδρε, το Κυπριακό βρίσκεται σε κρίσιμη καμπή.
Α. ΚΥΠΡΙΑΝΟΥ:
Συμφωνώ ότι χρειάζεται ενότητα.
ΠΡΟΕΔΡΕΥΩΝ:
(Σ. ΦΥΤΤΗΣ)
Το λόγο έχει ο κύριος Περδίκης.
Γ. ΠΕΡΔΙΚΗΣ:
Ευχαριστώ κύριε Πρόεδρε.
ΠΡΟΕΔΡΕΥΩΝ:
(Κτυπά τη σφύρα.)
Παρακαλώ ησυχία.
(Ώρα λήξης: 7.15 μ.μ.)
Παρόντες Βουλευτές
Κυπριανού Άντρος
Περδίκης Γιώργος
ΠΕΡΙΕΧΟΜΕΝΑ
Ομιλίες των βουλευτών."""
    ]
    doc, sp = cm.parse_sitting(
        pages=pages,
        text_source="pdf_digital",
        donem=10,
        iso_date="2015-01-15",
        seq_in_day=1,
        file_title="ΙΔ΄17 - 15.01.2015.pdf",
        source_url="https://www.parliament.cy/test.pdf",
        resolver=resolver,
    )
    assert doc["donem"] == 10
    assert doc["yasama_yili"] == 5
    assert doc["birlesim"] == 17
    names_roles = [(s["speaker_name"], s["speaker_role"]) for s in sp]
    assert names_roles == [
        ("Yanakis Omiru", "baskan"),
        ("Katip", "katip"),
        ("Andros Kiprianu", "milletvekili_bakan"),
        ("Andreas Kiprianu", "milletvekili_bakan"),
        ("Sofoklis Fittis", "gecici_baskan"),
        ("Yorgos Perdikis", "milletvekili_bakan"),
        ("Sofoklis Fittis", "gecici_baskan"),
    ]
    # Stage direction `(Κτυπά τη σφύρα.)` must remain inside speech text, while post-sitting roster/TOC is stripped
    assert "(Κτυπά τη σφύρα.)" in sp[-1]["text"]
    assert "ΠΕΡΙΕΧΟΜΕΝΑ" not in sp[-1]["text"]


def test_nllb_normalizers_and_cache_precedence(tmp_path: Path) -> None:
    import cy_translate as ct
    import cy_translate_nllb as ctn

    cleaned = ctn.normalize_greek_for_nllb("Κηρύσσω τήν έναρξιν καί διαπιστώση έάν ύπάρχη άπαρτία.")
    assert "έναρξη" in cleaned
    assert "απαρτία" in cleaned

    pol = ctn.polish_turkish_output("Yunan Kıbrıslılar ve Türk Kıbrıslılar iki bölgelik, iki topluluklu federasyon Temsilciler Evi.")
    assert pol == "Kıbrıslı Rumlar ve Kıbrıslı Türkler iki bölgeli, iki toplumlu federasyon Temsilciler Meclisi."

    # Primary cache.jsonl must take precedence over cache_nllb.jsonl
    (tmp_path / "cache_nllb.jsonl").write_text('{"h":"abc","tr":"nllb_ver"}\n{"h":"only_nllb","tr":"ok_nllb"}\n', encoding="utf-8")
    (tmp_path / "cache.jsonl").write_text('{"h":"abc","tr":"llm_ver"}\n', encoding="utf-8")
    merged = ct.load_cache(tmp_path / "cache.jsonl")
    assert merged["abc"] == "llm_ver"
    assert merged["only_nllb"] == "ok_nllb"


def test_harvest_attendance_with_parties_and_ib_page_offset() -> None:
    pages = [
        """Πρακτικά της Βουλής των Αντιπροσώπων
ΙΒ΄ ΒΟΥΛΕΥΤΙΚΗ ΠΕΡΙΟΔΟΣ - ΣΥΝΟΔΟΣ Α΄
Συνεδρίαση 16ης Σεπτεμβρίου 2021
(Αρ. 1)
ΠΡΟΕΔΡΟΣ:
Κηρύσσω την έναρξη της συνεδρίας.
ΒΟΥΛΕΥΤΡΙΑ:
Συμφωνούμε.
Σ. ΣΤΕΦΑΝΟΥ:
Ευχαριστώ, κυρία Πρόεδρε.
(Ώρα λήξης: 6.00 μ.μ.)
Παρόντες βουλευτές (αλφαβητικά κατά κόμμα)
ΔΗΣΥ
Νεοφύτου Αβέρωφ
ΑΚΕΛ-Αριστερά-Νέες Δυνάμεις
Στεφάνου Στέφανος
ΔΗΠΑ-Συνεργασία
Καρογιάν Μάριος
ΑΠΟΝΤΕΣ
Κάποιος Απών"""
    ]
    rows = cm.harvest_attendance_with_parties(pages)
    assert ("ΝΕΟΦΥΤΟΥ ΑΒΕΡΩΦ", "DİSİ") in rows
    assert ("ΣΤΕΦΑΝΟΥ ΣΤΕΦΑΝΟΣ", "AKEL") in rows
    assert ("ΚΑΡΟΓΙΑΝ ΜΑΡΙΟΣ", "DİPA") in rows
    assert all("ΑΠΩΝ" not in pair for pair, _ in rows)

    resolver = cm.PeriodSpeakerResolver(12, [pair for pair, _ in rows])
    doc, sp = cm.parse_sitting(
        pages=pages,
        text_source="pdf_digital",
        donem=12,
        iso_date="2021-09-16",
        seq_in_day=1,
        file_title="IB_A.pdf",
        source_url="https://www.parliament.cy/ib_a.pdf",
        resolver=resolver,
        page_offset=41,
        forced_yy=1,
    )
    assert doc["donem"] == 12
    assert doc["yasama_yili"] == 1
    assert [(s["speaker_name"], s["source_url"]) for s in sp] == [
        ("Annita Dimitriu", "https://www.parliament.cy/ib_a.pdf#page=42"),
        ("Bir Milletvekili", "https://www.parliament.cy/ib_a.pdf#page=42"),
        ("Stefanos Stefanu", "https://www.parliament.cy/ib_a.pdf#page=42"),
    ]


