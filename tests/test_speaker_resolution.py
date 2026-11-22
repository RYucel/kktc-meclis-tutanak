"""Unit + end-to-end tests for the 3rd-pass speaker resolution (src/speaker_resolution.py, src/recanonicalize.py).

Every guard rail is tested in both directions: the merge we want happens, and the look-alike merge
that would put words in the wrong politician's mouth does not.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speaker_resolution import (  # noqa: E402
    SpeakerResolver,
    alias_folds,
    fold,
    resolve_records,
)


def _st(count, years, docs=None):
    """stats entry with turns spread evenly over the given years"""
    years = list(years)
    per = max(1, count // len(years))
    return {"count": count, "years": Counter({y: per for y in years}), "docs": docs if docs is not None else max(1, count // 20)}


ROSTERS = {
    "KKTC:7": {"meclis": "KKTC", "donem": 7, "members": [{"name": "Hamza Ersan Saner"}, {"name": "Ünal Üstel"}]},
    "KKTC:3": {"meclis": "KKTC", "donem": 3, "members": [{"name": "Kutlu Evren"}, {"name": "Ünal Üstel"}]},
}


@pytest.fixture(scope="module")
def resolver():
    stats = {
        "Derviş Eroğlu": _st(5000, range(1976, 2010)),
        "Salih Coşar": _st(3000, range(1981, 2004)),
        "Hakkı Atun": _st(3000, range(1976, 1999)),
        "Osman N. Örek": _st(1400, range(1975, 1985)),
        "Osman Örek": _st(500, range(1973, 1991)),
        "Sait Güven": _st(2000, range(1985, 1991)),
        "Ut. Sait Güven": _st(160, [1989, 1990], docs=8),
        "Hamza Ersan Saner": _st(3000, range(2009, 2022)),
        "Ersan Saner": _st(182, range(2009, 2021)),
        "Kutlu Evren": _st(400, range(1990, 1996)),
        "Ünal Üstel": _st(1600, range(1991, 2027)),
        "Raif R. Denktaş": _st(3000, range(1976, 1986)),
        "Rauf Raif Denktaş": _st(1400, range(1960, 2005)),
        "Ahmet Atamsoy": _st(168, range(1977, 1986)),
        "Ahmet Yusuf": _st(122, [1975, 1976]),
        "Naci Talat Usar": _st(2000, range(1975, 1991)),
        "Mehmet Ali Talat": _st(2200, range(1994, 2006)),
        "Mehmet A. Yılmaz": _st(300, range(1990, 1994)),
        "Mehmet B. Yılmaz": _st(300, range(1990, 1994)),
        "Tahsin Ertuğruloğlu": _st(2000, range(1993, 2027)),
        "Özel Tahsin": _st(400, range(1976, 1990)),
        # identity that only ever appears wrapped in a title (pre-1983, no roster)
        "İkinci Başkan Vekili Ümit Süleyman": _st(461, [1975, 1976], docs=9),
        "Ümit Süleyman Lefkoşa)": _st(97, [1975, 1976], docs=3),
        # generic noise that must stay unverified
        "Konuşmaya Geçelim": _st(57, [1978], docs=1),
        "Tesbit Edilemeyen Bir Milletvekili": _st(140, range(1975, 2005), docs=60),
    }
    return SpeakerResolver(stats, ROSTERS)


def _r(resolver, name, role="milletvekili_bakan", year=None):
    res = resolver.resolve(name, role, year)
    return (res.name, res.method) if res else None


# ---------------------------------------------------------------- glued titles / OCR prefixes
@pytest.mark.parametrize(
    "label,year,expected",
    [
        ("Başbakandr . Dervişeroğlu", 1988, "Derviş Eroğlu"),
        ("Başbakanderviş Eroğlu", 1995, "Derviş Eroğlu"),
        ("Barbakan Dr Derviş Eroglu", 1986, "Derviş Eroğlu"),
        ("Ekonomi ve Maliyebakanısalihcoşar", 1990, "Salih Coşar"),
        ("Heclis Başkanı Hakkı Atun", 1986, "Hakkı Atun"),
        ("Başbakanhakkı Atun", 1995, "Hakkı Atun"),
        ("Bnşkan Vekili Osman Örek", 1984, "Osman Örek"),
        ("Hakkı Attın", 1986, "Hakkı Atun"),
    ],
)
def test_glued_titles_and_ocr_are_rescued(resolver, label, year, expected):
    out = _r(resolver, label, year=year)
    assert out is not None and out[0] == expected, out


def test_temporal_guard_blocks_inactive_anchor(resolver):
    # Hakkı Atun's activity ends in 1998 — a 2015 label must not be attributed to him
    assert _r(resolver, "Başbakanhakkı Atun", year=2015) is None


def test_unknown_year_requires_near_exact_match(resolver):
    assert _r(resolver, "Başbakandr . Dervişeroğlu", year=None)[0] == "Derviş Eroğlu"  # exact after cleanup
    assert _r(resolver, "Hakkı Attın", year=None) is None  # fuzzy-only evidence without a date


# ---------------------------------------------------------------- anchor consolidation
def test_initial_variant_merges_to_form_without_initial(resolver):
    assert resolver.anchor_map.get("Osman N. Örek") == "Osman Örek"
    assert _r(resolver, "Osman N. Örek", role="baskan", year=1980) == ("Osman Örek", "anchor_merge")


def test_conflicting_initials_are_different_people(resolver):
    assert "Mehmet A. Yılmaz" not in resolver.anchor_map
    assert "Mehmet B. Yılmaz" not in resolver.anchor_map


def test_ocr_prefix_twin_and_roster_confirmed_dropped_first_name(resolver):
    assert resolver.anchor_map.get("Ut. Sait Güven") == "Sait Güven"
    assert resolver.anchor_map.get("Ersan Saner") == "Hamza Ersan Saner"


def test_protected_denktas_family_never_fuzzy_merged(resolver):
    assert "Raif R. Denktaş" not in resolver.anchor_map
    assert "Rauf Raif Denktaş" not in resolver.anchor_map
    out = _r(resolver, "Raıf R. Denktas", year=1980)  # OCR twin of one of them: leave it alone
    assert out is None or out[0] == "Raif R. Denktaş"
    assert _r(resolver, "Rauf R. Denktaş", year=1980) is None


# ---------------------------------------------------------------- multi-person labels
def test_two_clerks_on_one_label_fall_back_to_role(resolver):
    assert _r(resolver, "Kutlu Evren, Ünal Üstel", role="katip", year=1993) == ("Katip", "multi_person")
    assert _r(resolver, "Kutlu Evren, Ünal Üstel", role="milletvekili_bakan", year=1993) is None


def test_ministry_title_with_commas_is_not_multi_person(resolver):
    out = _r(resolver, "İçişleri, Köy İşleri ve Doğal Kaynaklar Bakanı Derviş Eroğlu", year=1987)
    assert out is not None and out[0] == "Derviş Eroğlu"


def test_compound_name_with_other_surname_is_left_alone(resolver):
    # "Ahmet Yusuf" window matches exactly, but "Atamsoy" is another anchor's surname
    assert _r(resolver, "Ahmet Yusuf Atamsoy", year=1980) is None


# ---------------------------------------------------------------- role titles
@pytest.mark.parametrize(
    "label,expected",
    [("Îkînci Başkan", "İkinci Başkan"), ("İkinci Başkam", "İkinci Başkan"), ("Pleclis Başkan Vekili", "Başkan Vekili"), ("Geçîcî Katipler", "Geçici Katip")],
)
def test_role_title_variants_fold_to_one_label(resolver, label, expected):
    assert _r(resolver, label, role="baskan", year=1975) == (expected, "role_title")


def test_bakan_is_not_an_ocr_variant_of_baskan(resolver):
    out = _r(resolver, "Bakan", year=1990)
    assert out is None or out[0] != "Meclis Başkanı"


# ---------------------------------------------------------------- bare surnames
def test_bare_surname_needs_unique_active_anchor(resolver):
    assert _r(resolver, "Coşar", year=1990) == ("Salih Coşar", "surname_temporal")
    assert _r(resolver, "Talat", year=1999) == ("Mehmet Ali Talat", "surname_temporal")
    assert _r(resolver, "Talat", year=1985) is None  # Naci Talat Usar's surname is Usar


def test_bare_token_that_is_also_an_active_first_name_is_ambiguous(resolver):
    # "Tahsin" is Özel Tahsin's surname but Tahsin Ertuğruloğlu's first name in 1995
    assert _r(resolver, "Tahsin", year=1995) is None


# ---------------------------------------------------------------- core anchors & noise
def test_title_wrapped_identity_becomes_anchor(resolver):
    assert "Ümit Süleyman" in resolver.anchors
    assert _r(resolver, "İkinci Başkan Vekili Ümit Süleyman", role="baskan", year=1975)[0] == "Ümit Süleyman"
    assert _r(resolver, "Ümit Süleyman Lefkoşa)", year=1976)[0] == "Ümit Süleyman"


def test_noise_labels_stay_unverified(resolver):
    assert _r(resolver, "Konuşmaya Geçelim", year=1978) is None
    assert "Tesbit Edilemeyen Bir Milletvekili" not in resolver.verified_names()
    assert "Konuşmaya Geçelim" not in resolver.verified_names()


def test_non_person_roles_are_never_touched(resolver):
    assert resolver.resolve("Başbakandr . Dervişeroğlu", "diger", 1988) is None


# ---------------------------------------------------------------- record helpers
def test_apply_to_records_keeps_original_label_for_audit():
    recs = []
    for i in range(150):
        recs.append({"speaker_name": "Derviş Eroğlu", "speaker_role": "milletvekili_bakan", "tarih": "1988-03-01", "doc_id": f"d{i % 10}"})
    recs.append({"speaker_name": "Başbakandr . Dervişeroğlu", "speaker_role": "milletvekili_bakan", "tarih": "1988-05-02", "doc_id": "x", "canon_method": "exact"})
    rep = resolve_records(recs, rosters={})
    last = recs[-1]
    assert last["speaker_name"] == "Derviş Eroğlu"
    assert last["speaker_name_parsed"] == "Başbakandr . Dervişeroğlu"
    assert last["canon_method"] == "anchor_exact"
    assert rep["method_counts"] == {"anchor_exact": 1}


def test_search_aliases_are_title_free_cores():
    from speaker_index import core_alias

    assert core_alias("Başbakan Dr. Derviş Eroglu") == "dervis eroglu"
    assert core_alias("İkinci Başkan Vekili Ümit Süleyman") == "umit suleyman"
    assert core_alias("Eroğlu") is None  # a bare surname is not a safe alias
    assert alias_folds("Derviş Eroğlu", ["Derviş Eroğlu", "Maliye Bakanı Derviş Eroglu"]) == []  # folds equal base
    assert fold("Îkînci Başkan") == "ikinci baskan"


# ---------------------------------------------------------------- end-to-end on Parquet
def _write_fixture(tmp: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    docs = {}

    def add(name, role, tarih, doc, n, parsed=None):
        for i in range(n):
            rows.append({
                "speech_id": f"{doc}-{len(rows)}", "doc_id": doc, "meclis": "KKTC", "donem": 1, "yasama_yili": 1,
                "birlesim": 1, "oturum": 1, "tarih": tarih, "turn_index": len(rows), "speaker_name": name,
                "speaker_name_parsed": parsed or name, "speaker_raw": name, "speaker_role": role,
                "speaker_title": None, "constituency": None, "modifiers": None, "canon_method": "exact",
                "text": "Teşekkür ederim Sayın Başkan.", "char_count": 29, "word_count": 4,
                "text_source": "pdf_digital", "source_file": doc, "source_url": f"https://example.test/{doc}",
            })
        docs[doc] = {"doc_id": doc, "meclis": "KKTC", "donem": 1, "yasama_yili": 1, "birlesim": 1, "tarih": tarih,
                     "text_source": "pdf_digital", "source_url": f"https://example.test/{doc}", "konusmacilar": []}

    for y in range(1986, 1990):
        add("Derviş Eroğlu", "milletvekili_bakan", f"{y}-02-01", f"D{y}", 60)
        add("Hakkı Atun", "baskan", f"{y}-02-01", f"D{y}", 40)
    add("Başbakandr . Dervişeroğlu", "milletvekili_bakan", "1988-03-01", "D1988b", 3)
    add("Heclis Başkanı Hakkı Atun", "baskan", "1988-03-01", "D1988b", 2)
    add("Kutlu Evren, Ünal Üstel", "katip", "1988-03-01", "D1988b", 2)
    add("Sesler", "diger", "1988-03-01", "D1988b", 1)

    data = tmp / "in"
    data.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), data / "speeches.parquet")
    for d in docs.values():
        d["konusmacilar"] = sorted({r["speaker_name"] for r in rows if r["doc_id"] == d["doc_id"]})
    pq.write_table(pa.Table.from_pylist(list(docs.values())), data / "documents.parquet")
    pq.write_table(pa.Table.from_pylist([{"speaker_name": "x", "speech_count": 1, "word_count": 1, "doc_count": 1}]), data / "speakers.parquet")
    return data, len(rows)


def test_recanonicalize_end_to_end(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("huggingface_hub")
    import recanonicalize

    data, n_rows = _write_fixture(tmp_path)
    out = tmp_path / "out"
    args = recanonicalize.build_argparser().parse_args([
        "--data-dir", str(data), "--out-dir", str(out), "--space-dir", str(out / "space"),
        "--report", str(out / "report.json"), "--rosters", str(tmp_path / "none.json"), "--skip-aux",
    ])
    report = recanonicalize.run(args)

    con = duckdb.connect()
    sp = (out / "data" / "speeches.parquet").as_posix()
    assert con.execute(f"SELECT COUNT(*) FROM read_parquet('{sp}')").fetchone()[0] == n_rows
    names = dict(con.execute(f"SELECT speaker_name, COUNT(*) FROM read_parquet('{sp}') GROUP BY 1").fetchall())
    assert "Başbakandr . Dervişeroğlu" not in names and names["Derviş Eroğlu"] == 243
    assert names["Hakkı Atun"] == 162
    assert names["Katip"] == 2 and "Kutlu Evren, Ünal Üstel" not in names
    assert names["Sesler"] == 1
    # original labels are preserved for audit
    parsed = con.execute(
        f"SELECT DISTINCT speaker_name_parsed FROM read_parquet('{sp}') WHERE speaker_name = 'Derviş Eroğlu' ORDER BY 1"
    ).fetchall()
    assert ("Başbakandr . Dervişeroğlu",) in parsed
    # documents.konusmacilar refreshed
    doc = (out / "data" / "documents.parquet").as_posix()
    ks = con.execute(f"SELECT konusmacilar FROM read_parquet('{doc}') WHERE doc_id = 'D1988b'").fetchone()[0]
    assert "Derviş Eroğlu" in ks and "Başbakandr . Dervişeroğlu" not in ks

    idx = json.loads((out / "space" / "speakers_index.json").read_text(encoding="utf-8"))
    by_name = {e["n"]: e for e in idx["speakers"]}
    assert by_name["Derviş Eroğlu"].get("v") == 1
    assert by_name["Katip"].get("r") == 1
    assert "v" not in by_name["Sesler"] and "r" not in by_name["Sesler"]
    assert idx["stats"]["speakers_verified"] >= 2
    assert report["after"]["verified_turn_share"] > report["before"]["verified_turn_share"]
    assert report["turns_remapped_by_method"]["multi_person"] == 2
