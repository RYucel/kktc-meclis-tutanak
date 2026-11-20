"""Analyze parliamentary topics & synonym-unified keywords per legislative term (1960–2026).

Uses word-boundary RE2 regexes in a single vectorized multi-threaded DuckDB SQL scan so:
  1. Substring collisions are impossible (e.g. 'tarım' NEVER matches 'tasarım',
     'orman' NEVER matches 'performans'/'Borman', 'Aksa' NEVER matches 'maksat'/'baksak',
     'atık' NEVER matches 'demokratik'/'pratik', 'iskan' NEVER matches 'kıskanmak').
  2. Synonyms & OCR/hyphenation variants are cleanly unified
     (e.g. KIB-TEK = KIBTEK = KIPTEK = KIB TEK = Elektrik Kurumu).
  3. Exact RE2 regex `rx` is stored in `space_static/topics_index.json` and executed
     identically by DuckDB-WASM in the browser, guaranteeing badge_count === search_count.
"""

import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb

HF_DATA = Path("data/hf_dataset/data")
STATIC_SPACE = Path("space_static")
STATIC_SPACE.mkdir(parents=True, exist_ok=True)

try:
    from optimize_for_wasm import TERM_CATALOG
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from optimize_for_wasm import TERM_CATALOG


def norm_tr_i(s: str) -> str:
    """Exact equivalent of DuckDB `replace(lower(text), 'ı', 'i')`."""
    return s.replace("İ", "i").replace("I", "i").replace("ı", "i").lower().replace("\u0307", "")


# Word-boundary helpers for Turkish RE2 regexes (applied on `replace(lower(text), 'ı', 'i')`)
_WB_L = "(^|[^a-zçğöşü])"
_WB_R = "($|[^a-zçğöşü])"

# Short/ambiguous words requiring BOTH left AND right word boundaries (allowing Turkish apostrophe suffixes like Aksa'ya, KDV'nin)
_EXACT_BOUNDARY_WORDS = {
    "aksa", "kdv", "daü", "kthy", "brtk", "brt", "tmk", "isias",
    "atik", "çöp", "otel", "aşi", "yödak",
}


def build_re2_pattern(terms: list[str]) -> str:
    """Build a single RE2 regex with Turkish word-start boundaries (and full boundaries for short/exact tokens)."""
    alts = []
    for raw in terms:
        p = norm_tr_i(raw)
        esc = re.escape(p)
        if p in _EXACT_BOUNDARY_WORDS:
            if p == "atik":
                alts.append(f"{esc}(lar|lari|larin|su|suyu)?(?=$|[^a-zçğöşü])")
            elif p == "otel":
                alts.append(f"{esc}(ler|leri|lerin|cilik|ci)?(?=$|[^a-zçğöşü])")
            elif p == "çöp":
                alts.append(f"{esc}(ler|leri|lük)?(?=$|[^a-zçğöşü])")
            elif p == "aşi":
                alts.append(f"{esc}(lar|lari|lama|lanm)?(?=$|[^a-zçğöşü])")
            else:
                alts.append(f"{esc}(?=$|[^a-zçğöşü])")
        else:
            alts.append(esc)
    # Note: RE2 does not support lookahead (?=...), so encode right boundary cleanly without lookahead!
    prefix_alts = []
    exact_alts = []
    for raw in terms:
        p = norm_tr_i(raw)
        esc = re.escape(p)
        if p in _EXACT_BOUNDARY_WORDS:
            if p == "atik":
                exact_alts.append(f"{esc}(lar|lari|larin|su|suyu)?")
            elif p == "otel":
                exact_alts.append(f"{esc}(ler|leri|lerin|cilik|ci)?")
            elif p == "çöp":
                exact_alts.append(f"{esc}(ler|leri|lük)?")
            elif p == "aşi":
                exact_alts.append(f"{esc}(lar|lari|lama|lanm)?")
            else:
                exact_alts.append(esc)
        else:
            prefix_alts.append(esc)

    parts = []
    if prefix_alts:
        parts.append(f"{_WB_L}({'|'.join(prefix_alts)})")
    if exact_alts:
        parts.append(f"{_WB_L}({'|'.join(exact_alts)}){_WB_R}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# 1. 20 Curated TRNC Parliamentary Policy Topics (synonym-unified, word-bounded)
# ---------------------------------------------------------------------------
CURATED_TOPICS = [
    {
        "id": "kibtek_elektrik",
        "label": "KIB-TEK & Elektrik / Enerji",
        "note": "KIB-TEK, KIBTEK, KIPTEK, Elektrik Kurumu, elektrik, Teknecik, Aksa, akaryakıt, santral",
        "patterns": [
            "kib-tek", "kibtek", "kiptek", "kib tek", "kip tek",
            "elektrik kurumu", "elektrik", "teknecik", "aksa",
            "akaryakit", "santral", "güneş enerj",
        ],
    },
    {
        "id": "enflasyon_pahalilik",
        "label": "Enflasyon & Hayat Pahalılığı",
        "note": "enflasyon, hayat pahalılığı, pahalılık, alım gücü, geçim sıkıntısı, zamlar",
        "patterns": [
            "enflasyon", "hayat pahal", "pahalilik", "alim güc",
            "geçim sikint", "zamlar", "fiyat artiş",
        ],
    },
    {
        "id": "maas_ucret",
        "label": "Maaş, Asgari Ücret & Emekli",
        "note": "maaş, asgari ücret, eşel mobil, emekli, ek mesai, barem, ücret artışı",
        "patterns": [
            "maaş", "asgari ücret", "eşel mobil", "emekli",
            "ek mesai", "barem", "ücret artiş", "ek ödenek",
        ],
    },
    {
        "id": "butce_maliye",
        "label": "Bütçe, Vergi, Döviz & Borç",
        "note": "bütçe, vergi, KDV, döviz, sterlin, Türk Lirası, borçlanma, hazine, mali protokol",
        "patterns": [
            "bütçe", "vergi", "döviz", "sterlin", "türk liras",
            "mali protokol", "kdv", "cari açik", "borçlanm", "hazine",
        ],
    },
    {
        "id": "kibris_sorunu",
        "label": "Kıbrıs Sorunu, Müzakere & Maraş",
        "note": "Kıbrıs sorunu, federasyon, müzakere, Annan, Crans-Montana, Maraş, iki devlet, garantör",
        "patterns": [
            "kibris sorun", "kibris meseles", "federasyon", "konfederasyon",
            "müzakere", "annan", "crans", "maraş", "iki devlet",
            "egemen eşit", "garantör", "rum taraf", "rum yönetim",
        ],
    },
    {
        "id": "iskan_mulkiyet",
        "label": "İskan, Eşdeğer & Mülkiyet",
        "note": "iskan, eşdeğer, mülkiyet, Taşınmaz Mal Komisyonu, tahsis, koçan, tapu, göçmen",
        "patterns": [
            "iskan", "eşdeğer", "mülkiyet", "taşinmaz mal",
            "tahsis", "koçan", "tapu", "göçmen",
        ],
    },
    {
        "id": "saglik_pandemi",
        "label": "Sağlık, Hastane & Pandemi",
        "note": "hastane, sağlık bakanlığı, doktor, hekim, ilaç, pandemi, Covid, korona, karantina, aşı",
        "patterns": [
            "hastane", "sağlik bakan", "doktor", "hekim", "ilaç",
            "pandemi", "koronavir", "covid", "karantina", "aşi", "hemşire", "acil servis",
        ],
    },
    {
        "id": "egitim_universite",
        "label": "Eğitim, Okul & Üniversiteler",
        "note": "eğitim, üniversite, DAÜ, YÖDAK, öğretmen, öğrenci, okul, sahte diploma",
        "patterns": [
            "eğitim", "üniversite", "öğretmen", "öğrenci", "okul",
            "doğu akdeniz üniversite", "daü", "yödak", "sahte diploma", "müfredat",
        ],
    },
    {
        "id": "turizm_havacilik",
        "label": "Turizm, Ercan & KTHY",
        "note": "turizm, otel, Ercan Havalimanı, KTHY, Kıbrıs Türk Hava Yolları, kumarhane/casino, uçuş",
        "patterns": [
            "turizm", "otel", "ercan", "kthy", "hava yollar",
            "havayollar", "kumarhane", "casino", "uçuş", "turist",
        ],
    },
    {
        "id": "tarim_su_hellim",
        "label": "Tarım, Hayvancılık, Hellim & Su",
        "note": "tarım, hayvancılık, çiftçi, hellim, narenciye, SÜTEK, TÜK, Cypruvex, kuraklık, su projesi",
        "patterns": [
            "tarim", "hayvancilik", "çiftçi", "hellim", "narenciye",
            "sütek", "süt kurumu", "toprak ürünleri", "cypruvex",
            "kuraklik", "su proje", "içme suyu", "sulama",
        ],
    },
    {
        "id": "belediye_altyapi",
        "label": "Belediye, İmar, Yol & Trafik",
        "note": "belediye, yerel yönetim, imar planı, altyapı, karayolları, trafik, kanalizasyon",
        "patterns": [
            "belediye", "yerel yönetim", "imar plan", "altyapi",
            "karayollar", "trafik", "otoyol", "kanalizasyon",
        ],
    },
    {
        "id": "guvenlik_yargi",
        "label": "Polis, Yargı & Hukuk",
        "note": "polis, mahkeme, yargı, savcılık, Anayasa Mahkemesi, uyuşturucu, cezaevi, Güvenlik Kuvvetleri",
        "patterns": [
            "polis", "mahkeme", "yargi", "savcilik", "anayasa mahkeme",
            "uyuşturucu", "cezaevi", "güvenlik kuvvet",
        ],
    },
    {
        "id": "anayasa_secim",
        "label": "Anayasa, Seçim & Demokrasi",
        "note": "anayasa, seçim yasası, genel/yerel/ara seçim, YSK, dokunulmazlık, içtüzük, referandum",
        "patterns": [
            "anayasa", "seçim yasa", "genel seçim", "yerel seçim",
            "ara seçim", "yüksek seçim", "dokunulmazlik", "içtüzük", "referandum",
        ],
    },
    {
        "id": "nufus_vatandaslik",
        "label": "Nüfus, Vatandaşlık & Çalışma İzni",
        "note": "vatandaşlık, muhaceret, çalışma izni, nüfus sayımı, ikamet izni, yabancı/kaçak işçi",
        "patterns": [
            "vatandaşlik", "muhaceret", "çalişma izni", "nüfus sayim",
            "ikamet izni", "yabanci işçi", "kaçak işçi",
        ],
    },
    {
        "id": "turkiye_ab",
        "label": "Türkiye Protokolleri, AB & İzolasyon",
        "note": "Türkiye Cumhuriyeti, anavatan, ekonomik protokol, Avrupa Birliği, ambargo, izolasyon, Yeşil Hat",
        "patterns": [
            "türkiye cumhuriyet", "anavatan", "ekonomik protokol", "mali işbirliğ",
            "avrupa birliğ", "ambargo", "izolasyon", "yeşil hat",
        ],
    },
    {
        "id": "denetim_ihale",
        "label": "İhale, Sayıştay, Yolsuzluk & Özelleştirme",
        "note": "ihale, Sayıştay, yolsuzluk, usulsüzlük, özelleştirme, rüşvet, Meclis araştırması/soruşturması",
        "patterns": [
            "ihale", "sayiştay", "yolsuzluk", "usulsüzlük", "özelleştir",
            "rüşvet", "denetim rapor", "meclis araştirma", "meclis soruşturma",
        ],
    },
    {
        "id": "sendika_istihdam",
        "label": "Sendika, Grev, İşçi & İstihdam",
        "note": "sendika, grev, işsizlik, istihdam, Sosyal Sigortalar, İhtiyat Sandığı, işçi",
        "patterns": [
            "sendika", "grev", "işsizlik", "istihdam",
            "sosyal sigorta", "ihtiyat sandiğ", "işçi", "kamu çalişan",
        ],
    },
    {
        "id": "banka_kooperatif",
        "label": "Banka, Kooperatif, Kredi & Faiz",
        "note": "banka, kooperatif, Merkez Bankası, Kalkınma Bankası, kredi, faiz, tasarruf mevduatı",
        "patterns": [
            "banka", "kooperatif", "merkez bankas", "kalkinma bankas",
            "kredi", "faiz", "tasarruf mevduat",
        ],
    },
    {
        "id": "cevre_orman",
        "label": "Çevre, Orman, Deniz & Atık",
        "note": "çevre koruma/kirliliği/dairesi, ekoloji, orman, yangın, çöp, atık su, katı atık, sahil, taş ocağı",
        "patterns": [
            "çevre koruma", "çevre kirl", "çevre bakan", "çevre dairesi", "çevre yasa",
            "çevre etki", "çevreci", "ekoloji", "orman", "yangin", "kirlilik",
            "çöp", "atik su", "atiksu", "kati atik", "sahil", "kiyi şerid", "taş ocağ",
        ],
    },
    {
        "id": "medya_telekom",
        "label": "Basın, BRT, Telekomünikasyon & Fiber",
        "note": "basın, gazeteci, BRT/BRTK, Bayrak Radyo, telekomünikasyon, Telefon Dairesi, fiber, internet",
        "patterns": [
            "basin", "gazeteci", "brtk", "bayrak radyo",
            "telekomünikasyon", "telefon dairesi", "fiber", "internet", "yayin yüksek",
        ],
    },
]


# ---------------------------------------------------------------------------
# 2. 91 Synonym-Unified Parliamentary Concepts / Keywords (word-bounded)
# ---------------------------------------------------------------------------
UNIFIED_CONCEPTS = [
    ("KIB-TEK / Elektrik Kurumu", ["kib-tek", "kibtek", "kiptek", "kib tek", "kip tek", "elektrik kurumu"], "KIB-TEK, KIBTEK, KIPTEK, Elektrik Kurumu"),
    ("elektrik", ["elektrik"], "elektrik kesintisi, elektrik enerjisi, elektrik faturaları"),
    ("Aksa", ["aksa"], "Aksa Enerji / enerji alım sözleşmeleri"),
    ("Teknecik", ["teknecik"], "Teknecik Elektrik Santrali"),
    ("akaryakıt", ["akaryakit"], "akaryakıt fiyatları, yakıt alımı, fuel-oil"),
    ("enflasyon", ["enflasyon"], "enflasyon oranı ve fiyat artışları"),
    ("hayat pahalılığı", ["hayat pahal", "pahalilik"], "hayat pahalılığı, pahalılık ödeneği"),
    ("maaş", ["maaş"], "kamu ve özel sektör maaşları"),
    ("asgari ücret", ["asgari ücret"], "asgari ücret tespiti ve artışları"),
    ("eşel mobil", ["eşel mobil"], "eşel mobil sistemi"),
    ("emekli", ["emekli"], "emeklilik ve emekli maaşları"),
    ("bütçe", ["bütçe"], "mali yıl bütçesi ve bütçe dengesi"),
    ("vergi", ["vergi"], "vergi yasaları ve vergi gelirleri"),
    ("KDV", ["kdv", "katma değer vergi"], "KDV / Katma Değer Vergisi"),
    ("döviz / sterlin", ["döviz", "sterlin"], "döviz kurları, sterlin ve kur dalgalanmaları"),
    ("ekonomi", ["ekonomi"], "ekonomik politika ve kalkınma"),
    ("ekonomik kriz", ["kriz"], "ekonomik / mali / bankacılık krizi"),
    ("KTHY / Hava Yolları", ["kthy", "kibris türk hava yollar"], "KTHY, Kıbrıs Türk Hava Yolları"),
    ("Ercan Havalimanı", ["ercan"], "Ercan Havalimanı / Havaalanı"),
    ("turizm / otel", ["turizm", "otel"], "turizm sektörü, oteller ve teşvikler"),
    ("casino / kumarhane", ["casino", "kumarhane"], "casinolar, kumarhaneler ve şans oyunları"),
    ("SÜTEK / Süt Kurumu", ["sütek", "süt kurumu", "süt endüstrisi"], "SÜTEK, Süt Endüstrisi Kurumu"),
    ("TÜK / Toprak Ürünleri", ["toprak ürünleri kurumu", "toprak ürünleri"], "TÜK, Toprak Ürünleri Kurumu"),
    ("Cypruvex", ["cypruvex", "sipruveks"], "Cypruvex / narenciye işletmesi"),
    ("hellim", ["hellim"], "hellim üretimi, coğrafi tescil ve ihracat"),
    ("narenciye", ["narenciye"], "narenciye üretimi ve ihracatı"),
    ("tarım / çiftçi", ["tarim", "çiftçi"], "tarım politikaları ve çiftçi destekleri"),
    ("hayvancılık", ["hayvancilik"], "hayvancılık, yem ve süt üreticileri"),
    ("kuraklık", ["kuraklik"], "kuraklık ve kuraklık tazminatları"),
    ("su projesi / içme suyu", ["su proje", "içme suyu", "sulama suyu"], "su temin projesi, içme ve sulama suyu"),
    ("pandemi / Covid-19", ["pandemi", "covid", "koronavir", "korona"], "pandemi, Covid-19, koronavirüs"),
    ("karantina", ["karantina"], "pandemi dönemi karantina uygulamaları"),
    ("aşı", ["aşi"], "aşı temini ve aşılama"),
    ("hastane", ["hastane"], "devlet hastaneleri ve sağlık altyapısı"),
    ("ilaç / doktor", ["ilaç", "doktor", "hekim"], "ilaç temini, doktor ve hekim kadroları"),
    ("deprem / İsias", ["deprem", "isias", "şampiyon melek"], "deprem, bina güvenliği, İsias davası"),
    ("sahte diploma", ["sahte diploma"], "sahte diploma soruşturması ve YÖDAK"),
    ("DAÜ / Doğu Akdeniz Ünv.", ["daü", "doğu akdeniz üniversite"], "DAÜ, Doğu Akdeniz Üniversitesi"),
    ("üniversite / YÖDAK", ["üniversite", "yödak"], "yükseköğretim, üniversiteler ve YÖDAK"),
    ("öğretmen / okul", ["öğretmen", "okul"], "okullar, öğretmen kadroları ve eğitim"),
    ("Kıbrıs sorunu", ["kibris sorun", "kibris meseles"], "Kıbrıs sorunu / meselesi"),
    ("federasyon", ["federasyon"], "federasyon / federal çözüm tartışmaları"),
    ("iki devlet / egemenlik", ["iki devlet", "egemen eşit", "egemenlik"], "iki devletli çözüm ve egemen eşitlik"),
    ("müzakere", ["müzakere"], "Kıbrıs müzakereleri ve toplumlararası görüşmeler"),
    ("Annan Planı", ["annan"], "Annan Planı ve 2004 referandumu"),
    ("Crans-Montana", ["crans"], "Crans-Montana konferansı"),
    ("Maraş / Kapalı Maraş", ["maraş"], "Kapalı Maraş açılımı ve Maraş bölgesi"),
    ("Yeşil Hat", ["yeşil hat"], "Yeşil Hat Tüzüğü ve ticareti"),
    ("Avrupa Birliği", ["avrupa birliğ"], "Avrupa Birliği ilişkileri ve uyum yasaları"),
    ("Birleşmiş Milletler", ["birleşmiş milletler"], "BM kararları, Barış Gücü ve Genel Sekreterlik"),
    ("ambargo / izolasyon", ["ambargo", "izolasyon"], "ambargolar ve uluslararası izolasyonlar"),
    ("garantör / Türkiye protokolü", ["garantör", "ekonomik protokol", "mali işbirliğ"], "garantörlük, Türkiye mali/ekonomik protokolleri"),
    ("iskan", ["iskan"], "iskan politikaları, iskan ve rehabilitasyon"),
    ("eşdeğer", ["eşdeğer"], "eşdeğer puan ve mal tazminatları"),
    ("Taşınmaz Mal Kom.", ["taşinmaz mal komisyon", "taşinmaz mal"], "Taşınmaz Mal Komisyonu ve taşınmaz mülkiyet"),
    ("koçan / tapu", ["koçan", "tapu"], "tapu, koçan ve mülkiyet belgeleri"),
    ("göçmen", ["göçmen"], "göçmen yurttaşlar ve yerleşim hakları"),
    ("belediye", ["belediye"], "belediyeler, belediye reformu ve yerel yönetimler"),
    ("imar planı", ["imar plan"], "imar planları ve şehir planlama"),
    ("trafik / karayolları", ["trafik", "karayollar"], "trafik güvenliği ve karayolları"),
    ("polis", ["polis"], "Polis Örgütü ve asayiş"),
    ("yargı / mahkeme", ["yargi", "mahkeme", "anayasa mahkeme"], "yargı bağımsızlığı, mahkemeler ve hukuk"),
    ("uyuşturucu", ["uyuşturucu"], "uyuşturucu ile mücadele"),
    ("cezaevi", ["cezaevi"], "Merkezi Cezaevi ve ıslah düzenlemeleri"),
    ("anayasa", ["anayasa"], "Anayasa değişiklikleri ve anayasal düzen"),
    ("seçim / referandum", ["seçim yasa", "genel seçim", "yerel seçim", "referandum"], "seçim yasaları, seçimler ve referandum"),
    ("dokunulmazlık", ["dokunulmazlik"], "yasama dokunulmazlığı"),
    ("vatandaşlık", ["vatandaşlik"], "KKTC vatandaşlığı ve yurttaşlık yasası"),
    ("çalışma izni / muhaceret", ["çalişma izni", "muhaceret", "ikamet izni"], "çalışma izinleri, muhaceret ve yabancı işgücü"),
    ("ihale", ["ihale"], "kamu ihaleleri ve ihale yasası"),
    ("Sayıştay", ["sayiştay"], "Sayıştay denetimi ve raporları"),
    ("yolsuzluk / usulsüzlük", ["yolsuzluk", "usulsüzlük", "rüşvet"], "yolsuzluk, usulsüzlük ve denetim"),
    ("özelleştirme", ["özelleştir"], "kamu iktisadi teşebbüsleri ve özelleştirme"),
    ("sendika / grev", ["sendika", "grev"], "sendikalar, grevler ve çalışma hayatı"),
    ("işsizlik / istihdam", ["işsizlik", "istihdam"], "işsizlik, istihdam ve işgücü"),
    ("Sosyal Sigortalar", ["sosyal sigorta"], "Sosyal Sigortalar Dairesi ve primler"),
    ("İhtiyat Sandığı", ["ihtiyat sandiğ"], "İhtiyat Sandığı"),
    ("banka / Merkez Bankası", ["banka", "merkez bankas"], "bankacılık sektörü ve Merkez Bankası"),
    ("kooperatif", ["kooperatif"], "kooperatifler ve Koop-Bank"),
    ("kredi / faiz", ["kredi", "faiz"], "krediler, faiz oranları ve borç yapılandırma"),
    ("çevre / kirlilik", ["çevre koruma", "çevre kirl", "çevre bakan", "çevre dairesi", "çevre yasa", "çevreci", "ekoloji", "kirlilik", "atik su", "atiksu", "kati atik", "çöp"], "çevre koruma, çevre kirliliği, ekoloji, çöp, atık su, katı atık"),
    ("orman / yangın", ["orman", "yangin"], "orman alanları ve orman yangınları"),
    ("BRT / BRTK", ["brtk", "bayrak radyo"], "BRT, BRTK, Bayrak Radyo Televizyon Kurumu"),
    ("basın / medya", ["basin", "gazeteci"], "basın özgürlüğü, gazeteciler ve medya"),
    ("fiber / telekomünikasyon", ["fiber", "telekomünikasyon", "telefon dairesi", "internet"], "fiber optik, Telefon Dairesi, telekomünikasyon"),
    ("Vakıflar / Evkaf", ["vakiflar", "evkaf"], "Vakıflar İdaresi ve Evkaf malları"),
    ("mücahit", ["mücahit"], "mücahitler, mücahit tazminat ve hakları"),
    ("cemaat meclisi / cemaat", ["cemaat"], "Türk Cemaat Meclisi ve cemaat işleri"),
    ("kadın hakları", ["kadin hak", "kadina şiddet", "toplumsal cinsiyet"], "kadın hakları ve toplumsal cinsiyet eşitliği"),
    ("gençlik / spor", ["gençlik", "spor"], "gençlik politikaları ve spor"),
    ("ihracat / gümrük", ["ihracat", "ithalat", "gümrük"], "dış ticaret, ihracat, ithalat ve gümrük"),
    ("liman / denizcilik", ["liman", "denizcilik"], "limanlar (Gazimağusa, Girne) ve deniz taşımacılığı"),
]


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_topics_index(con: duckdb.DuckDBPyConnection, sp_path: str) -> dict:
    """Single vectorized multi-threaded DuckDB SQL scan using word-bounded RE2 regexes."""
    t0 = time.perf_counter()

    topic_rx_list = [build_re2_pattern(tp["patterns"]) for tp in CURATED_TOPICS]
    concept_rx_list = [build_re2_pattern(pats) for _, pats, _ in UNIFIED_CONCEPTS]

    agg_cols = ["COUNT(*) AS turns"]
    for i, rx in enumerate(topic_rx_list):
        agg_cols.append(f"COUNT(*) FILTER (WHERE regexp_matches(nt, {_sql_quote(rx)})) AS tp_{i}")
    for j, rx in enumerate(concept_rx_list):
        agg_cols.append(f"COUNT(*) FILTER (WHERE regexp_matches(nt, {_sql_quote(rx)})) AS cp_{j}")

    sql = f"""
        WITH norm AS (
            SELECT meclis, donem, replace(lower(text), 'ı', 'i') AS nt
            FROM read_parquet('{sp_path}')
            WHERE text IS NOT NULL AND length(text) > 0
        )
        SELECT meclis, donem, {', '.join(agg_cols)}
        FROM norm
        GROUP BY 1, 2
    """
    rows = con.execute(sql).fetchall()

    n_topics = len(CURATED_TOPICS)
    n_concepts = len(UNIFIED_CONCEPTS)

    term_total_turns: Counter = Counter()
    term_topic_counts: dict[str, dict[str, int]] = defaultdict(dict)
    term_concept_counts: dict[str, dict[int, int]] = defaultdict(dict)
    global_topic_counts: Counter = Counter()
    global_concept_counts: Counter = Counter()
    total_all_turns = 0

    for row in rows:
        meclis, donem, turns = row[0], row[1], int(row[2])
        tid = f"{meclis}:{donem}" if donem is not None else str(meclis)
        term_total_turns[tid] = turns
        total_all_turns += turns
        for i, tp in enumerate(CURATED_TOPICS):
            c = int(row[3 + i])
            term_topic_counts[tid][tp["id"]] = c
            global_topic_counts[tp["id"]] += c
        for j in range(n_concepts):
            c = int(row[3 + n_topics + j])
            if c > 0:
                term_concept_counts[tid][j] = c
                global_concept_counts[j] += c

    # False-positive sanity assertions!
    aksa_idx = next(i for i, c in enumerate(UNIFIED_CONCEPTS) if c[0] == "Aksa")
    assert term_concept_counts["KC"].get(aksa_idx, 0) == 0, "False positive: Aksa matched in 1960!"
    assert term_concept_counts["KTY:2"].get(aksa_idx, 0) == 0, "False positive: Aksa matched in 1970-75!"
    cevre_idx = next(i for i, c in enumerate(UNIFIED_CONCEPTS) if c[0] == "çevre / kirlilik")
    assert term_concept_counts["KC"].get(cevre_idx, 0) < 25, (
        f"False positive: çevre/atık matched {term_concept_counts['KC'].get(cevre_idx, 0)} in 1960!"
    )

    print(
        f"   Vectorized word-bounded DuckDB scan complete: {total_all_turns:,} turns, "
        f"{n_topics} topics, {n_concepts} unified concepts in {time.perf_counter()-t0:.1f}s "
        f"(false-positive sanity checks PASSED!)"
    )

    topics_meta = [
        {
            "id": tp["id"],
            "label": tp["label"],
            "note": tp["note"],
            "terms": [norm_tr_i(p) for p in tp["patterns"]],
            "rx": topic_rx_list[i],
            "total": int(global_topic_counts[tp["id"]]),
        }
        for i, tp in enumerate(CURATED_TOPICS)
    ]

    concepts_meta = [
        {
            "id": idx,
            "label": lbl,
            "note": note,
            "terms": [norm_tr_i(p) for p in pats],
            "rx": concept_rx_list[idx],
            "total": int(global_concept_counts[idx]),
        }
        for idx, (lbl, pats, note) in enumerate(UNIFIED_CONCEPTS)
    ]

    terms_out = []
    for meclis, donem, label, years, group_label in TERM_CATALOG:
        tid = f"{meclis}:{donem}" if donem is not None else meclis
        tot = int(term_total_turns.get(tid, 0))
        if tot == 0:
            continue

        t_list = []
        for tp in CURATED_TOPICS:
            cnt = int(term_topic_counts[tid].get(tp["id"], 0))
            pct = round((cnt / tot) * 100.0, 2) if tot else 0.0
            t_list.append({"id": tp["id"], "c": cnt, "p": pct})
        t_list.sort(key=lambda x: -x["c"])

        c_scores = []
        cc_map = {}
        for idx in range(n_concepts):
            cnt = int(term_concept_counts[tid].get(idx, 0))
            if cnt == 0:
                continue
            cc_map[str(idx)] = cnt
            term_rate = cnt / tot
            lift = ((cnt + 3.0) / (tot + 300.0)) / ((global_concept_counts[idx] + 20.0) / (total_all_turns + 2000.0))
            c_scores.append((idx, cnt, round(term_rate * 100.0, 2), round(lift, 2)))

        top_freq = [
            {"id": idx, "c": cnt, "p": pct, "l": lift}
            for idx, cnt, pct, lift in sorted(c_scores, key=lambda x: -x[1])[:22]
        ]

        min_dist_cnt = 2 if tot < 500 else (5 if tot < 10000 else 15)
        distinctive = [
            {"id": idx, "c": cnt, "p": pct, "l": lift}
            for idx, cnt, pct, lift in sorted(c_scores, key=lambda x: (-x[3], -x[1]))
            if cnt >= min_dist_cnt and lift >= 1.30
        ][:14]

        terms_out.append(
            {
                "id": tid,
                "meclis": meclis,
                "donem": donem,
                "label": label,
                "years": years,
                "group": group_label,
                "turns": tot,
                "topics": t_list,
                "top_concepts": top_freq,
                "distinctive": distinctive,
                "cc": cc_map,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_turns": total_all_turns,
        "topics": topics_meta,
        "concepts": concepts_meta,
        "terms": terms_out,
    }


def main():
    sp_path = (HF_DATA / "speeches.parquet").as_posix()
    con = duckdb.connect(":memory:")
    print(f"Analyzing word-bounded topics & unified keywords across {sp_path}...")
    payload = build_topics_index(con, sp_path)
    out_path = STATIC_SPACE / "topics_index.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, "
        f"{len(payload['topics'])} topics, {len(payload['concepts'])} unified concepts, "
        f"{len(payload['terms'])} terms)"
    )

    c_by_id = {c["id"]: c["label"] for c in payload["concepts"]}
    print("\n=== Word-Bounded Distinctive Keywords Audit by Era ===")
    for t in payload["terms"]:
        if t["id"] in ("KKTC:10", "KKTC:9", "KKTC:6", "KKTC:4", "KKTC:1", "KTFD:1", "KTY:2", "KC"):
            top5 = [f"{c_by_id[x['id']]}({x['c']})" for x in t["top_concepts"][:5]]
            dist5 = [f"{c_by_id[x['id']]}({x['c']}, {x['l']}x)" for x in t["distinctive"][:6]]
            print(f"  {t['id']:8s} ({t['years']:9s}):")
            print(f"    Top:  {', '.join(top5)}")
            print(f"    Özgü: {', '.join(dist5)}")


if __name__ == "__main__":
    main()
