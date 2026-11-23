# KKTC Cumhuriyet Meclisi Tutanakları — Hugging Face Veri Seti & Konuşmacı Bazlı Arama

* **Canlı Web Arama Uygulaması (Hugging Face Space):** https://huggingface.co/spaces/stevevaius/KKTC-Meclis-Tutanak-Arama
* **Açık Veri Seti (Hugging Face Hub):** https://huggingface.co/datasets/stevevaius/KKTC_Meclis_Tutanak

Kuzey Kıbrıs Türk Cumhuriyeti (KKTC) Cumhuriyet Meclisi ve öncül meclislerin (1960–günümüz) resmî tutanak arşivinden (`evrakcm.gov.ct.tr`) tam otomatik olarak indirilen, metne dönüştürülen ve **konuşmacı bazlı (speaker-turn)** ayrıştırılan Hugging Face veri seti ve arama uygulaması.

## Boru Hattı (Pipeline)

1. `python src/crawl_manifest.py`
   SharePoint belge kütüphanesindeki tüm klasörleri (68 klasör, 3.747 belge) tarar ve `data/manifest.jsonl` + `data/manifest_summary.json` dosyalarını üretir.
2. `python src/download_files.py`
   Tüm `.pdf`, `.doc`, `.docx` ve `.rtf` belgelerini kesintisiz/kaldığı yerden devam eden (resumable) yapıda `data/raw/` altına indirir ve SHA-256 doğrulaması yapar.
3. `python src/extract_text.py`
   - `.docx` -> doğrudan OpenXML ayrıştırma
   - `.doc` / `.rtf` -> MS Word COM + saf Python OLE2 (`src/ole_doc_fallback.py`, Word 6/95 & bozuk tablo kurtarma)
   - `.pdf` -> PyMuPDF (`fitz`) metin katmanı çıkarımı ve OCR kalite sınıflandırması (`pdf_digital`, `pdf_embedded_ocr`, `pdf_needs_ocr`)
4. `python src/parse_speakers.py`
   Oturum başlıklarını (`BAŞKAN: ...`, `KATİP: ...`), oturum içi kürsü devirlerini (`Başkanlık Kürsüsünü ... devreder`) ve tüm konuşmacı geçişlerini ayrıştırarak `data/parsed/documents.jsonl` ile `data/parsed/speeches.jsonl` dosyalarını üretir.
5. `python src/build_dataset.py`
   İki aşamalı konuşmacı kanonikleştirme (belge içi soyadı çözümleme + dönem/genel bulanık yazım düzeltme) uygular, Hugging Face uyumlu Parquet dosyalarını (`data/hf_dataset/`) ve yerel SQLite FTS5 arama veritabanını (`data/search.db`) oluşturur.
   Ardından 3. geçiş (`src/speaker_resolution.py`) çalışır: resmî milletvekili listeleri ve güçlü kanıtlı kimlikler üzerinden OCR ile bozulmuş, unvanı bitişik yazılmış veya iki kişiyi birden taşıyan etiketleri zaman penceresi kontrolüyle çözer (ayrıntılar aşağıda).
6. `python src/search_cli.py`
   Konuşmacı adı, anahtar kelime, meclis, dönem ve tarih aralığına göre anlık arama yapar.
7. `python src/upload_hf.py --repo-id <kullanici>/KKTC_Meclis_Tutanak`
   Üretilen veri setini (ve istenirse ham arşivi) doğrudan Hugging Face Hub'a yükler.

## Otomatik Güncelleme (Windows Görev Zamanlayıcı)

[`scripts/run_sync.ps1`](scripts/run_sync.ps1), [`src/sync_incremental.py`](src/sync_incremental.py) betiğini haftada bir (Pazartesi 10:00, bilgisayar kapalıysa açıldığı ilk anda otomatik telafi edecek şekilde `StartWhenAvailable` ile) çalıştırır:
1. `evrakcm.gov.ct.tr`'yi tarar ve Hugging Face'teki mevcut `documents.parquet` listesiyle (~50 KB HTTP range isteğiyle) karşılaştırır.
2. Yeni tutanak yoksa ~9 saniyede çıkar.
3. Yeni tutanak varsa yalnızca yenileri indirip işler, Parquet dosyalarını yeniden sıralayıp hem Hugging Face Dataset'e hem Static Space (`speakers_index.json`) deposuna yükler.

## Hızlı Arama Örnekleri

```bash
# Konuşmacı ara
python src/search_cli.py --list-speakers "Erhürman"

# Belirli bir konuşmacının belirli bir konudaki sözlerini ara
python src/search_cli.py --speaker "Tufan Erhürman" --query "elektrik OR kib-tek" -n 10

# Tarih ve dönem filtresiyle arama
python src/search_cli.py --speaker "Derviş Eroğlu" --from-date 1985-01-01 --to-date 1995-12-31 --query "anayasa"
```

## Konuşmacı Kimlik Çözümleme (3. Geçiş)

1960–1995 arası taranmış tutanaklarda OCR aynı kişiyi onlarca farklı yazımla üretir (`Başbakandr . Dervişeroğlu`, `Barbakan Dr Derviş Eroglu`, `Hakkı Attın` …) ve bazı etiketler iki kişiyi birden taşır (`Kutlu Evren, Ünal Üstel`). [`src/speaker_resolution.py`](src/speaker_resolution.py) bunları **önce doğruluk** ilkesiyle çözer:

* **Çapalar (anchor):** `data/official_rosters.json` üyeleri + en az 5 belgede tekrarlanan temiz adlar (≥100 söz, ya da 1983 öncesi unvan içi kimlikler için ≥40 söz). Tek oturumluk OCR çöpleri ve iki kişilik etiketler (`Mustafa Akıncı Hüseyin Özgürgün`) çapa olamaz.
* **Bölünmüş kimlik birleştirme:** baş harf farkı (`Osman N. Örek` = `Osman Örek`), eksik soyad (`Ümit Süleyman` → `Ümit Süleyman Onan`), kesik soyad (`Mustafa Hacıahmet` → `Mustafa Hacıahmetoğlu`), resmî listeyle doğrulanan düşmüş ilk ad (`Ersan Saner` → `Hamza Ersan Saner`) ve ilk harf hatası dahil 20 kat güçlü bir kimliğin OCR ikizi (`Haci T. Usar` → `Naci Talat Usar`, `Czker Özgür` → `Özker Özgür`). Çelişen baş harfler (`Mehmet A.` / `Mehmet B.`), kendi başına soyad olan kesikler (`Öztürk` ≠ `Öztürkler`) ve Denktaş ailesi gibi korunan soyadları asla birleştirilmez.
* **Kurtarma:** unvan/önek/bitişik yazım temizlenir; eşleşme ancak tek bir çapaya net farkla uyuyorsa **ve** o kişi ilgili yılda ±2 aktifse uygulanır.
* **Çok kişili etiketler** hiçbir kişiye atanmaz; başkan/katip sözleri genel makam etiketine düşer.

Yayındaki veri setine ham arşiv olmadan uygulamak için:

```bash
pip install -r requirements.txt                                   # rapidfuzz dahil (yoksa ~100x yavaş difflib yedeği)
python src/recanonicalize.py --from-hub                           # kuru çalıştırma: data/recanon/ + space_static/*.json + rapor
python src/recanonicalize.py --from-hub --upload --upload-ui      # rapor uygunsa Dataset + Space (indeksler + arayüz) tek seferde
```

Rapordaki (`data/parsed/speaker_resolution_report.json`) her eşleme (`from` → `to`, yöntem, söz sayısı) incelenebilir; ham etiket her zaman `speaker_name_parsed` sütununda kalır.

**Deploy güvenliği:** Space indeksleri her zaman Parquet ile birlikte üretilmelidir. `recanonicalize.py` indeksleri varsayılan olarak `space_static/` içine yazar (repo kopyası eskimez), `--upload --skip-aux` birleşimini reddeder ve Space dosyalarını tek commit'te yükler. `src/deploy_space.py` artık varsayılan olarak yalnızca `index.html` + `README.md` yükler; `--with-indices` verilirse yerel `speakers_index.json` canlıdakinden eskiyse yüklemeyi reddeder.

**Arayüz:** `speakers_index.json` girdileri `v` (doğrulanmış kişi), `r` (makam etiketi) ve `p` (eşleşmeyen ama temiz, nadir gerçek isim; ör. `Fazıl Küçük`, `Emine Çolak`) bayraklarını taşır. Otomatik tamamlama bu üç grubu normal listeler; hiçbir bayrağı olmayan etiketler gizlenmez, en fazla 4 tanesi en altta soluk ve "eşleştirilmemiş" rozetiyle gösterilir.

Testler: `python -m pytest tests/test_speaker_resolution.py tests/test_parser.py tests/test_topics.py`
