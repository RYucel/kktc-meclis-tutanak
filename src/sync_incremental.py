"""Stateless incremental sync for KKTC Meclisi minutes (designed for GitHub Actions & local cron).

1. Crawls live SharePoint archive (evrakcm.gov.ct.tr) for all folders/files.
2. Reads existing doc_ids directly from Hugging Face Hub Parquet via DuckDB HTTP range (~50 KB).
3. If no new files exist -> exits in ~15 seconds without downloading large files or committing.
4. If new files exist -> downloads ONLY new files, extracts text (pure Python DOCX/OLE2/PyMuPDF,
   zero MS Word / LibreOffice dependency), parses & canonicalizes speakers, merges into sorted
   Parquet files, regenerates README.md + speakers_index.json, and pushes to HF Dataset & Space.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import ssl
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

# Ensure sibling imports work when invoked as `python src/sync_incremental.py`
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crawl_manifest import ROOT_REL, crawl_subtree, list_folder_all_pages
from extract_text import clean_word_text, detect_magic, extract_docx, extract_pdf
from ole_doc_fallback import extract_ole_doc_text
from parse_speakers import parse_document

DATASET_REPO = os.environ.get("KKTC_DATASET_REPO", "stevevaius/KKTC_Meclis_Tutanak")
SPACE_REPO = os.environ.get("KKTC_SPACE_REPO", "stevevaius/KKTC-Meclis-Tutanak-Arama")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

_TR_FOLD = str.maketrans(
    {
        "Ç": "c", "ç": "c", "Ğ": "g", "ğ": "g", "İ": "i", "I": "i", "ı": "i",
        "Ö": "o", "ö": "o", "Ş": "s", "ş": "s", "Ü": "u", "ü": "u",
        "Â": "a", "â": "a", "Î": "i", "î": "i", "Û": "u", "û": "u",
    }
)


def fold_ascii(s: str) -> str:
    s = s.translate(_TR_FOLD).lower().replace("\u0307", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


def crawl_live_manifest() -> list[dict]:
    top_folders, root_files = list_folder_all_pages(ROOT_REL)
    all_records: list[dict] = []
    for f in root_files:
        filename = f.rsplit("/", 1)[-1]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        all_records.append(
            {
                "rel_path": f,
                "local_rel_path": filename,
                "download_url": f"https://evrakcm.gov.ct.tr{f}",
                "filename": filename,
                "ext": ext,
                "top_folder": "",
                "subpath": "",
                "meclis": "UNKNOWN",
                "donem": None,
                "yasama_yili": None,
            }
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for recs in ex.map(crawl_subtree, top_folders):
            all_records.extend(recs)
    all_records.sort(key=lambda r: r["local_rel_path"].lower())
    return all_records


def fetch_existing_doc_ids(dataset_repo: str) -> set[str]:
    """Read ONLY the doc_id column via HTTP Range request (~50 KB) using DuckDB."""
    docs_url = f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/data/documents.parquet"
    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        rows = con.execute("SELECT doc_id FROM read_parquet(?)", [docs_url]).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception:
        # Fallback: download documents.parquet via huggingface_hub
        p = hf_hub_download(repo_id=dataset_repo, filename="data/documents.parquet", repo_type="dataset")
        rows = con.execute("SELECT doc_id FROM read_parquet(?)", [p]).fetchall()
        return {r[0] for r in rows if r[0]}
    finally:
        con.close()


def download_temp_file(rec: dict, out_dir: Path, retries: int = 4) -> tuple[Path, str, int]:
    dest = out_dir / Path(*rec["local_rel_path"].split("/"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                rec["download_url"],
                headers={"User-Agent": "Mozilla/5.0 KKTC-Tutanak-Sync/1.0"},
            )
            h = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp, dest.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 17)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    total += len(chunk)
            if total == 0:
                raise ValueError("Empty download")
            return dest, h.hexdigest(), total
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download failed for {rec['local_rel_path']}: {last_err}")


def extract_any_linux(path: Path, ext: str) -> dict:
    """Pure-Python cross-platform extraction for .docx, .pdf, .doc (OLE2), and .rtf."""
    magic = detect_magic(path)
    if magic == "zip_docx":
        return extract_docx(path)
    if magic == "pdf":
        return extract_pdf(path)
    if magic == "ole_doc":
        raw = extract_ole_doc_text(path)
        cleaned = clean_word_text(raw)
        return {
            "text": cleaned,
            "pages": None,
            "page_count": None,
            "text_source": "doc_ole_fallback",
            "total_chars": len(cleaned),
            "empty_pages": 0,
            "image_pages": 0,
        }
    if magic == "rtf" or ext == "rtf":
        raw = path.read_bytes().decode("cp1254", errors="replace")
        # Minimal RTF text strip fallback
        txt = re.sub(r"\\[a-z]+\d*\s?", " ", raw)
        txt = re.sub(r"[{}]", "", txt)
        cleaned = clean_word_text(txt)
        return {
            "text": cleaned,
            "pages": None,
            "page_count": None,
            "text_source": "rtf",
            "total_chars": len(cleaned),
            "empty_pages": 0,
            "image_pages": 0,
        }
    raise ValueError(f"Unsupported file format magic={magic} ext={ext}")


def canonicalize_new_turns(new_speeches: list[dict], existing_speakers_parquet: str):
    """Canonicalize new turns against the existing high-frequency speaker roster."""
    con = duckdb.connect(":memory:")
    spk_rows = con.execute(
        "SELECT speaker_name, speech_count FROM read_parquet(?) WHERE speech_count >= 5",
        [existing_speakers_parquet],
    ).fetchall()
    con.close()

    folded_to_canon: dict[str, str] = {}
    spaceless_to_canon: dict[str, list[str]] = defaultdict(list)
    for nm, _cnt in sorted(spk_rows, key=lambda x: -x[1]):
        fk = fold_ascii(nm)
        if fk and fk not in folded_to_canon:
            folded_to_canon[fk] = nm
        if " " in fk and len(fk.replace(" ", "")) >= 8:
            spaceless_to_canon[fk.replace(" ", "")].append(nm)
    spaceless_unique = {k: v[0] for k, v in spaceless_to_canon.items() if len(set(v)) == 1}
    canon_items = list(folded_to_canon.items())

    # Pass 1: within-doc single-word surname resolution
    by_doc = defaultdict(list)
    for s in new_speeches:
        s["speaker_name_parsed"] = s["speaker_name"]
        s["canon_method"] = "exact"
        by_doc[s["doc_id"]].append(s)

    for turns in by_doc.values():
        surname_map = defaultdict(set)
        for t in turns:
            if t["speaker_role"] == "milletvekili_bakan" and len(t["speaker_name"].split()) >= 2:
                fp = fold_ascii(t["speaker_name"]).split()
                if len(fp) >= 2:
                    surname_map[fp[-1]].add(t["speaker_name"])
        for t in turns:
            if t["speaker_role"] == "milletvekili_bakan" and len(t["speaker_name"].split()) == 1:
                fw = fold_ascii(t["speaker_name"])
                cands = list(surname_map.get(fw, ()))
                if len(cands) == 1:
                    t["speaker_name"] = cands[0]
                    t["canon_method"] = "doc_surname"

    # Pass 2: match against existing canonical roster
    for s in new_speeches:
        nm = s["speaker_name"]
        fk = fold_ascii(nm)
        if not fk:
            continue
        if fk in folded_to_canon:
            target = folded_to_canon[fk]
            if target != nm:
                s["speaker_name"] = target
                if s["canon_method"] == "exact":
                    s["canon_method"] = "diacritic_fold"
            continue
        ns = fk.replace(" ", "")
        if ns in spaceless_unique:
            s["speaker_name"] = spaceless_unique[ns]
            s["canon_method"] = "spaceless_exact"
            continue
        words = fk.split()
        if 2 <= len(words) <= 4:
            best_score, second_score, best_target = 0.0, 0.0, None
            for c_fk, c_nm in canon_items:
                if abs(len(c_fk) - len(fk)) > 4:
                    continue
                ratio = SequenceMatcher(None, fk, c_fk).ratio()
                if ratio > best_score:
                    second_score = best_score
                    best_score = ratio
                    best_target = c_nm
                elif ratio > second_score:
                    second_score = ratio
            if best_target and best_score >= 0.88 and (best_score - second_score >= 0.04 or best_score >= 0.93):
                s["speaker_name"] = best_target
                s["canon_method"] = "fuzzy_global"


def rebuild_speakers_and_card(
    con: duckdb.DuckDBPyConnection,
    sp_path: Path,
    doc_path: Path,
    spk_out: Path,
    readme_out: Path,
    index_out: Path,
    dataset_repo: str,
):
    """Rebuild speakers.parquet, HF Dataset README.md, and Static Space speakers_index.json via DuckDB."""
    # Rebuild speakers.parquet cleanly from speeches.parquet
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT speaker_name,
                       COUNT(*)::BIGINT AS speech_count,
                       SUM(word_count)::BIGINT AS word_count,
                       COUNT(DISTINCT doc_id)::BIGINT AS doc_count,
                       MIN(tarih) AS first_date,
                       MAX(tarih) AS last_date,
                       list_sort(list_distinct(list(meclis) FILTER (WHERE meclis IS NOT NULL))) AS meclisler,
                       list_sort(list_distinct(list(donem) FILTER (WHERE donem IS NOT NULL))) AS donemler,
                       list_distinct(list(speaker_title) FILTER (WHERE speaker_title IS NOT NULL))[:15] AS titles,
                       list_distinct(list(constituency) FILTER (WHERE constituency IS NOT NULL))[:5] AS constituencies,
                       list_distinct(list(speaker_name_parsed) FILTER (WHERE speaker_name_parsed IS NOT NULL))[:20] AS raw_variants
                FROM read_parquet('{sp_path.as_posix()}')
                GROUP BY speaker_name
            )
            SELECT * FROM base
            ORDER BY speech_count DESC, speaker_name ASC
        ) TO '{spk_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Stats for README.md and speakers_index.json
    st_row = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT doc_id), SUM(word_count), SUM(char_count), MIN(tarih), MAX(tarih) FROM read_parquet('{sp_path.as_posix()}')"
    ).fetchone()
    total_docs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{doc_path.as_posix()}')").fetchone()[0]
    total_spk = con.execute(f"SELECT COUNT(*) FROM read_parquet('{spk_out.as_posix()}')").fetchone()[0]

    by_meclis_doc = dict(
        con.execute(f"SELECT meclis, COUNT(*) FROM read_parquet('{doc_path.as_posix()}') GROUP BY 1").fetchall()
    )
    by_meclis_sp = dict(
        con.execute(f"SELECT meclis, COUNT(*) FROM read_parquet('{sp_path.as_posix()}') GROUP BY 1").fetchall()
    )
    by_src = con.execute(
        f"SELECT text_source, COUNT(*) AS c FROM read_parquet('{doc_path.as_posix()}') GROUP BY 1 ORDER BY c DESC"
    ).fetchall()
    top20 = con.execute(
        f"SELECT speaker_name, speech_count, word_count, first_date, last_date FROM read_parquet('{spk_out.as_posix()}') LIMIT 20"
    ).fetchall()

    meclis_names = {
        "KKTC": "KKTC Cumhuriyet Meclisi (Dönem 1–10)",
        "KTFD": "Kıbrıs Türk Federe Meclisi",
        "KTY": "Kıbrıs Türk Yönetimi Meclisi",
        "KTFD_KM": "KTFD Kurucu Meclisi",
        "KKTC_KM": "KKTC Kurucu Meclisi",
        "KKTC_ARA": "KKTC Aradönem",
        "KC": "Kıbrıs Cumhuriyeti Dönemi (1960)",
        "TCM": "Türk Cemaat Meclisi (1963)",
    }
    rows_md = [
        f"| `{code}` | {meclis_names.get(code, code)} | {d_cnt:,} | {by_meclis_sp.get(code, 0):,} |"
        for code, d_cnt in sorted(by_meclis_doc.items(), key=lambda x: -x[1])
    ]
    top_spk_md = [
        f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3] or '—'} – {r[4] or '—'} |"
        for r in top20
    ]

    card = f"""---
language:
- tr
license: cc-by-4.0
task_categories:
- text-retrieval
- text-generation
pretty_name: KKTC Cumhuriyet Meclisi Tutanakları (Konuşmacı Bazlı)
size_categories:
- 1M<n<10M
configs:
- config_name: speeches
  default: true
  data_files:
  - split: train
    path: data/speeches.parquet
- config_name: documents
  data_files:
  - split: train
    path: data/documents.parquet
- config_name: speakers
  data_files:
  - split: train
    path: data/speakers.parquet
---

# KKTC Cumhuriyet Meclisi Tutanakları (Konuşmacı Bazlı Veri Seti)

Kuzey Kıbrıs Türk Cumhuriyeti (KKTC) Cumhuriyet Meclisi ve öncül meclislerin (1960–günümüz) resmî tutanak dergilerinin **konuşmacı bazlı (speaker-turn)** yapılandırılmış Parquet veri seti.

* **Web Arama Uygulaması:** [KKTC Meclis Tutanak Arama (HF Space)](https://huggingface.co/spaces/{SPACE_REPO})
* **Kaynak:** [KKTC Cumhuriyet Meclisi Evrak Arşivi](https://evrakcm.gov.ct.tr/siteler/belgeler/tutanaklar/Shared%20Documents/Forms/Belge%20Grntleme.aspx)
* **Toplam Belge:** {total_docs:,}
* **Toplam Konuşma Bloğu (Turn):** {st_row[0]:,}
* **Benzersiz Konuşmacı (Normalize):** {total_spk:,}
* **Toplam Kelime / Karakter:** {st_row[2]:,} kelime · {st_row[3]:,} karakter
* **Tarih Aralığı (bulunanlar):** `{st_row[4]}` – `{st_row[5]}`

## Meclis Dağılımı

| Kod | Meclis | Belge | Konuşma Bloğu |
|---|---|---:|---:|
{chr(10).join(rows_md)}

## Metin Kaynağı Dağılımı

{", ".join(f"`{k}`: {v:,}" for k, v in by_src)}

## Alt Veri Setleri (Configs)

1. **`speeches` (varsayılan):** Her satır tek bir konuşmacı sözüdür (`speech_id`, `doc_id`, `meclis`, `donem`, `yasama_yili`, `birlesim`, `oturum`, `tarih`, `turn_index`, `speaker_name`, `speaker_name_parsed`, `speaker_raw`, `speaker_role`, `speaker_title`, `constituency`, `modifiers`, `canon_method`, `text`, `char_count`, `word_count`, `text_source`, `source_file`, `source_url`).
2. **`documents`:** Belge düzeyinde tam metin ve meta veriler.
3. **`speakers`:** Konuşmacı dizini (konuşma sayısı, kelime sayısı, aktif tarih aralığı, unvanlar ve seçim bölgeleri).

## En Çok Söz Alan 20 Konuşmacı

| Konuşmacı | Konuşma | Kelime | Tarih Aralığı |
|---|---:|---:|---|
{chr(10).join(top_spk_md)}

## Örnek Kullanım (Python)

```python
from datasets import load_dataset

ds = load_dataset("{dataset_repo}", "speeches", split="train")
erhurman = ds.filter(lambda x: x["speaker_name"] == "Tufan Erhürman")
print(len(erhurman), erhurman[0]["tarih"], erhurman[0]["text"][:200])
```

## Lisans ve Atıf

Orijinal tutanak belgelerinin hakları **KKTC Cumhuriyet Meclisi**'ne aittir. Bu veri seti arşiv, araştırma ve konuşmacı bazlı arama amacıyla derlenmiştir; yapılandırma, ayrıştırma ve normalizasyon katmanı **CC BY 4.0** ile paylaşılmıştır. Atıf: *KKTC Cumhuriyet Meclisi Tutanak Dergisi, evrakcm.gov.ct.tr*.
"""
    readme_out.write_text(card, encoding="utf-8")

    # Rebuild compact speakers_index.json for Static Space
    tl_rows = con.execute(
        f"""
        SELECT speaker_name, substr(tarih, 1, 4) AS yr, COUNT(*) AS cnt
        FROM read_parquet('{sp_path.as_posix()}')
        WHERE tarih IS NOT NULL AND length(tarih) >= 4
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    timelines: dict[str, list[list]] = {}
    for nm, yr, cnt in tl_rows:
        timelines.setdefault(nm, []).append([yr, int(cnt)])

    spk_rows = con.execute(
        f"""
        SELECT speaker_name, speech_count, word_count, doc_count,
               first_date, last_date, titles, constituencies
        FROM read_parquet('{spk_out.as_posix()}')
        ORDER BY speech_count DESC
        """
    ).fetchall()
    speakers_list = []
    for nm, sc, wc, dc, fd, ld, titles, consts in spk_rows:
        entry = {
            "n": nm,
            "f": fold_ascii(nm),
            "s": int(sc),
            "w": int(wc),
            "d": int(dc),
            "a": (fd or "")[:10],
            "b": (ld or "")[:10],
            "c": list(consts)[:3] if consts else [],
            "t": list(titles)[:5] if titles else [],
        }
        if int(sc) >= 5 and nm in timelines:
            entry["y"] = timelines[nm]
        speakers_list.append(entry)

    from datetime import datetime, timezone
    latest_row = con.execute(
        f"""
        SELECT tarih, meclis, donem, yasama_yili, birlesim, doc_id, source_url
        FROM read_parquet('{doc_path.as_posix()}')
        WHERE tarih IS NOT NULL
        ORDER BY tarih DESC, doc_id DESC
        LIMIT 1
        """
    ).fetchone()

    index_payload = {
        "stats": {
            "speeches": int(st_row[0]),
            "docs": int(st_row[1]),
            "total_docs": int(total_docs),
            "words": int(st_row[2] or 0),
            "min_date": st_row[4],
            "max_date": st_row[5],
            "speakers": len(speakers_list),
            "latest_date": latest_row[0] if latest_row else st_row[5],
            "latest_meclis": latest_row[1] if latest_row else "KKTC",
            "latest_donem": int(latest_row[2]) if (latest_row and latest_row[2] is not None) else None,
            "latest_yasama_yili": int(latest_row[3]) if (latest_row and latest_row[3] is not None) else None,
            "latest_birlesim": int(latest_row[4]) if (latest_row and latest_row[4] is not None) else None,
            "latest_doc_id": latest_row[5] if latest_row else None,
            "latest_source_url": latest_row[6] if latest_row else None,
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "speakers": speakers_list,
    }
    index_out.write_text(json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Incremental sync of KKTC Meclisi minutes to Hugging Face.")
    ap.add_argument("--dataset-repo", default=DATASET_REPO)
    ap.add_argument("--space-repo", default=SPACE_REPO)
    ap.add_argument("--dry-run", action="store_true", help="Run full pipeline without uploading to HF Hub")
    ap.add_argument(
        "--simulate-drop",
        type=int,
        default=0,
        help="Testing flag: pretend the newest N docs are missing (implies --dry-run)",
    )
    args = ap.parse_args()
    if args.simulate_drop > 0:
        args.dry_run = True

    t0 = time.perf_counter()
    print(f"[1/6] Crawling live KKTC SharePoint archive...")
    live_records = crawl_live_manifest()
    print(f"      Live files on evrakcm.gov.ct.tr: {len(live_records):,}")

    print(f"[2/6] Checking existing doc_ids on Hugging Face ({args.dataset_repo})...")
    existing_ids = fetch_existing_doc_ids(args.dataset_repo)
    print(f"      Existing docs in HF dataset: {len(existing_ids):,}")

    if args.simulate_drop > 0:
        # Drop N recent .docx docs to test the full incremental merge path end-to-end
        docx_ids = sorted(i for i in existing_ids if i.lower().endswith(".docx"))
        dropped = set(docx_ids[-args.simulate_drop :])
        existing_ids -= dropped
        print(f"      [SIMULATE-DROP] Pretending {len(dropped)} docs are new: {sorted(dropped)}")

    new_records = [r for r in live_records if r["local_rel_path"] not in existing_ids]
    print(f"      New files detected: {len(new_records)}")

    if not new_records:
        print(f"Up to date! Zero new files found. Exiting cleanly in {time.perf_counter() - t0:.1f}s.")
        return

    with tempfile.TemporaryDirectory(prefix="kktc_sync_") as tmp_str:
        tmp = Path(tmp_str)
        raw_dir = tmp / "raw"
        out_hf = tmp / "hf_upload"
        (out_hf / "data").mkdir(parents=True, exist_ok=True)

        print(f"[3/6] Downloading & extracting {len(new_records)} new files...")
        new_docs: list[dict] = []
        new_speeches: list[dict] = []
        errors: list[str] = []
        for idx, rec in enumerate(new_records, start=1):
            try:
                fpath, sha, sz = download_temp_file(rec, raw_dir)
                ext_res = extract_any_linux(fpath, rec["ext"])
                extracted_doc = {
                    "local_rel_path": rec["local_rel_path"],
                    "filename": rec["filename"],
                    "ext": rec["ext"],
                    "meclis": rec["meclis"],
                    "donem": rec["donem"],
                    "yasama_yili": rec["yasama_yili"],
                    "top_folder": rec["top_folder"],
                    "download_url": rec["download_url"],
                    "sha256": sha,
                    "size_bytes": sz,
                    **ext_res,
                }
                doc_rec, turns = parse_document(extracted_doc)
                new_docs.append(doc_rec)
                new_speeches.extend(turns)
                print(
                    f"      [{idx}/{len(new_records)}] {rec['local_rel_path']} -> "
                    f"date={doc_rec['tarih']} turns={len(turns)}"
                )
            except Exception as e:
                errors.append(f"{rec['local_rel_path']}: {e}")
                print(f"      [ERR] {rec['local_rel_path']}: {e}")

        if not new_docs:
            raise RuntimeError(f"All {len(new_records)} new files failed: {errors}")

        print(f"[4/6] Downloading existing Parquet files from {args.dataset_repo}...")
        old_sp_path = hf_hub_download(repo_id=args.dataset_repo, filename="data/speeches.parquet", repo_type="dataset")
        old_doc_path = hf_hub_download(repo_id=args.dataset_repo, filename="data/documents.parquet", repo_type="dataset")
        old_spk_path = hf_hub_download(repo_id=args.dataset_repo, filename="data/speakers.parquet", repo_type="dataset")

        print(f"[5/6] Canonicalizing {len(new_speeches):,} new turns & merging Parquet files...")
        canonicalize_new_turns(new_speeches, old_spk_path)
        # Refresh doc konusmacilar from canonicalized new turns
        by_doc_spks = defaultdict(set)
        for s in new_speeches:
            by_doc_spks[s["doc_id"]].add(s["speaker_name"])
        for d in new_docs:
            d["konusmacilar"] = sorted(by_doc_spks.get(d["doc_id"], set()))

        # Match exact PyArrow schemas of existing Parquet files
        old_sp_schema = pq.read_schema(old_sp_path)
        old_doc_schema = pq.read_schema(old_doc_path)

        new_sp_tmp = tmp / "new_speeches.parquet"
        new_doc_tmp = tmp / "new_documents.parquet"
        pq.write_table(pa.Table.from_pylist(new_speeches, schema=old_sp_schema), new_sp_tmp)
        pq.write_table(pa.Table.from_pylist(new_docs, schema=old_doc_schema), new_doc_tmp)

        merged_sp = out_hf / "data/speeches.parquet"
        merged_doc = out_hf / "data/documents.parquet"
        merged_spk = out_hf / "data/speakers.parquet"
        readme_out = out_hf / "README.md"
        index_out = tmp / "speakers_index.json"

        con = duckdb.connect(":memory:")
        new_doc_ids_sql = ", ".join("'" + d["doc_id"].replace("'", "''") + "'" for d in new_docs)

        # Merge + deduplicate by doc_id (so re-running never duplicates!) + sort for WASM range pruning
        con.execute(
            f"""
            COPY (
                SELECT * FROM (
                    SELECT * FROM read_parquet('{Path(old_sp_path).as_posix()}')
                    WHERE doc_id NOT IN ({new_doc_ids_sql})
                    UNION ALL
                    SELECT * FROM read_parquet('{new_sp_tmp.as_posix()}')
                )
                ORDER BY speaker_name ASC, tarih DESC NULLS LAST, turn_index ASC
            ) TO '{merged_sp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT * FROM (
                    SELECT * FROM read_parquet('{Path(old_doc_path).as_posix()}')
                    WHERE doc_id NOT IN ({new_doc_ids_sql})
                    UNION ALL
                    SELECT * FROM read_parquet('{new_doc_tmp.as_posix()}')
                )
                ORDER BY doc_id ASC
            ) TO '{merged_doc.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128)
            """
        )

        rebuild_speakers_and_card(
            con, merged_sp, merged_doc, merged_spk, readme_out, index_out, args.dataset_repo
        )
        new_total_sp = con.execute(f"SELECT COUNT(*) FROM read_parquet('{merged_sp.as_posix()}')").fetchone()[0]
        new_total_doc = con.execute(f"SELECT COUNT(*) FROM read_parquet('{merged_doc.as_posix()}')").fetchone()[0]
        con.close()

        print(
            f"      Merged totals: {new_total_doc:,} docs, {new_total_sp:,} speeches | "
            f"speeches={merged_sp.stat().st_size/(1024*1024):.1f} MB, "
            f"index={index_out.stat().st_size/1024:.1f} KB"
        )

        if args.dry_run:
            print(f"[6/6] DRY-RUN complete in {time.perf_counter() - t0:.1f}s (no upload performed).")
            return

        print(f"[6/6] Uploading updates to HF Dataset ({args.dataset_repo}) and Space ({args.space_repo})...")
        api = HfApi()
        msg = f"Auto-sync: +{len(new_docs)} KKTC minutes (+{len(new_speeches):,} speech turns)"
        api.upload_folder(
            folder_path=str(out_hf),
            repo_id=args.dataset_repo,
            repo_type="dataset",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(index_out),
            path_in_repo="speakers_index.json",
            repo_id=args.space_repo,
            repo_type="space",
            commit_message=msg,
        )
        print(f"SYNC UPLOADED SUCCESSFULLY in {time.perf_counter() - t0:.1f}s!")


if __name__ == "__main__":
    main()
