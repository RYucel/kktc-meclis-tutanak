"""Re-extract and re-parse selected documents from the local raw files, then patch the published dataset.

Use this after an extractor fix (e.g. letter-spaced PDF text layers) or, later, after re-OCR. Only the listed
documents change; every other row is copied through untouched, so permalinks of other speeches stay stable.

Typical flow:
  python src/ocr_quality.py --data-dir data/hf_dataset/data                 # writes data/ocr_quality/spaced_docs.txt
  python src/reprocess_docs.py --docs data/ocr_quality/spaced_docs.txt --data-dir data/hf_dataset/data   # dry run
  python src/reprocess_docs.py --docs data/ocr_quality/spaced_docs.txt --from-hub --upload

Outputs (--out-dir, default data/reprocess): data/{speeches,documents,speakers}.parquet, README.md, terms_cache/,
space/*.json, reprocess_report.json (per-document before/after turns and OCR score).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ocr_quality import build_vocab, score_text  # noqa: E402
from parse_speakers import parse_document  # noqa: E402
from speaker_index import stats_from_parquet  # noqa: E402
from speaker_resolution import load_rosters, resolve_records  # noqa: E402
from sync_incremental import (  # noqa: E402
    DATASET_REPO,
    SPACE_REPO,
    canonicalize_new_turns,
    extract_any_linux,
    rebuild_speakers_and_card,
)


def _p(path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def load_index(path: Path) -> dict[str, dict]:
    recs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["local_rel_path"]] = r
    return recs


def reparse(doc_ids: list[str], index: dict[str, dict], raw_dir: Path, cache_dir: Path, budget_s: float = 0,
            reocr_dir: Path | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """Parse each document once; results are cached per document so an interrupted run resumes."""
    import hashlib

    docs, speeches, errors = [], [], []
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, did in enumerate(doc_ids, 1):
        cp = cache_dir / (hashlib.sha1(did.encode("utf-8")).hexdigest() + ".json")
        if cp.exists():
            c = json.loads(cp.read_text(encoding="utf-8"))
            docs.append(c["doc"])
            speeches.extend(c["turns"])
            continue
        if budget_s and time.perf_counter() - t0 > budget_s:
            raise SystemExit(f"Time budget reached after {i - 1}/{len(doc_ids)} documents; re-run to continue.")
        rec = index.get(did)
        src = raw_dir / Path(*did.split("/"))
        if rec is None or not src.exists():
            errors.append(f"{did}: {'not in download index' if rec is None else 'raw file missing'}")
            continue
        try:
            ext = extract_any_linux(src, rec["ext"])
            if reocr_dir and ext.get("page_count"):
                from reocr_vlm import cached_pages

                vlm_pages = cached_pages(reocr_dir, did, ext["page_count"])
                if vlm_pages is not None:  # use the vision-model transcription only when every page was re-read
                    full = "\n\n".join(p.strip() for p in vlm_pages if p.strip())
                    ext = {**ext, "pages": vlm_pages, "text": full, "total_chars": len(full), "text_source": "pdf_vlm_ocr"}
            doc = {k: rec.get(k) for k in ("local_rel_path", "filename", "ext", "meclis", "donem", "yasama_yili", "top_folder", "download_url", "sha256", "size_bytes")}
            doc_rec, turns = parse_document({**doc, **ext})
        except Exception as e:  # keep going: one bad file must not block the batch
            errors.append(f"{did}: {e}")
            continue
        cp.write_text(json.dumps({"doc": doc_rec, "turns": turns, "pages_fixed": ext.get("spaced_pages_fixed", 0)}, ensure_ascii=False, default=str), encoding="utf-8")
        docs.append(doc_rec)
        speeches.extend(turns)
        print(f"   [{i}/{len(doc_ids)}] {did}: pages_fixed={ext.get('spaced_pages_fixed', 0)} turns={len(turns)}")
    return docs, speeches, errors


def stream_replace_documents(old_doc: str, new_docs: list[dict], out: Path, remove: set[str] | None = None) -> int:
    """Copy documents.parquet batch by batch: swap re-parsed rows in place, drop `remove`, append brand-new documents.

    Returns the number of documents written."""
    remove = remove or set()
    pf = pq.ParquetFile(old_doc)
    schema = pf.schema_arrow
    by_id = {d["doc_id"]: d for d in new_docs}
    seen: set[str] = set()
    n = 0
    tmp = out.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(str(tmp), schema, compression="zstd") as w:
        for batch in pf.iter_batches(batch_size=128):
            rows = batch.to_pylist()
            if any(r["doc_id"] in by_id or r["doc_id"] in remove for r in rows):
                rows = [
                    {k: by_id[r["doc_id"]].get(k, r[k]) for k in r} if r["doc_id"] in by_id else r
                    for r in rows if r["doc_id"] not in remove
                ]
                if not rows:
                    continue
                batch = pa.RecordBatch.from_pylist(rows, schema=schema)
            seen.update(r for r in batch.column(batch.schema.get_field_index("doc_id")).to_pylist())
            n += batch.num_rows
            w.write_batch(batch, row_group_size=128)
        fresh = [d for d in new_docs if d["doc_id"] not in seen and d["doc_id"] not in remove]
        fresh.sort(key=lambda d: d["doc_id"])
        for i in range(0, len(fresh), 128):
            chunk = [{f.name: d.get(f.name) for f in schema} for d in fresh[i:i + 128]]
            w.write_batch(pa.RecordBatch.from_pylist(chunk, schema=schema), row_group_size=128)
            n += len(chunk)
    os.replace(tmp, out)
    return n


def run(args) -> dict:
    t0 = time.perf_counter()
    if args.upload and args.skip_aux:
        raise SystemExit("--upload cannot be combined with --skip-aux.")
    if not args.docs and not args.add_dir:
        raise SystemExit("Give --docs (re-parse) and/or --add-dir (pre-built documents/speeches JSONL).")
    doc_ids = [ln.strip() for ln in Path(args.docs).read_text(encoding="utf-8-sig").splitlines() if ln.strip()] if args.docs else []
    remove_ids = {ln.strip() for ln in Path(args.remove_docs).read_text(encoding="utf-8-sig").splitlines() if ln.strip()} if args.remove_docs else set()
    if args.limit:
        doc_ids = doc_ids[: args.limit]
    out_root = Path(args.out_dir)
    out_data = out_root / "data"
    space_dir = out_root / "space"
    for d in (out_data, space_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.from_hub:
        from huggingface_hub import hf_hub_download

        src = {n: hf_hub_download(repo_id=args.dataset_repo, filename=f"data/{n}.parquet", repo_type="dataset") for n in ("speeches", "documents", "speakers")}
    else:
        src = {n: str(Path(args.data_dir) / f"{n}.parquet") for n in ("speeches", "documents", "speakers")}

    con = duckdb.connect(":memory:")
    tmp_dir = out_root / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{_p(tmp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    if args.threads:
        con.execute(f"SET threads={int(args.threads)}")

    known = {r[0] for r in con.execute(f"SELECT doc_id FROM read_parquet('{_p(src['documents'])}')").fetchall()}
    missing = [d for d in doc_ids if d not in known]
    if missing:
        raise SystemExit(f"{len(missing)} doc_ids are not in the dataset, e.g. {missing[:3]}")

    new_docs, new_sp, errors = [], [], []
    if doc_ids:
        print(f"[1/4] Re-extracting {len(doc_ids)} documents from {args.raw_dir}...")
        new_docs, new_sp, errors = reparse(doc_ids, load_index(Path(args.index)), Path(args.raw_dir), out_root / ("parsed_vlm" if args.reocr_dir else "parsed"), args.budget,
                                           Path(args.reocr_dir) if args.reocr_dir else None)
        if args.parse_only:
            print(f"   parsed {len(new_docs)} documents, {len(new_sp):,} turns (--parse-only)")
            return {}
        if not new_docs:
            raise SystemExit(f"Nothing re-parsed: {errors}")
        print(f"[2/4] Canonicalizing {len(new_sp):,} turns against the published speakers...")
        canonicalize_new_turns(new_sp, src["speakers"])
        res = resolve_records(new_sp, load_rosters(args.rosters), base_stats=stats_from_parquet(con, src["speeches"]))
    else:
        res = {}
    if args.add_dir:
        # Pre-built rows (e.g. cy_build.py): names are already canonical; only hand-checked aliases apply
        from speaker_resolution import apply_manual_aliases

        ad = Path(args.add_dir)
        add_docs = [json.loads(x) for x in (ad / "documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        add_sp = [json.loads(x) for x in (ad / "speeches.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        apply_manual_aliases(add_sp)
        print(f"[1/4] Adding {len(add_docs)} pre-built documents ({len(add_sp):,} turns) from {ad}; removing {len(remove_ids)}")
        new_docs += add_docs
        new_sp += add_sp
    by_doc = defaultdict(set)
    for s in new_sp:
        by_doc[s["doc_id"]].add(s["speaker_name"])
    for d in new_docs:
        d["konusmacilar"] = sorted(by_doc.get(d["doc_id"], set()))

    # before/after report per document
    vocab_dir = Path(args.quality_dir)
    vocab = build_vocab(con, src["documents"], vocab_dir) if vocab_dir else set()
    ids_sql = ", ".join("'" + x.replace("'", "''") + "'" for x in sorted({d["doc_id"] for d in new_docs} | remove_ids)) or "''"
    old_turns = dict(con.execute(f"SELECT doc_id, COUNT(*) FROM read_parquet('{_p(src['speeches'])}') WHERE doc_id IN ({ids_sql}) GROUP BY doc_id").fetchall())
    old_text = dict(con.execute(f"SELECT doc_id, text FROM read_parquet('{_p(src['documents'])}') WHERE doc_id IN ({ids_sql})").fetchall())
    new_turns = defaultdict(int)
    for s in new_sp:
        new_turns[s["doc_id"]] += 1
    per_doc = []
    for d in new_docs:
        b, a = score_text(old_text.get(d["doc_id"]) or "", vocab), score_text(d.get("text") or "", vocab)
        per_doc.append({"doc_id": d["doc_id"], "tarih": d.get("tarih"), "turns_before": old_turns.get(d["doc_id"], 0), "turns_after": new_turns[d["doc_id"]], "score_before": b["score"], "score_after": a["score"]})
    del old_text

    print("[3/4] Patching speeches/documents parquet...")
    sp_out, doc_out, spk_out = out_data / "speeches.parquet", out_data / "documents.parquet", out_data / "speakers.parquet"
    new_sp_tmp = out_root / "new_speeches.parquet"
    pq.write_table(pa.Table.from_pylist(new_sp, schema=pq.read_schema(src["speeches"])), new_sp_tmp)
    con.execute(
        f"""
        COPY (
          SELECT * FROM (
            SELECT * FROM read_parquet('{_p(src['speeches'])}') WHERE doc_id NOT IN ({ids_sql})
            UNION ALL SELECT * FROM read_parquet('{_p(new_sp_tmp)}')
          ) ORDER BY speaker_name ASC, tarih DESC NULLS LAST, turn_index ASC
        ) TO '{_p(sp_out)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)
        """
    )
    n_written = stream_replace_documents(src["documents"], new_docs, doc_out, remove_ids)

    print("[4/4] Rebuilding speakers, card and Space indices...")
    from optimize_for_wasm import EXTRA_SEARCH_ALIASES, build_terms_index, export_term_shards

    rebuild_speakers_and_card(
        con, sp_out, doc_out, spk_out, out_root / "README.md", space_dir / "speakers_index.json", args.dataset_repo,
        extra_aliases={k: set(v) for k, v in EXTRA_SEARCH_ALIASES.items()},
    )
    if not args.skip_aux:
        from analyze_topics import build_speaker_topics, build_topics_index

        rp = Path(args.rosters) if args.rosters else ROOT / "data" / "official_rosters.json"
        spk_cnt = {r[0]: int(r[1]) for r in con.execute(f"SELECT speaker_name, speech_count FROM read_parquet('{_p(spk_out)}')").fetchall()}
        for fn, payload in (
            ("terms_index.json", lambda: build_terms_index(con, sp_out.as_posix(), spk_cnt, rp)),
            ("topics_index.json", lambda: build_topics_index(con, sp_out.as_posix())),
            ("speaker_topics.json", lambda: build_speaker_topics(con, sp_out.as_posix())),
        ):
            (space_dir / fn).write_text(json.dumps(payload(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        touched = {(d["meclis"], d["donem"]) for d in new_docs} | {
            (m, dn) for m, dn in con.execute(f"SELECT DISTINCT meclis, donem FROM read_parquet('{_p(src['documents'])}') WHERE doc_id IN ({ids_sql})").fetchall()}
        export_term_shards(con, sp_out.as_posix(), out_root / "terms_cache", only_terms=touched)

    n_in = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_p(src['documents'])}')").fetchone()[0]
    n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_p(doc_out)}')").fetchone()[0]
    removed_present = len(remove_ids & known)
    added = len({d["doc_id"] for d in new_docs} - known - remove_ids)
    if n_out != n_in - removed_present + added or n_out != n_written:
        raise RuntimeError(f"Document count mismatch: {n_in} - {removed_present} + {added} != {n_out}")
    total_sp = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_p(sp_out)}')").fetchone()[0]
    con.close()
    new_sp_tmp.unlink(missing_ok=True)

    report = {
        "documents": len(new_docs),
        "errors": errors,
        "turns_before": sum(p["turns_before"] for p in per_doc),
        "turns_after": sum(p["turns_after"] for p in per_doc),
        "total_speeches": total_sp,
        "resolution": res.get("method_counts", {}),
        "per_doc": sorted(per_doc, key=lambda p: p["score_after"] - p["score_before"]),
    }
    (out_root / "reprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   docs={len(new_docs)} errors={len(errors)} turns {report['turns_before']:,} -> {report['turns_after']:,}; total speeches {total_sp:,} ({time.perf_counter() - t0:.0f}s)")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        msg = f"Re-extract {len(new_docs)} documents ({args.reason}); turns {report['turns_before']:,} -> {report['turns_after']:,}"
        api.upload_folder(folder_path=str(out_root), repo_id=args.dataset_repo, repo_type="dataset", commit_message=msg,
                          allow_patterns=["data/*.parquet", "README.md", "terms_cache/**"])
        files = ["speakers_index.json", "terms_index.json", "topics_index.json", "speaker_topics.json"]
        import shutil

        extra = [(Path(args.quality_dir) / "ocr_variants.json", "ocr_variants.json")]
        if args.upload_ui:
            extra += [(ROOT / "space_static" / "index.html", "index.html"), (ROOT / "space_static" / "og-image.png", "og-image.png")]
        for srcf, name in extra:
            if srcf.exists():
                shutil.copyfile(srcf, space_dir / name)
                files.append(name)
        api.upload_folder(folder_path=str(space_dir), repo_id=args.space_repo, repo_type="space", allow_patterns=files, commit_message=msg)
        print(f"Uploaded dataset + Space files: {files}")
    return report


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=None, help="Text file with one doc_id (local_rel_path) per line to re-parse")
    ap.add_argument("--add-dir", default=None, help="Directory with pre-built documents.jsonl + speeches.jsonl to add/replace")
    ap.add_argument("--remove-docs", default=None, help="Text file with doc_ids to delete (with their speeches)")
    ap.add_argument("--data-dir", default="data/hf_dataset/data")
    ap.add_argument("--from-hub", action="store_true", help="Read the current published parquet from the Hub")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--index", default="data/download_index.jsonl")
    ap.add_argument("--out-dir", default="data/reprocess")
    ap.add_argument("--quality-dir", default="data/ocr_quality", help="Where the clean vocabulary is cached")
    ap.add_argument("--rosters", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reason", default="letter-spaced PDF text fix")
    ap.add_argument("--memory-limit", default="1.5GB")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-aux", action="store_true")
    ap.add_argument("--reocr-dir", default=None, help="Use vision-model page transcriptions from reocr_vlm.py when complete")
    ap.add_argument("--parse-only", action="store_true", help="Only extract+parse (cached per document), then stop")
    ap.add_argument("--budget", type=float, default=0, help="Stop parsing after this many seconds (resume later)")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--upload-ui", action="store_true", help="Also publish space_static/index.html + og-image.png")
    ap.add_argument("--dataset-repo", default=DATASET_REPO)
    ap.add_argument("--space-repo", default=SPACE_REPO)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
