"""Apply the 3rd-pass speaker resolution to the *published* dataset — no raw archive needed.

Typical use (Linux/Windows, ~2–4 minutes for 1.28M turns):

    # 1. dry run straight from the Hub: Parquet under data/recanon/, Space JSON under space_static/,
    #    audit report in data/parsed/speaker_resolution_report.json
    python src/recanonicalize.py --from-hub

    # 2. happy with data/parsed/speaker_resolution_report.json?  push dataset + Space indices + UI
    python src/recanonicalize.py --from-hub --upload --upload-ui

Space JSON indices are written to space_static/ by default, so the committed copies never go stale
(src/deploy_space.py uploads only the UI unless --with-indices, and refuses stale indices).

What it does:
1. Loads speeches/documents/speakers Parquet (local ``data/hf_dataset/data`` or ``--from-hub``).
2. Builds per (speaker_name, speaker_role, year) groups and resolves them with ``SpeakerResolver``.
3. Rewrites ``speeches.parquet`` (``speaker_name`` + ``canon_method``; ``speaker_name_parsed`` keeps
   the original label, so the change is fully auditable and reversible) sorted for WASM range pruning,
   and refreshes ``documents.konusmacilar``.
4. Regenerates speakers.parquet, dataset card, speakers/terms/topics/speaker_topics JSON and
   per-term shards with the same functions the weekly sync uses.
5. Writes ``data/parsed/speaker_resolution_report.json`` (every mapping with counts + before/after metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_index import stats_from_parquet  # noqa: E402
from speaker_resolution import ROLE_TITLE_NAMES, SpeakerResolver, load_rosters  # noqa: E402

ROOT = SRC_DIR.parent
DATASET_REPO = os.environ.get("KKTC_DATASET_REPO", "stevevaius/KKTC_Meclis_Tutanak")
SPACE_REPO = os.environ.get("KKTC_SPACE_REPO", "stevevaius/KKTC-Meclis-Tutanak-Arama")


def _p(path: Path | str) -> str:
    return Path(path).as_posix().replace("'", "''")


def resolve_groups(con, sp_path: str, resolver: SpeakerResolver) -> list[dict]:
    rows = con.execute(
        f"""
        SELECT speaker_name, speaker_role, TRY_CAST(substr(tarih, 1, 4) AS INTEGER) AS yr, COUNT(*) AS n
        FROM read_parquet('{_p(sp_path)}')
        WHERE speaker_name IS NOT NULL
        GROUP BY 1, 2, 3
        """
    ).fetchall()
    out = []
    for name, role, yr, n in rows:
        res = resolver.resolve(name, role, yr)
        if res and res.name != name:
            out.append({"old_name": name, "role": role, "yr": yr, "new_name": res.name, "method": res.method, "n": int(n)})
    return out


def quality_metrics(con, sp_path: str, rosters: dict) -> dict:
    stats = stats_from_parquet(con, sp_path)
    resolver = SpeakerResolver(stats, rosters)
    verified = resolver.verified_names()
    total = sum(v["count"] for v in stats.values())
    ver_turns = sum(v["count"] for k, v in stats.items() if k in verified)
    return {
        "distinct_speaker_names": len(stats),
        "verified_names": sum(1 for k in stats if k in verified and k not in ROLE_TITLE_NAMES),
        "names_with_le2_turns": sum(1 for v in stats.values() if v["count"] <= 2),
        "verified_turn_share": round(ver_turns / total, 4) if total else 0.0,
        "turns": total,
    }


def run(args) -> dict:
    t0 = time.perf_counter()
    if args.upload and args.skip_aux:
        # terms/topics/speaker_topics/term shards would keep the old speaker names -> inconsistent tabs
        raise SystemExit("--upload cannot be combined with --skip-aux: every Space index must be rebuilt together.")
    if args.upload_ui and not args.upload:
        raise SystemExit("--upload-ui requires --upload.")
    rosters = load_rosters(args.rosters)

    if args.from_hub:
        from huggingface_hub import hf_hub_download

        src = {
            name: hf_hub_download(repo_id=args.dataset_repo, filename=f"data/{name}.parquet", repo_type="dataset")
            for name in ("speeches", "documents", "speakers")
        }
    else:
        src = {name: str(Path(args.data_dir) / f"{name}.parquet") for name in ("speeches", "documents", "speakers")}
    for k, v in src.items():
        if not Path(v).exists():
            raise FileNotFoundError(f"Missing {k} parquet: {v}")

    out_root = Path(args.out_dir)
    out_data = out_root / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    space_dir = Path(args.space_dir)
    space_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    # Bounded memory: sorting 1.3M speech texts must spill to disk instead of exhausting small machines
    tmp_dir = Path(args.temp_dir) if args.temp_dir else out_root / ".duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{_p(tmp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    if args.threads:
        con.execute(f"SET threads={int(args.threads)}")
    print("[1/5] Measuring current speaker quality...")
    before = quality_metrics(con, src["speeches"], rosters)
    print(f"      before: {before}")

    print("[2/5] Resolving (speaker, role, year) groups...")
    stats = stats_from_parquet(con, src["speeches"])
    resolver = SpeakerResolver(stats, rosters)
    mapping = resolve_groups(con, src["speeches"], resolver)
    by_method = Counter()
    for m in mapping:
        by_method[m["method"]] += m["n"]
    print(f"      anchors={len(resolver.anchors):,} anchor_merges={len(resolver.report.get('anchor_merges', [])):,} "
          f"groups_remapped={len(mapping):,} turns_remapped={sum(by_method.values()):,} {dict(by_method)}")

    map_tbl = pa.Table.from_pylist(
        mapping or [{"old_name": "", "role": None, "yr": None, "new_name": "", "method": "", "n": 0}],
        schema=pa.schema([
            ("old_name", pa.string()), ("role", pa.string()), ("yr", pa.int32()),
            ("new_name", pa.string()), ("method", pa.string()), ("n", pa.int64()),
        ]),
    )
    con.register("name_map", map_tbl)

    sp_out = out_data / "speeches.parquet"
    doc_out = out_data / "documents.parquet"
    spk_out = out_data / "speakers.parquet"
    tmp_sp = out_data / "speeches.parquet.tmp"
    tmp_doc = out_data / "documents.parquet.tmp"

    print("[3/5] Rewriting speeches.parquet + documents.parquet...")
    sp_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_p(src['speeches'])}')").fetchall()}
    parsed_expr = (
        "COALESCE(s.speaker_name_parsed, s.speaker_name) AS speaker_name_parsed" if "speaker_name_parsed" in sp_cols else None
    )
    replace_parts = [
        "COALESCE(m.new_name, s.speaker_name) AS speaker_name",
        "COALESCE(m.method, s.canon_method) AS canon_method" if "canon_method" in sp_cols else None,
        parsed_expr,
    ]
    replace_sql = ", ".join(p for p in replace_parts if p)
    con.execute(
        f"""
        COPY (
            SELECT s.* REPLACE ({replace_sql})
            FROM read_parquet('{_p(src['speeches'])}') s
            LEFT JOIN name_map m
              ON s.speaker_name = m.old_name
             AND s.speaker_role IS NOT DISTINCT FROM m.role
             AND TRY_CAST(substr(s.tarih, 1, 4) AS INTEGER) IS NOT DISTINCT FROM m.yr
            ORDER BY speaker_name ASC, tarih DESC NULLS LAST, turn_index ASC
        ) TO '{_p(tmp_sp)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)
        """
    )
    n_in = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_p(src['speeches'])}')").fetchone()[0]
    n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_p(tmp_sp)}')").fetchone()[0]
    if n_in != n_out:
        raise RuntimeError(f"Row count changed during rewrite: {n_in} -> {n_out}")
    os.replace(tmp_sp, sp_out)

    doc_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_p(src['documents'])}')").fetchall()}
    if "konusmacilar" in doc_cols:
        con.execute(
            f"""
            COPY (
                SELECT d.* REPLACE (COALESCE(k.ks, d.konusmacilar) AS konusmacilar)
                FROM read_parquet('{_p(src['documents'])}') d
                LEFT JOIN (
                    SELECT doc_id, list_sort(list_distinct(list(speaker_name))) AS ks
                    FROM read_parquet('{_p(sp_out)}') GROUP BY doc_id
                ) k USING (doc_id)
                ORDER BY doc_id ASC
            ) TO '{_p(tmp_doc)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 128)
            """
        )
        os.replace(tmp_doc, doc_out)
    elif Path(src["documents"]).resolve() != doc_out.resolve():
        shutil.copyfile(src["documents"], doc_out)

    print("[4/5] Rebuilding speakers.parquet, dataset card and Space indices...")
    from optimize_for_wasm import EXTRA_SEARCH_ALIASES, build_terms_index, export_term_shards
    from sync_incremental import rebuild_speakers_and_card

    index_out = space_dir / "speakers_index.json"
    rebuild_speakers_and_card(
        con, sp_out, doc_out, spk_out, out_root / "README.md", index_out, args.dataset_repo,
        extra_aliases={k: set(v) for k, v in EXTRA_SEARCH_ALIASES.items()},
    )
    if not args.skip_aux:
        from analyze_topics import build_speaker_topics, build_topics_index

        spk_cnt = {r[0]: int(r[1]) for r in con.execute(f"SELECT speaker_name, speech_count FROM read_parquet('{_p(spk_out)}')").fetchall()}
        rp = Path(args.rosters) if args.rosters else ROOT / "data" / "official_rosters.json"
        (space_dir / "terms_index.json").write_text(
            json.dumps(build_terms_index(con, sp_out.as_posix(), spk_cnt, rp), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        (space_dir / "topics_index.json").write_text(
            json.dumps(build_topics_index(con, sp_out.as_posix()), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        (space_dir / "speaker_topics.json").write_text(
            json.dumps(build_speaker_topics(con, sp_out.as_posix()), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        export_term_shards(con, sp_out.as_posix(), out_root / "terms_cache")

    after = quality_metrics(con, sp_out.as_posix(), rosters)
    print(f"      after:  {after}")

    agg: dict[tuple, int] = Counter()
    for m in mapping:
        agg[(m["old_name"], m["new_name"], m["method"])] += m["n"]
    report = {
        "before": before,
        "after": after,
        "turns_remapped_by_method": dict(by_method),
        "anchors": len(resolver.anchors),
        "anchor_merges": resolver.report.get("anchor_merges", []),
        "core_anchors": resolver.report.get("core_anchors", []),
        "multi_person_labels": resolver.report.get("multi_person", []),
        "mappings": [
            {"from": a, "to": b, "method": meth, "turns": n}
            for (a, b, meth), n in sorted(agg.items(), key=lambda kv: -kv[1])
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    con.close()
    print(f"[5/5] Report: {report_path} ({len(report['mappings']):,} distinct mappings) in {time.perf_counter() - t0:.1f}s")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        msg = (
            f"Speaker resolution: {sum(by_method.values()):,} turns re-attributed, "
            f"verified share {before['verified_turn_share']:.2%} -> {after['verified_turn_share']:.2%}"
        )
        api.upload_folder(
            folder_path=str(out_root), repo_id=args.dataset_repo, repo_type="dataset", commit_message=msg,
            allow_patterns=["data/*.parquet", "README.md", "terms_cache/**"],
        )
        space_files = ["speakers_index.json", "terms_index.json", "topics_index.json", "speaker_topics.json"]
        for fn in space_files:
            fp = space_dir / fn
            if not fp.exists():
                raise SystemExit(f"Missing {fp}; refusing a partial Space upload.")
        if args.upload_ui:
            for fn in ("index.html", "og-image.png"):
                src_ui = ROOT / "space_static" / fn
                if not src_ui.exists():
                    continue
                if (space_dir / fn).resolve() != src_ui.resolve():
                    shutil.copyfile(src_ui, space_dir / fn)
                space_files.append(fn)
        # one commit for all Space files: the UI never sees a half-updated set of indices
        api.upload_folder(
            folder_path=str(space_dir), repo_id=args.space_repo, repo_type="space",
            allow_patterns=space_files, commit_message=msg,
        )
        print(f"Uploaded dataset + Space files: {space_files}")
    return report


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-hub", action="store_true", help="Download current Parquet files from the HF dataset")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "hf_dataset" / "data"), help="Local Parquet input dir")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "recanon"), help="Output dataset root (data/, README.md, terms_cache/)")
    ap.add_argument(
        "--space-dir", default=str(ROOT / "space_static"),
        help="Output dir for Space JSON indices (default space_static/, so the repo copy never goes stale)",
    )
    ap.add_argument("--report", default=str(ROOT / "data" / "parsed" / "speaker_resolution_report.json"))
    ap.add_argument("--rosters", default=None, help="Path to official_rosters.json (default: data/official_rosters.json)")
    ap.add_argument("--dataset-repo", default=DATASET_REPO)
    ap.add_argument("--space-repo", default=SPACE_REPO)
    ap.add_argument("--skip-aux", action="store_true", help="Skip terms/topics/speaker_topics/term shard rebuild")
    ap.add_argument("--memory-limit", default="1.5GB", help="DuckDB memory limit (spills to --temp-dir beyond it)")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill directory (default: <out-dir>/.duckdb_tmp)")
    ap.add_argument("--threads", type=int, default=0, help="DuckDB threads (0 = DuckDB default)")
    ap.add_argument("--upload", action="store_true", help="Upload results to the HF dataset + Space")
    ap.add_argument("--upload-ui", action="store_true", help="With --upload: also upload space_static/index.html + og-image.png")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
