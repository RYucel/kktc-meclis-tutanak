"""Deploy the KKTC Meclisi speaker-based search web app to Hugging Face Spaces.

By default only the UI (index.html + README.md) is uploaded. The JSON indices (speakers/terms/topics/
speaker_topics) are produced by the data pipeline (sync_incremental.py / recanonicalize.py --upload)
and must always match the Parquet files on the dataset; uploading a stale local copy over them would
break speaker search for every renamed or merged speaker. Use --with-indices only after rebuilding
them locally; the script refuses if the local speakers_index.json is older than the live one.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

INDEX_FILES = ("speakers_index.json", "terms_index.json", "topics_index.json", "speaker_topics.json")


def _stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("stats", {})


def local_is_stale(local: dict, live: dict) -> str | None:
    """Return a reason string if the local index must not replace the live one."""
    for key in ("latest_date", "synced_at"):
        if (local.get(key) or "") < (live.get(key) or ""):
            return f"local {key}={local.get(key)} is older than live {key}={live.get(key)}"
    if local.get("speeches", 0) < live.get("speeches", 0):
        return f"local index covers {local.get('speeches')} speeches, live covers {live.get('speeches')}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="stevevaius/KKTC-Meclis-Tutanak-Arama")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--with-indices", action="store_true", help="Also upload the JSON indices from space_static/")
    ap.add_argument("--force", action="store_true", help="Skip the live-vs-local freshness check")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated on Hugging Face as: {user.get('name')}")

    space_dir = Path("space_static")
    assert (space_dir / "index.html").exists()

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )

    patterns = ["index.html", "README.md"]
    if args.with_indices:
        for fn in INDEX_FILES:
            assert (space_dir / fn).exists(), f"missing {space_dir / fn}"
        if not args.force:
            try:
                live = _stats(Path(hf_hub_download(args.repo_id, "speakers_index.json", repo_type="space", force_download=True)))
            except Exception as e:  # first deploy: nothing live yet
                print(f"(no live speakers_index.json to compare: {e})")
                live = {}
            reason = local_is_stale(_stats(space_dir / "speakers_index.json"), live)
            if reason:
                raise SystemExit(
                    f"Refusing to upload indices: {reason}.\n"
                    "Rebuild them (python src/recanonicalize.py --from-hub --space-dir space_static) or pass --force."
                )
        patterns += list(INDEX_FILES)

    print(f"Uploading {patterns} from {space_dir} -> https://huggingface.co/spaces/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=args.repo_id,
        repo_type="space",
        allow_patterns=patterns,
        commit_message="Deploy serverless DuckDB-WASM KKTC Meclis Tutanak search Static Space",
    )
    print(f"Space deployed: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
