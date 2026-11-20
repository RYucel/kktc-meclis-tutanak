"""Deploy the KKTC Meclisi speaker-based search web app to Hugging Face Spaces."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="stevevaius/KKTC-Meclis-Tutanak-Arama")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated on Hugging Face as: {user.get('name')}")

    space_dir = Path("space_static")
    assert (space_dir / "index.html").exists() and (space_dir / "speakers_index.json").exists()
    assert (space_dir / "terms_index.json").exists() and (space_dir / "topics_index.json").exists()

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )
    print(f"Uploading {space_dir} -> https://huggingface.co/spaces/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Deploy serverless DuckDB-WASM KKTC Meclis Tutanak search Static Space",
    )
    print(f"Space deployed: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
