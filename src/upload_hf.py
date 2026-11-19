"""Upload the generated KKTC Meclisi Parquet dataset (and optionally raw files) to Hugging Face Hub."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description="Upload KKTC Meclis Tutanak dataset to Hugging Face.")
    ap.add_argument("--repo-id", required=True, help="Target HF dataset repo, e.g. 'username/KKTC_Meclis_Tutanak'")
    ap.add_argument("--private", action="store_true", help="Create as private dataset")
    ap.add_argument("--include-raw", action="store_true", help="Also upload raw files to a separate repo (--raw-repo-id)")
    ap.add_argument("--raw-repo-id", help="Optional raw archive HF repo, e.g. 'username/KKTC_Meclis_Tutanak_raw'")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated on Hugging Face as: {user.get('name') or user.get('fullname')}")

    hf_dir = Path("data/hf_dataset")
    if not (hf_dir / "data/speeches.parquet").exists():
        raise SystemExit("Run src/build_dataset.py first to generate data/hf_dataset/.")

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Uploading {hf_dir} -> https://huggingface.co/datasets/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(hf_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload KKTC Cumhuriyet Meclisi speaker-indexed dataset",
    )
    print(f"Uploaded dataset: https://huggingface.co/datasets/{args.repo_id}")

    if args.include_raw:
        raw_repo = args.raw_repo_id or f"{args.repo_id}_raw"
        api.create_repo(repo_id=raw_repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"Uploading data/raw -> https://huggingface.co/datasets/{raw_repo} ...")
        api.upload_large_folder(
            folder_path="data/raw",
            repo_id=raw_repo,
            repo_type="dataset",
        )
        print(f"Uploaded raw archive: https://huggingface.co/datasets/{raw_repo}")


if __name__ == "__main__":
    main()
