import argparse
import os
import sys
from huggingface_hub import snapshot_download

def download_dataset(dl_dir):
    """
    Downloads the MLS English dataset from Hugging Face to `$dl_dir/mls_english`.
    """
    repo_id = 'parler-tts/mls_eng'
    local_dataset_dir = os.path.join(dl_dir, 'mls_english')

    print(f"Attempting to download '{repo_id}' to '{local_dataset_dir}'...")

    # Ensure the parent directory exists
    os.makedirs(dl_dir, exist_ok=True)

    try:
        # snapshot_download handles LFS and large files robustly
        # local_dir_use_symlinks=False is generally safer for datasets,
        # especially on network file systems or if you intend to move the data
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dataset_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded '{repo_id}' to '{local_dataset_dir}'")
    except Exception as e:
        print(f"Error downloading dataset '{repo_id}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MLS English dataset from Hugging Face."
    )
    parser.add_argument(
        "--dl-dir",
        type=str,
        required=True,
        help="The base directory where the 'mls_english' dataset will be downloaded.",
    )
    args = parser.parse_args()

    download_dataset(args.dl_dir)