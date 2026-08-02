"""Download and validate the IEEE-CIS Fraud Detection dataset.

Usage:
    python scripts/download_ieee_cis.py
    python scripts/download_ieee_cis.py --keep-archive
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "ieee-fraud-detection"
ARCHIVE_NAME = f"{COMPETITION}.zip"

EXPECTED_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def validate_dataset(raw_dir: Path) -> list[Path]:
    """Validate that all expected dataset files exist and are non-empty."""
    expected_paths = [raw_dir / filename for filename in EXPECTED_FILES]

    missing = [path.name for path in expected_paths if not path.exists()]
    empty = [
        path.name
        for path in expected_paths
        if path.exists() and path.stat().st_size == 0
    ]

    problems: list[str] = []

    if missing:
        problems.append(f"missing files: {', '.join(missing)}")

    if empty:
        problems.append(f"empty files: {', '.join(empty)}")

    if problems:
        raise RuntimeError(
            "IEEE-CIS dataset validation failed — " + "; ".join(problems)
        )

    return expected_paths


def download_archive(raw_dir: Path) -> Path:
    """Download the Kaggle competition archive into the raw directory."""
    kaggle_executable = shutil.which("kaggle")

    if kaggle_executable is None:
        raise RuntimeError(
            "Kaggle CLI was not found. Install the development dependencies "
            "with: python -m pip install -r requirements-dev.txt"
        )

    archive_path = raw_dir / ARCHIVE_NAME

    command = [
        kaggle_executable,
        "competitions",
        "download",
        "--competition",
        COMPETITION,
        "--path",
        str(raw_dir),
        "--force",
    ]

    print("Downloading IEEE-CIS Fraud Detection data...")
    subprocess.run(command, check=True)

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise RuntimeError(
            f"Kaggle download did not create a valid archive: {archive_path}"
        )

    return archive_path


def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    """Extract the downloaded Kaggle ZIP archive."""
    print(f"Extracting {archive_path.name}...")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded archive is not a valid ZIP file: {archive_path}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate the IEEE-CIS Fraud Detection data."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw competition files are stored.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded ZIP archive after successful extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        archive_path = download_archive(raw_dir)
        extract_archive(archive_path, raw_dir)
        dataset_files = validate_dataset(raw_dir)
    except subprocess.CalledProcessError as exc:
        print(
            "\nKaggle download failed. Confirm that you:\n"
            "1. Accepted the IEEE-CIS competition rules.\n"
            "2. Configured ~/.kaggle/kaggle.json.\n"
            "3. Set its permissions with chmod 600.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    print("\nDataset validation passed:")

    for path in dataset_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  OK  {path.name:<24} {size_mb:>9.1f} MB")

    print(f"\nRaw dataset directory: {raw_dir}")
    print("Raw files are local-only and must not be committed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
