"""
Copy JSON files from paper batches into a dedicated folder.

For each model-group subdirectory in paper_batches/, the script finds the
matching batch folder in batch_inference_results/ and copies only the *.json
files (no plots) to:

    paper_batch_jsons/<model_group>/<batch_folder>/<json files>

Usage:
    python copy_paper_batch_jsons.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).parent
BATCH_RESULTS_DIR = WORKSPACE / "batch_inference_results"
PAPER_BATCHES_DIR = WORKSPACE / "paper_batches"
DEST_DIR = WORKSPACE / "paper_batch_jsons"

# Subdirectories of paper_batches/ that contain batch folders
MODEL_GROUP_NAMES = [
    "reflectorch",
    "NF_baseline",
    "NF_mean_conditioned_sweep",
    "NF_qweighted",
    "NF_qweighted_exp1_alpha2_beta2_sweep",
    "NF_qweighted_exp2_alpha4_beta4_sweep",
    "anaklasis",
]


def collect_paper_batch_names(model_group_dir: Path) -> list[str]:
    """Return names of all batch subdirectories inside a model-group folder."""
    return [
        entry.name
        for entry in sorted(model_group_dir.iterdir())
        if entry.is_dir()
    ]


def copy_jsons(batch_src: Path, batch_dest: Path, dry_run: bool) -> list[str]:
    """Copy *.json files from batch_src to batch_dest. Returns copied file names."""
    json_files = list(batch_src.glob("*.json"))
    if not json_files:
        return []
    if not dry_run:
        batch_dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for json_file in json_files:
        dest_file = batch_dest / json_file.name
        if not dry_run:
            shutil.copy2(json_file, dest_file)
        copied.append(json_file.name)
    return copied


def main(dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    total_copied = 0
    total_missing = 0

    for group_name in MODEL_GROUP_NAMES:
        group_paper_dir = PAPER_BATCHES_DIR / group_name
        if not group_paper_dir.is_dir():
            print(f"[SKIP] paper_batches/{group_name}/ not found")
            continue

        batch_names = collect_paper_batch_names(group_paper_dir)
        if not batch_names:
            print(f"[SKIP] paper_batches/{group_name}/ is empty")
            continue

        print(f"\n{group_name}/ ({len(batch_names)} batches)")

        for batch_name in batch_names:
            src = BATCH_RESULTS_DIR / batch_name
            dest = DEST_DIR / group_name / batch_name

            if not src.is_dir():
                print(f"  [MISSING] {batch_name}")
                total_missing += 1
                continue

            copied = copy_jsons(src, dest, dry_run)
            if copied:
                status = "DRY" if dry_run else "COPY"
                print(f"  [{status}] {batch_name} -> {len(copied)} JSON file(s)")
                total_copied += len(copied)
            else:
                print(f"  [NO JSON] {batch_name}")

    print(f"\nDone. {total_copied} JSON file(s) {'would be ' if dry_run else ''}copied, "
          f"{total_missing} batch folder(s) not found in batch_inference_results/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied without writing any files.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
