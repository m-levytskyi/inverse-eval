#!/usr/bin/env python3
"""
Replot specific batch results in paper style.
Wrapper around replot_batch_results.py for specific batches.
"""

from pathlib import Path
from replot_batch_results import replot_batch_results


def find_batch_directory(batch_id, base_dir="batch_inference_results"):
    """Find batch directory by batch ID."""
    base_path = Path(base_dir)
    pattern = f"{batch_id:03d}_*"

    matches = list(base_path.glob(pattern))
    if matches:
        return matches[0]
    return None


def replot_batch_paper(batch_id):
    """Replot batch using replot_batch_results with paper_mode=True."""
    batch_dir = find_batch_directory(batch_id)

    if not batch_dir:
        print(f"Batch {batch_id} not found")
        return

    print(f"\nProcessing batch {batch_id}: {batch_dir.name}")

    # Use replot_batch_results with paper mode enabled
    replot_batch_results(batch_dir, output_dir=None, paper_mode=True)


def main():
    """Replot specified batches in paper style."""
    # Batch ranges: 269-286, 232-250, 291, 102-119
    batch_ids = []

    # Add range 269-286
    batch_ids.extend(range(269, 287))

    # Add range 232-250
    batch_ids.extend(range(232, 251))

    # Add batch 291
    batch_ids.append(291)

    # Add range 102-119
    batch_ids.extend(range(102, 120))

    print(f"Replotting {len(batch_ids)} batches in paper style")
    print(f"Batch IDs: {sorted(set(batch_ids))}")

    success_count = 0
    for batch_id in sorted(set(batch_ids)):
        try:
            replot_batch_paper(batch_id)
            success_count += 1
        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")

    print(f"\nCompleted: {success_count}/{len(set(batch_ids))} batches")


if __name__ == "__main__":
    main()
