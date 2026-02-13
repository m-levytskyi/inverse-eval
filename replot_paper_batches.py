#!/usr/bin/env python3
"""Replot specific batch results into categorized output directories.
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


def get_batch_category(batch_id):
    """Determine which category a batch belongs to."""
    if 102 <= batch_id <= 119:
        return "reflectorch"
    elif 269 <= batch_id <= 286:
        return "NF_baseline"
    elif 232 <= batch_id <= 250:
        return "NF_qweighted"
    elif batch_id == 291:
        return "NF_mean_conditioned"
    elif batch_id == 299:
        return "NF_qweighted_exp1_alpha2_beta2"
    else:
        return "other"


def replot_batch_paper(batch_id, base_output_dir="paper_batches"):
    """Replot batch results into categorized output directories."""
    batch_dir = find_batch_directory(batch_id)

    if not batch_dir:
        print(f"Batch {batch_id} not found")
        return

    print(f"\nProcessing batch {batch_id}: {batch_dir.name}")
    
    category = get_batch_category(batch_id)
    output_dir = Path(base_output_dir) / category / batch_dir.name

    replot_batch_results(batch_dir, output_dir=output_dir)


def main():
    """Replot specified batches into categorized output directories."""
    # Batch ranges: 269-286, 232-250, 291, 102-119
    batch_ids = []

    # Add range 269-286
    batch_ids.extend(range(269, 287))

    # Add range 232-250
    batch_ids.extend(range(232, 251))

    # Add batch 291
    batch_ids.append(291)

    # Add batch 299 (NF qweighted exp1, alpha=beta=2)
    batch_ids.append(299)

    # Add range 102-119
    batch_ids.extend(range(102, 120))

    print(f"Replotting {len(batch_ids)} batches")
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
