import sys
from pathlib import Path
import glob
import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_experimental_reflectivity import load_experimental_data, plot_reflectivity


def detect_peaks(
    y, min_prominence=0.2, min_rise=0.05, min_width=10, analyze_first_half=True
):
    y = np.log10(np.asarray(y))
    n = len(y)

    # If analyze_first_half is True, only analyze the first half of the data
    if analyze_first_half:
        n = n // 2

    peaks = []

    i = 1
    while i < n - 1:
        if y[i] > y[i - 1] + min_rise:
            start = i - 1
            while i < n - 1 and y[i + 1] >= y[i]:
                i += 1
            peak_idx = i
            while i < n - 1 and y[i + 1] <= y[i]:
                i += 1
            end = i

            if end - start >= min_width:
                y_peak = y[peak_idx]
                y_before = y[start]
                y_after = y[end]
                prominence = y_peak - max(y_before, y_after)
                width = (peak_idx - start) * 2
                if prominence >= min_prominence:
                    peaks.append(
                        {"index": peak_idx, "prominence": prominence, "width": width}
                    )
        else:
            i += 1

    return peaks


def has_significant_peaks(
    y, min_prominence=0.2, min_rise=0.05, min_width=3, analyze_first_half=True
):
    return (
        len(detect_peaks(y, min_prominence, min_rise, min_width, analyze_first_half))
        > 0
    )


def count_curves_with_peaks(
    y, min_prominence=0.05, min_rise=0.02, min_width=3, analyze_first_half=True
):
    count = 0
    if has_significant_peaks(
        y, min_prominence, min_rise, min_width, analyze_first_half
    ):
        count += 1
    return count


def find_experiments_with_prominent_peaks(
    layer_count,
    data_directory="data",
    min_prominence=0.2,
    min_rise=0.05,
    min_width=10,
    analyze_first_half=True,
    verbose=True,
):
    """
    Find experiments that have prominent peaks in their reflectivity curves.

    Args:
        layer_count: Number of layers (1 or 2)
        data_directory: Path to the data directory
        min_prominence: Minimum prominence for peak detection
        min_rise: Minimum rise for peak detection
        min_width: Minimum width for peak detection
        analyze_first_half: Whether to analyze only the first half of the data
        verbose: Whether to print progress and statistics

    Returns:
        List of experiment IDs that have prominent peaks
    """
    # Search patterns to try (prioritizing theoretical curves, then experimental)
    # Also checking different directory structures
    patterns = [
        f"{data_directory}/MARIA_VIPR_dataset/{layer_count}/s*_theoretical_curve.dat",
        f"{data_directory}/test_data/{layer_count}/s*_theoretical_curve.dat",
        f"{data_directory}/s*_theoretical_curve.dat",
        f"{data_directory}/MARIA_VIPR_dataset/{layer_count}/s*_experimental_curve.dat",
        f"{data_directory}/test_data/{layer_count}/s*_experimental_curve.dat",
        f"{data_directory}/s*_experimental_curve.dat",
    ]

    filepaths = []
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            filepaths = found
            if verbose:
                print(f"Found {len(filepaths)} files using pattern: {pattern}")
            break

    if verbose:
        print(f"Scanning {len(filepaths)} experiments for prominent peaks...")
        print(f"Layer count: {layer_count}")
        print(
            f"Parameters: prominence≥{min_prominence}, rise≥{min_rise}, width≥{min_width}"
        )
        print(f"Analyzing: {'first half' if analyze_first_half else 'full curve'}")

    experiments_with_peaks = []
    experiments_without_peaks = []
    all_peaks = []

    iterator = tqdm.tqdm(filepaths) if verbose else filepaths

    for path in iterator:
        exp_id = Path(path).stem.split("_")[0]
        try:
            q, r, dr, dq = load_experimental_data(path)
            peaks = detect_peaks(
                r, min_prominence, min_rise, min_width, analyze_first_half
            )

            if peaks:
                experiments_with_peaks.append(exp_id)
                all_peaks.extend(peaks)
            else:
                experiments_without_peaks.append(exp_id)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {exp_id}: {e}")
            experiments_without_peaks.append(exp_id)

    if verbose:
        print(f"\nResults:")
        print(f"  Experiments with prominent peaks: {len(experiments_with_peaks)}")
        print(
            f"  Experiments without prominent peaks: {len(experiments_without_peaks)}"
        )
        print(f"  Examples with peaks: {experiments_with_peaks[:5]}")

        if all_peaks:
            prominences = [p["prominence"] for p in all_peaks]
            print(f"  Peak statistics: {len(all_peaks)} total peaks")
            print(
                f"    Prominence range: {np.min(prominences):.3f} - {np.max(prominences):.3f}"
            )
            print(f"    Mean prominence: {np.mean(prominences):.3f}")

    return experiments_with_peaks


def plot_and_save_peaks(
    q, r, peaks, exp_id, output_dir, analyze_first_half=True, paper_mode=False
):
    """
    Plot and save peak detection visualization.

    Args:
        q: Q values
        r: Reflectivity values
        peaks: List of detected peaks
        exp_id: Experiment ID
        output_dir: Output directory
        analyze_first_half: Whether only first half was analyzed
        paper_mode: Use paper styling if True
    """
    title = f"Reflectivity: {exp_id}"

    fig = _create_peaks_plot(
        q, r, peaks, exp_id, title, analyze_first_half, paper_mode=paper_mode
    )

    if paper_mode:
        output_path = Path(output_dir) / f"{exp_id}_peaks.pdf"
    else:
        output_path = Path(output_dir) / f"{exp_id}_peaks.png"
    
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def _create_peaks_plot(
    q, r, peaks, exp_id, title, analyze_first_half, paper_mode=False
):
    """Internal function to create peaks plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # If analyzing first half, only plot the first half
    if analyze_first_half:
        half_idx = len(q) // 2
        q_plot = q[:half_idx]
        r_plot = r[:half_idx]
        if not paper_mode:
            title += " (First Half)"
    else:
        q_plot = q
        r_plot = r

    ax.errorbar(q_plot, r_plot, fmt="o", markersize=3, label="Experimental")
    ax.set_yscale("log")
    ax.set_xlabel("$Q$ [Å$^{-1}$]")
    ax.set_ylabel("$R$")
    if not paper_mode:
        ax.set_title(title)

    # Only highlight the first peak (if any exist)
    if peaks:
        peak = peaks[0]
        peak_q = q[peak["index"]]
        peak_r = r[peak["index"]]

        # More visible highlighting: hollow circle with red contour, 2x bigger
        ax.scatter(
            [peak_q],
            [peak_r],
            s=300,
            facecolors="none",
            edgecolors="red",
            zorder=5,
            linewidths=2.5,
        )

    # Fixed legend (black is experimental, not "detected peaks")
    if not paper_mode:
        ax.legend(loc="best")

    # Remove ticks
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()

    return fig


def main():

    layers_count = 2
    filepaths = sorted(
        glob.glob(f"data/MARIA_VIPR_dataset/{layers_count}/s*_theoretical_curve.dat")
    )

    output_dir = f"peaks_plots/{layers_count}/positive"
    os.makedirs(output_dir, exist_ok=True)

    positive_curves = []
    negative_curves = []
    all_peaks = []

    for path in tqdm.tqdm(filepaths):
        exp_id = Path(path).stem.split("_")[0]
        q, r, dr, dq = load_experimental_data(path)
        peaks = detect_peaks(r, analyze_first_half=True)

        if peaks:
            positive_curves.append((exp_id, r))
            all_peaks.extend(peaks)
            plot_and_save_peaks(
                q, r, peaks, exp_id, output_dir, analyze_first_half=True
            )
        else:
            negative_curves.append((exp_id, r))

    print(
        f"Positive curves: {len(positive_curves)}, e.g {[exp_id for exp_id, _ in positive_curves[:5]]}"
    )
    print(
        f"Negative curves: {len(negative_curves)}, e.g {[exp_id for exp_id, _ in negative_curves[:5]]}"
    )

    if all_peaks:
        prominences = [p["prominence"] for p in all_peaks]
        widths = [p["width"] for p in all_peaks]

        print("\n--- Peak Statistics ---")
        print("\nProminence:")
        print(f"  Min: {np.min(prominences):.4f}")
        print(f"  Max: {np.max(prominences):.4f}")
        print(f"  Mean: {np.mean(prominences):.4f}")
        print(f"  Std Dev: {np.std(prominences):.4f}")
        print(f"  Median: {np.median(prominences):.4f}")

        print("\nWidth:")
        print(f"  Min: {np.min(widths)}")
        print(f"  Max: {np.max(widths)}")
        print(f"  Mean: {np.mean(widths):.4f}")
        print(f"  Std Dev: {np.std(widths):.4f}")
        print(f"  Median: {np.median(widths)}")


if __name__ == "__main__":
    main()
