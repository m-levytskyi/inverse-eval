import sys
from pathlib import Path
import glob
import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_experimental_reflectivity import load_experimental_data, plot_reflectivity

def detect_peaks(y, min_prominence=0.2, min_rise=0.05, min_width=3):
    y = np.log10(np.asarray(y))
    n = len(y)
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
                width = end - start
                if prominence >= min_prominence:
                    peaks.append({'index': peak_idx, 'prominence': prominence, 'width': width})
        else:
            i += 1

    return peaks

def has_significant_peaks(y, min_prominence=0.2, min_rise=0.05, min_width=3):
    return len(detect_peaks(y, min_prominence, min_rise, min_width)) > 0

def count_curves_with_peaks(y, min_prominence=0.05, min_rise=0.02, min_width=3):
    count = 0
    if has_significant_peaks(y, min_prominence, min_rise, min_width):
        count += 1
    return count

def plot_and_save_peaks(q, r, peaks, exp_id, output_dir):
    title = f"Reflectivity: {exp_id}"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(q, r, fmt='o', markersize=3, color='black', label='Experimental')
    ax.set_yscale('log')
    ax.set_xlabel('$Q$)')
    ax.set_ylabel('Reflectivity $R$')
    ax.set_title(title)

    for i, peak in enumerate(peaks):
        peak_q = q[peak['index']]
        peak_r = r[peak['index']]
        ax.axvline(x=peak_q, color='r', linestyle='--')
        
        text_label = (
            f"Peak {i+1}:\n"
            f"  Q = {peak_q:.4f}\n"
            f"  Prominence = {peak['prominence']:.2f}\n"
            f"  Width = {peak['width']}"
        )
        
        ax.text(peak_q, peak_r, text_label, fontsize=8, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    ax.legend(["Experimental", "Detected Peaks"])
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{exp_id}_peaks.png"
    plt.savefig(output_path)
    plt.close(fig)

def main():
    output_dir = "peaks_plots/positive"
    os.makedirs(output_dir, exist_ok=True)

    filepaths = sorted(glob.glob('data/MARIA_VIPR_dataset/1/s*_theoretical_curve.dat'))

    positive_curves = []
    negative_curves = []
    all_peaks = []

    for path in tqdm.tqdm(filepaths):
        exp_id = Path(path).stem.split('_')[0]
        q, r, dr, dq = load_experimental_data(path)
        peaks = detect_peaks(r)

        if peaks:
            positive_curves.append((exp_id, r))
            all_peaks.extend(peaks)
            plot_and_save_peaks(q, r, peaks, exp_id, output_dir)
        else:
            negative_curves.append((exp_id, r))
            
    print(f"Positive curves: {len(positive_curves)}, e.g {[exp_id for exp_id, _ in positive_curves[:5]]}")
    print(f"Negative curves: {len(negative_curves)}, e.g {[exp_id for exp_id, _ in negative_curves[:5]]}")

    if all_peaks:
        prominences = [p['prominence'] for p in all_peaks]
        widths = [p['width'] for p in all_peaks]

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
