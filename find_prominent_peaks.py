import sys
from pathlib import Path
import glob
import tqdm

import numpy as np
import matplotlib.pyplot as plt

from plot_experimental_reflectivity import load_experimental_data, plot_reflectivity

def detect_peaks(y, min_prominence=0.05, min_rise=0.02, min_width=3):
    y = np.asarray(y)
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
                if prominence >= min_prominence:
                    peaks.append((peak_idx, prominence))
        else:
            i += 1

    return peaks

def has_significant_peaks(y, min_prominence=0.05, min_rise=0.02, min_width=3):
    return len(detect_peaks(y, min_prominence, min_rise, min_width)) > 0

def count_curves_with_peaks(y, min_prominence=0.05, min_rise=0.02, min_width=3):
    count = 0
    if has_significant_peaks(y, min_prominence, min_rise, min_width):
        count += 1
    return count

def main():
    # title = f"Reflectivity: {data_path.name}"
    # plot_reflectivity(q, r, dr, dq, title=title)

    filepaths = sorted(glob.glob('data/MARIA_VIPR_dataset/1/s*_theoretical_curve.dat'))

    good_curves = []
    bad_curves = []

    for path in tqdm.tqdm(filepaths):
        exp_id = Path(path).stem.split('_')[0]
        q, r, dr, dq = load_experimental_data(path)

        if has_significant_peaks(r):
            good_curves.append((exp_id, r))
        else:
            bad_curves.append((exp_id, r))

    print(f"Good curves: {len(good_curves)}")
    print(f"Bad curves: {len(bad_curves)}")

if __name__ == "__main__":
    main()