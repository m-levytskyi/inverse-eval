#!/usr/bin/env python3
"""
Plot reflectivity curves from experimental data files (3- or 4-column format).

Usage:
    python plot_experimental_reflectivity.py path/to/datafile.dat

Supports both 3-column (Q, R, dR) and 4-column (Q, R, dR, dQ) formats.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_experimental_data(data_path):
    data = np.loadtxt(data_path)
    if data.shape[1] == 3:
        q = data[:, 0]
        r = data[:, 1]
        dr = data[:, 2]
        dq = None
    elif data.shape[1] == 4:
        q = data[:, 0]
        r = data[:, 1]
        dr = data[:, 2]
        dq = data[:, 3]
    else:
        raise ValueError(f"Unsupported data format: {data.shape[1]} columns")
    return q, r, dr, dq

def plot_reflectivity(q, r, dr, dq=None, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if dq is not None:
        ax.errorbar(q, r, yerr=dr, xerr=dq, fmt='o', markersize=3, color='black', label='Experimental')
    else:
        ax.errorbar(q, r, yerr=dr, fmt='o', markersize=3, color='black', label='Experimental')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Q$ ($\AA^{-1}$)')
    ax.set_ylabel('Reflectivity $R$')
    ax.set_title(title or 'Experimental Reflectivity Curve')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_experimental_reflectivity.py path/to/datafile.dat")
        sys.exit(1)
    data_path = Path(sys.argv[1])
    if not data_path.exists():
        print(f"File not found: {data_path}")
        sys.exit(1)
    q, r, dr, dq = load_experimental_data(data_path)
    title = f"Reflectivity: {data_path.name}"
    plot_reflectivity(q, r, dr, dq, title=title)

if __name__ == "__main__":
    main()