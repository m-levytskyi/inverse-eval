#!/usr/bin/env python3
"""
Plot reflectivity curves from experimental data files (3- or 4-column format).

Usage:
    python plot_experimental_reflectivity.py path/to/datafile.dat [--paper] [--save output.pdf]

Supports both 3-column (Q, R, dR) and 4-column (Q, R, dR, dQ) formats.
"""

import sys
import argparse
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


def plot_reflectivity(q, r, dr, dq=None, title=None, save_path=None, paper_mode=False):
    """
    Plot reflectivity curve with error bars.

    Args:
        q: Q values (momentum transfer)
        r: Reflectivity values
        dr: Reflectivity uncertainties
        dq: Optional Q uncertainties
        title: Plot title
        save_path: Optional path to save figure
        paper_mode: Use paper styling if True
    """
    fig = _create_reflectivity_plot(q, r, dr, dq, title)
    
    if save_path:
        if paper_mode:
            save_path = Path(save_path).with_suffix(".pdf")
            plt.savefig(save_path)
            print(f"paper figure saved to: {save_path}")
        else:
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    return fig


def _create_reflectivity_plot(q, r, dr, dq=None, title=None):
    """Internal function to create reflectivity plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if dq is not None:
        ax.errorbar(q, r, yerr=dr, xerr=dq, fmt="o", markersize=3, label="Experimental")
    else:
        ax.errorbar(q, r, yerr=dr, fmt="o", markersize=3, label="Experimental")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q$ ($\AA^{-1}$)")
    ax.set_ylabel("Reflectivity $R$")
    ax.set_title(title or "Experimental Reflectivity Curve")
    ax.legend()
    ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot reflectivity curves from experimental data"
    )
    parser.add_argument("data_file", help="Path to data file (.dat)")
    parser.add_argument(
        "--paper", action="store_true", help="Use paper/publication styling"
    )
    parser.add_argument(
        "--save", help="Save figure to file (extension determines format)"
    )

    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"File not found: {data_path}")
        sys.exit(1)

    q, r, dr, dq = load_experimental_data(data_path)
    title = f"Reflectivity: {data_path.name}"
    plot_reflectivity(
        q, r, dr, dq, title=title, save_path=args.save, paper_mode=args.paper
    )


if __name__ == "__main__":
    main()
