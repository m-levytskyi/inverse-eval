#!/usr/bin/env python3
"""
Per-parameter MAPE heatmap for 30 % priors / no SLD fix / no prominent.

    Rows   : model categories
    Columns: individual parameters
    Values : mean constraint MAPE per parameter (%)

Usage:
    python plot_aggregate_heatmap.py
    python plot_aggregate_heatmap.py --stats path/to/aggregate_stats.json --output-dir figures/
"""

import argparse
import json
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pathlib import Path

matplotlib.use("pdf")
paper_mplstyle = Path(__file__).parent / "paper.mplstyle"
plt.style.use(["science", str(paper_mplstyle)])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "overall":         "Overall",
    "param:thickness": "Thickness",
    "param:amb_rough": "Ambient roughness",
    "param:sub_rough": "Substrate roughness",
    "param:layer_sld": "Layer SLD",
    "param:sub_sld":   "Sub. SLD",
}

MODEL_LABELS = {
    "reflectorch":                          r"reflectorch",
    "NF_baseline":                          r"NF baseline",
    "NF_qweighted":                         r"NF q-weighted $\alpha2\beta3$",
    "NF_qweighted_exp1_alpha2_beta2_sweep": r"NF q-wt $\alpha2\beta2$",
    "NF_qweighted_exp2_alpha4_beta4_sweep": r"NF q-wt $\alpha4\beta4$",
    "NF_mean_conditioned_sweep":            r"NF mean-cond.",
}

TARGET_SETUP = "width30_sldnone"

# ---------------------------------------------------------------------------
# Matplotlib gallery helpers (from matplotlib.org/stable/gallery/…/image_annotated_heatmap.html)
# ---------------------------------------------------------------------------

def _contrast_color(im, val) -> str:
    """Return 'black' or 'white' based on WCAG relative luminance of the cell color."""
    rgba = im.cmap(im.norm(val))
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * linearize(rgba[0]) + 0.7152 * linearize(rgba[1]) + 0.0722 * linearize(rgba[2])
    return "black" if lum > 0.179 else "white"


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.ma.is_masked(val) or (isinstance(val, float) and np.isnan(val)):
                continue
            kw["color"] = _contrast_color(im, val)
            texts.append(im.axes.text(j, i, valfmt(val, None), **kw))
    return texts


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_matrix(stats: dict) -> tuple[np.ndarray, list[str], list[str]]:
    param_keys = list(PARAM_LABELS.keys())
    # preserve insertion order of MODEL_LABELS, skip models absent from stats
    models = [m for m in MODEL_LABELS if m in stats and TARGET_SETUP in stats[m].get("by_setup", {})]

    matrix = np.full((len(models), len(param_keys)), np.nan)
    for i, model in enumerate(models):
        mape = stats[model]["by_setup"][TARGET_SETUP]["mape"]
        for j, pk in enumerate(param_keys):
            # "overall" is a top-level key; per-param keys are e.g. "param:thickness"
            entry = mape.get(pk, {})
            val = entry.get("mean")
            if val is not None:
                matrix[i, j] = val

    row_labels = [MODEL_LABELS.get(m, m) for m in models]
    col_labels = list(PARAM_LABELS.values())
    return matrix, row_labels, col_labels


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(stats: dict, output_path: Path):
    matrix, row_labels, col_labels = build_matrix(stats)

    if matrix.size == 0:
        print(f"[warn] No models found for setup '{TARGET_SETUP}'; nothing to plot.")
        return

    n_models, n_params = matrix.shape
    default_w, default_h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(n_params * default_w / 4, n_models * default_h / 3))

    im, _ = heatmap(
        np.ma.array(matrix, mask=np.isnan(matrix)),
        row_labels, col_labels, ax=ax,
        cmap="viridis",
        cbarlabel=r"Mean MAPE (\%)",
    )
    annotate_heatmap(im, data=matrix, valfmt="{x:.2f}")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(stats_path: str = "aggregate_stats.json", output_dir: str = "."):
    stats_file = Path(stats_path)
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    with open(stats_file) as f:
        stats = json.load(f)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot(stats, out / "heatmap_perparam_mape_width30.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-parameter MAPE heatmap from aggregate_stats.json"
    )
    parser.add_argument("--stats", default="aggregate_stats.json", metavar="FILE")
    parser.add_argument("--output-dir", default=".", metavar="DIR")
    args = parser.parse_args()
    run(stats_path=args.stats, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
