#!/usr/bin/env python3
"""
Compare overall constraint MAPE distributions for:
    - anaklasis (pickle predictions)
    - PANPE baseline (NF_baseline batch summaries)

Default comparison uses the 30% prior width / no SLD fix / no prominent setup.

Usage:
  python plot_constraint_mape_comparison.py
  python plot_constraint_mape_comparison.py --prior-width 30 --output figures/
  python plot_constraint_mape_comparison.py --pickle-file results_exp_1L_fitconstraints0_width0.3_simple.pkl
"""

import argparse
import contextlib
import io
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from error_calculation import calculate_parameter_metrics


PARAM_NAMES_5 = ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]
PICKLE_TO_REFLECTORCH_INDICES = [3, 1, 4, 2, 5]
# Constraint MAPE uses fixed constraint widths by parameter type, so numeric bounds are placeholders.
PLACEHOLDER_PRIOR_BOUNDS_5 = [[0.0, 1.0] for _ in PARAM_NAMES_5]


def load_constraint_mapes_from_summary(summary_file: Path) -> list[float]:
    """Read overall constraint MAPE values from batch_summary_1layer.json."""
    with open(summary_file, "r") as f:
        summary = json.load(f)

    mapes: list[float] = []
    for exp_info in summary.get("debug_info", []):
        pm = exp_info.get("param_metrics", {})
        overall = pm.get("overall", {})
        value = overall.get("constraint_mape")
        if value is not None:
            mapes.append(float(value))

    return mapes


def parse_setup_from_dir_name(dir_name: str) -> dict:
    """Parse prior width / SLD fix / prominent from a batch directory name."""
    parts = dir_name.split("_")

    prior_width = None
    fix_sld = "none"
    prominent = False

    for token in parts:
        if token.endswith("constraint"):
            try:
                prior_width = int(token.replace("constraint", ""))
            except ValueError:
                continue
        elif token == "PROMINENT":
            prominent = True
        elif token == "backSLDfix":
            fix_sld = "back"
        elif token == "allSLDfix":
            fix_sld = "all"

    return {
        "prior_width": prior_width,
        "fix_sld": fix_sld,
        "prominent": prominent,
    }


def find_matching_batch_dir(
    category_dir: Path,
    prior_width: int,
    fix_sld: str,
    prominent: bool,
) -> Path:
    """Find the latest batch directory matching the requested setup."""
    candidates = sorted([d for d in category_dir.iterdir() if d.is_dir()])
    matches = []
    for d in candidates:
        setup = parse_setup_from_dir_name(d.name)
        if (
            setup["prior_width"] == prior_width
            and setup["fix_sld"] == fix_sld
            and setup["prominent"] == prominent
        ):
            matches.append(d)

    if matches:
        return max(matches, key=lambda p: int(p.name.split("_")[0]))

    raise FileNotFoundError(
        f"No batch found in {category_dir} with setup "
        f"prior_width={prior_width}, fix_sld={fix_sld}, prominent={prominent}."
    )


def load_anaklasis_constraint_mapes(pickle_file: Path) -> list[float]:
    """Compute overall constraint MAPEs from anaklasis pickle predictions."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    targets = data[0]
    predictions = data[1]
    bounds_flags = data[5]

    mapes: list[float] = []

    for position in range(len(bounds_flags)):
        if int(bounds_flags[position]) == 1:
            continue

        true_vals = [float(targets[position][i]) for i in PICKLE_TO_REFLECTORCH_INDICES]
        pred_vals = [float(predictions[position][i]) for i in PICKLE_TO_REFLECTORCH_INDICES]

        # Convert SLD units to 10^-6 Å^-2
        true_vals[3] *= 1e6
        true_vals[4] *= 1e6
        pred_vals[3] *= 1e6
        pred_vals[4] *= 1e6

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = calculate_parameter_metrics(
                pred_params=pred_vals,
                true_params=true_vals,
                param_names=PARAM_NAMES_5,
                prior_bounds=PLACEHOLDER_PRIOR_BOUNDS_5,
                priors_type="constraint_based",
            )

        value = metrics.get("overall", {}).get("constraint_mape")
        if value is not None:
            mapes.append(float(value))

    return mapes


def summary_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    arr = np.array(values, dtype=float)
    return (
        f"n={len(arr)}, mean={np.mean(arr):.2f}%, "
        f"median={np.median(arr):.2f}%, std={np.std(arr):.2f}%"
    )


def _get_mape_ranges_and_labels() -> tuple[list[int], list[str]]:
    mape_ranges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
    return mape_ranges, range_labels


def _count_mapes_in_ranges(mapes: list[float], mape_ranges: list[int]) -> list[int]:
    return [
        sum(1 for m in mapes if mape_ranges[i] <= m < mape_ranges[i + 1])
        for i in range(len(mape_ranges) - 1)
    ]


def plot_comparison_histogram(
    output_path: Path,
    anaklasis_mapes: list[float],
    panpe_baseline_mapes: list[float],
    prior_width: int,
    fix_sld: str,
    prominent: bool,
    bins: int,
):
    if not anaklasis_mapes or not panpe_baseline_mapes:
        raise ValueError("Both Anaklasis and PANPE baseline MAPE lists must be non-empty.")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    combined = np.array(anaklasis_mapes + panpe_baseline_mapes, dtype=float)
    x_max = float(np.percentile(combined, 99.5))
    x_max = max(x_max, 1.0)
    bin_edges = np.linspace(0.0, x_max, bins + 1).tolist()

    ax.hist(
        panpe_baseline_mapes,
        bins=bin_edges,
        alpha=0.45,
        color="#7f8c8d",
        edgecolor="black",
        linewidth=0.4,
        label=f"PANPE baseline (n={len(panpe_baseline_mapes)})",
    )
    ax.hist(
        anaklasis_mapes,
        bins=bin_edges,
        alpha=0.45,
        color="#2e86de",
        edgecolor="black",
        linewidth=0.4,
        label=f"Anaklasis (n={len(anaklasis_mapes)})",
    )

    panpe_mean = float(np.mean(panpe_baseline_mapes))
    anaklasis_mean = float(np.mean(anaklasis_mapes))
    ax.axvline(panpe_mean, color="#4d5656", linestyle="--", linewidth=1.8)
    ax.axvline(anaklasis_mean, color="#1f618d", linestyle="--", linewidth=1.8)

    prominent_text = "prominent" if prominent else "no prominent"
    ax.set_title(
        "Constraint MAPE Histogram: PANPE Baseline vs Anaklasis\n"
        f"{prior_width}% prior, fix_sld={fix_sld}, {prominent_text}"
    )
    ax.set_xlabel("Overall Constraint MAPE (%)")
    ax.set_ylabel("Number of Experiments")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    stats_text = (
        f"PANPE baseline: {summary_stats(panpe_baseline_mapes)}\n"
        f"Anaklasis: {summary_stats(anaklasis_mapes)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_random_style_range_comparison(
    output_path: Path,
    anaklasis_mapes: list[float],
    panpe_baseline_mapes: list[float],
    prior_width: int,
    fix_sld: str,
    prominent: bool,
):
    if not anaklasis_mapes or not panpe_baseline_mapes:
        raise ValueError("Both Anaklasis and PANPE baseline MAPE lists must be non-empty.")

    mape_ranges, range_labels = _get_mape_ranges_and_labels()
    anaklasis_counts = _count_mapes_in_ranges(anaklasis_mapes, mape_ranges)
    panpe_counts = _count_mapes_in_ranges(panpe_baseline_mapes, mape_ranges)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    x = np.arange(len(range_labels))
    width = 0.35

    ax.bar(
        x - width / 2,
        panpe_counts,
        width,
        alpha=0.8,
        color="#7f8c8d",
        label="PANPE baseline",
    )
    ax.bar(
        x + width / 2,
        anaklasis_counts,
        width,
        alpha=0.8,
        color="#2e86de",
        label="Anaklasis",
    )

    prominent_text = "prominent" if prominent else "no prominent"
    ax.set_title(
        "Constraint MAPE Range Comparison (Random-Style)\n"
        f"{prior_width}% prior, fix_sld={fix_sld}, {prominent_text}"
    )
    ax.set_xlabel("MAPE Range (%)")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.legend()

    stats_text = (
        f"PANPE baseline mean: {np.mean(panpe_baseline_mapes):.2f}%\n"
        f"Anaklasis mean: {np.mean(anaklasis_mapes):.2f}%"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_median_comparison_bar(
    output_path: Path,
    anaklasis_mapes: list[float],
    panpe_baseline_mapes: list[float],
    prior_width: int,
    fix_sld: str,
    prominent: bool,
):
    if not anaklasis_mapes or not panpe_baseline_mapes:
        raise ValueError("Both Anaklasis and PANPE baseline MAPE lists must be non-empty.")

    anaklasis_median = float(np.median(anaklasis_mapes))
    panpe_median = float(np.median(panpe_baseline_mapes))

    labels = ["Anaklasis", "PANPE baseline"]
    medians = [anaklasis_median, panpe_median]
    colors = ["#2e86de", "#7f8c8d"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(labels, medians, color=colors, alpha=0.85, width=0.62)

    y_max = max(medians) * 1.25 if max(medians) > 0 else 1.0
    ax.set_ylim(0, y_max)
    prominent_text = "prominent" if prominent else "no prominent"
    ax.set_title(
        "Median Constraint MAPE Comparison\n"
        f"{prior_width}% prior, fix_sld={fix_sld}, {prominent_text}"
    )
    ax.set_ylabel("Median Constraint MAPE (%)")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_max * 0.02,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(
    base_dir: str = ".",
    prior_width: int = 30,
    fix_sld: str = "none",
    prominent: bool = False,
    pickle_file: str = "results_exp_1L_fitconstraints0_width0.3_simple.pkl",
    output: str = "paper_batches/constraint_mape_histogram_anaklasis_vs_panpe_baseline.pdf",
    output_random_style: str = "paper_batches/constraint_mape_random_style_anaklasis_vs_panpe_baseline.pdf",
    output_median: str = "paper_batches/constraint_mape_median_anaklasis_vs_panpe_baseline.pdf",
    bins: int = 40,
):
    root = Path(base_dir)

    nf_dir = root / "paper_batch_jsons" / "NF_baseline"
    pkl_file = root / pickle_file

    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")

    nf_batch = find_matching_batch_dir(nf_dir, prior_width, fix_sld, prominent)
    nf_summary = nf_batch / "batch_summary_1layer.json"

    if not nf_summary.exists():
        raise FileNotFoundError(f"Missing summary file: {nf_summary}")

    print(f"Using PANPE baseline batch: {nf_batch.name}")
    print(f"Using anaklasis pickle: {pkl_file.name}")

    anaklasis_mapes = load_anaklasis_constraint_mapes(pkl_file)
    panpe_baseline_mapes = load_constraint_mapes_from_summary(nf_summary)

    print(f"{'Anaklasis':<22} -> {summary_stats(anaklasis_mapes)}")
    print(f"{'PANPE baseline':<22} -> {summary_stats(panpe_baseline_mapes)}")

    output_path = root / output
    output_random_style_path = root / output_random_style
    output_median_path = root / output_median

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_random_style_path.parent.mkdir(parents=True, exist_ok=True)
    output_median_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparison_histogram(
        output_path=output_path,
        anaklasis_mapes=anaklasis_mapes,
        panpe_baseline_mapes=panpe_baseline_mapes,
        prior_width=prior_width,
        fix_sld=fix_sld,
        prominent=prominent,
        bins=bins,
    )
    plot_random_style_range_comparison(
        output_path=output_random_style_path,
        anaklasis_mapes=anaklasis_mapes,
        panpe_baseline_mapes=panpe_baseline_mapes,
        prior_width=prior_width,
        fix_sld=fix_sld,
        prominent=prominent,
    )
    plot_median_comparison_bar(
        output_path=output_median_path,
        anaklasis_mapes=anaklasis_mapes,
        panpe_baseline_mapes=panpe_baseline_mapes,
        prior_width=prior_width,
        fix_sld=fix_sld,
        prominent=prominent,
    )

    print(f"Saved overlap histogram plot -> {output_path}")
    print(f"Saved random-style comparison plot -> {output_random_style_path}")
    print(f"Saved median comparison plot -> {output_median_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot constraint MAPE histogram for PANPE baseline vs Anaklasis."
    )
    parser.add_argument("--base-dir", default=".", help="Project root directory")
    parser.add_argument("--prior-width", type=int, default=30, help="Constraint width in percent")
    parser.add_argument(
        "--fix-sld",
        choices=["none", "back", "all"],
        default="none",
        help="SLD fix mode for PANPE baseline batch selection",
    )
    parser.add_argument(
        "--prominent",
        action="store_true",
        help="Use prominent-feature batches for PANPE baseline",
    )
    parser.add_argument(
        "--pickle-file",
        default="results_exp_1L_fitconstraints0_width0.3_simple.pkl",
        help="Path to anaklasis pickle file, relative to base dir",
    )
    parser.add_argument(
        "--output",
        default="paper_batches/constraint_mape_histogram_anaklasis_vs_panpe_baseline.pdf",
        help="Output path for overlap histogram, relative to base dir",
    )
    parser.add_argument(
        "--output-random-style",
        default="paper_batches/constraint_mape_random_style_anaklasis_vs_panpe_baseline.pdf",
        help="Output path for random-style range comparison plot, relative to base dir",
    )
    parser.add_argument(
        "--output-median",
        default="paper_batches/constraint_mape_median_anaklasis_vs_panpe_baseline.pdf",
        help="Output path for median comparison plot, relative to base dir",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins",
    )

    args = parser.parse_args()
    run(
        base_dir=args.base_dir,
        prior_width=args.prior_width,
        fix_sld=args.fix_sld,
        prominent=args.prominent,
        pickle_file=args.pickle_file,
        output=args.output,
        output_random_style=args.output_random_style,
        output_median=args.output_median,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
