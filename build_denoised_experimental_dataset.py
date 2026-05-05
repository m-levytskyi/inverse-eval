#!/usr/bin/env python3
"""Build denoised curves on experimental q-grids and save comparison plots."""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create denoised dataset by combining experimental q-values/ranges "
            "with corresponding denoised R values from a pickle file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="1_new",
        help="Directory containing s*_experimental_curve.dat files.",
    )
    parser.add_argument(
        "--pickle-file",
        default="reflectometry_1L_results.pkl",
        help="Pickle file containing q and R_denoised.",
    )
    parser.add_argument(
        "--output-dir",
        default="denoised_on_experimental_q",
        help="Output directory for generated s*_experimental_curve.dat files.",
    )
    parser.add_argument(
        "--plot-count",
        type=int,
        default=8,
        help="Number of per-curve comparison plots to generate.",
    )
    parser.add_argument(
        "--pattern",
        default="s*_experimental_curve.dat",
        help="Glob pattern for experimental files in --input-dir.",
    )
    return parser.parse_args()


def extract_experiment_id(path: Path) -> str:
    match = re.search(r"(s\d+)_experimental_curve\.dat$", path.name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse experiment ID from filename: {path.name}")


def experiment_index(exp_id: str) -> int:
    return int(exp_id[1:])


def load_pickle_data(pickle_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with pickle_file.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in pickle, got {type(data).__name__}")
    if "q" not in data or "R_denoised" not in data:
        raise KeyError("Pickle must contain keys: 'q' and 'R_denoised'")

    q_ref = np.asarray(data["q"], dtype=float)
    r_denoised = np.asarray(data["R_denoised"], dtype=float)

    if q_ref.ndim != 1:
        raise ValueError(f"Expected 1D q array, got shape {q_ref.shape}")
    if r_denoised.ndim != 2:
        raise ValueError(f"Expected 2D R_denoised array, got shape {r_denoised.shape}")
    if r_denoised.shape[1] != q_ref.shape[0]:
        raise ValueError(
            "Incompatible shapes: "
            f"R_denoised.shape={r_denoised.shape}, q.shape={q_ref.shape}"
        )

    return q_ref, r_denoised


def write_curve_file(
    out_path: Path,
    q_exp: np.ndarray,
    r_new: np.ndarray,
    original_data: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original column count when possible to keep compatibility.
    if original_data.shape[1] >= 3:
        out_data = np.column_stack([q_exp, r_new, original_data[:, 2]])
        header = "Q(A^-1)        R           dR"
        np.savetxt(out_path, out_data, fmt="%.16e", header=header, comments="# ")
    else:
        out_data = np.column_stack([q_exp, r_new])
        header = "Q(A^-1)        R"
        np.savetxt(out_path, out_data, fmt="%.16e", header=header, comments="# ")


def plot_curve_comparison(
    out_file: Path,
    exp_id: str,
    q_exp: np.ndarray,
    r_exp: np.ndarray,
    r_denoised_interp: np.ndarray,
) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(q_exp, r_exp, "o", markersize=2.5, alpha=0.65, label="Experimental")
    plt.plot(q_exp, r_denoised_interp, "-", linewidth=1.6, label="Denoised (on exp q)")
    plt.yscale("log")
    plt.xlabel("Q (A^-1)")
    plt.ylabel("R")
    plt.title(f"{exp_id}: Experimental vs Denoised")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    pickle_file = Path(args.pickle_file)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not pickle_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

    q_ref, r_denoised = load_pickle_data(pickle_file)

    exp_files = sorted(input_dir.glob(args.pattern), key=lambda p: experiment_index(extract_experiment_id(p)))
    if not exp_files:
        raise FileNotFoundError(f"No files found with pattern {args.pattern} in {input_dir}")

    out_of_range_points = 0
    copied_model_files = 0
    selected_for_plots: list[Path] = []

    if args.plot_count > 0:
        positions = np.linspace(0, len(exp_files) - 1, num=min(args.plot_count, len(exp_files)), dtype=int)
        selected_for_plots = [exp_files[i] for i in np.unique(positions)]

    for row_idx, exp_file in enumerate(exp_files):
        exp_id = extract_experiment_id(exp_file)

        exp_data = np.loadtxt(exp_file, comments="#")
        if exp_data.ndim != 2 or exp_data.shape[1] < 2:
            raise ValueError(f"Invalid experimental format in {exp_file}: shape={exp_data.shape}")

        q_exp = exp_data[:, 0]
        r_exp = exp_data[:, 1]

        # Interpolate denoised curve to exactly match each experiment's q grid and range.
        curve_denoised = r_denoised[row_idx]
        r_interp = np.interp(
            q_exp,
            q_ref,
            curve_denoised,
            left=curve_denoised[0],
            right=curve_denoised[-1],
        )

        out_of_range_points += int(np.sum((q_exp < q_ref[0]) | (q_exp > q_ref[-1])))

        write_curve_file(
            out_path=output_dir / f"{exp_id}_experimental_curve.dat",
            q_exp=q_exp,
            r_new=r_interp,
            original_data=exp_data,
        )

        # Keep matching model files alongside generated curves for direct reuse.
        model_src = input_dir / f"{exp_id}_model.txt"
        if model_src.exists():
            shutil.copy2(model_src, output_dir / model_src.name)
            copied_model_files += 1

        if exp_file in selected_for_plots:
            plot_curve_comparison(
                out_file=plots_dir / f"{exp_id}_denoised_vs_experimental.png",
                exp_id=exp_id,
                q_exp=q_exp,
                r_exp=r_exp,
                r_denoised_interp=r_interp,
            )

    print("=" * 80)
    print("DENOISED DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Input experimental files:   {len(exp_files)}")
    print(f"Output curve files:         {len(exp_files)}")
    print(f"Output directory:           {output_dir}")
    print(f"Model files copied:         {copied_model_files}")
    print(f"Plots generated:            {len(selected_for_plots)}")
    print(f"Plots directory:            {plots_dir}")
    print(f"Out-of-range q points:      {out_of_range_points}")
    print(f"Reference q range:          [{q_ref.min():.6f}, {q_ref.max():.6f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
