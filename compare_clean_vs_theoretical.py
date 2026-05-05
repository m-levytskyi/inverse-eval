#!/usr/bin/env python3
"""Compare clean curves from a pickle file with theoretical curve files."""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CurveComparison:
    experiment_id: str
    q_match: bool
    r_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare R_clean from a pickle file with s*_theoretical_curve.dat files "
            "in a data directory."
        )
    )
    parser.add_argument(
        "--pickle-file",
        default="reflectometry_1L_results.pkl",
        help="Path to pickle file containing R_clean and optional q.",
    )
    parser.add_argument(
        "--curves-dir",
        default="1_new",
        help="Directory containing s*_theoretical_curve.dat files.",
    )
    parser.add_argument(
        "--pattern",
        default="s*_theoretical_curve.dat",
        help="Glob pattern for curve files inside --curves-dir.",
    )
    parser.add_argument(
        "--r-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for R curve comparisons.",
    )
    parser.add_argument(
        "--r-rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for R curve comparisons.",
    )
    parser.add_argument(
        "--q-atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for q-grid comparisons.",
    )
    parser.add_argument(
        "--q-rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for q-grid comparisons.",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="How many mismatches to print in detail.",
    )
    return parser.parse_args()


def _extract_experiment_id(path: Path) -> str:
    match = re.search(r"(s\d+)_", path.name)
    return match.group(1) if match else path.stem


def _extract_experiment_index(exp_id: str) -> int:
    match = re.search(r"s(\d+)", exp_id)
    if not match:
        raise ValueError(f"Cannot parse experiment index from id: {exp_id}")
    return int(match.group(1))


def load_pickle_curves(pickle_file: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with pickle_file.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "Expected pickle to contain a dict with key 'R_clean'. "
            f"Got type: {type(data).__name__}"
        )

    if "R_clean" not in data:
        raise KeyError("Pickle does not contain required key 'R_clean'.")

    r_clean = np.asarray(data["R_clean"], dtype=float)
    if r_clean.ndim != 2:
        raise ValueError(
            f"Expected R_clean to be 2D (n_experiments, n_q), got shape {r_clean.shape}."
        )

    q_values = None
    if "q" in data:
        q_values = np.asarray(data["q"], dtype=float)
        if q_values.ndim != 1:
            raise ValueError(f"Expected q to be 1D, got shape {q_values.shape}.")

    return r_clean, q_values


def load_theoretical_curve(curve_file: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(curve_file, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns (q, R) in {curve_file}, got shape {data.shape}."
        )
    q_vals = data[:, 0]
    r_vals = data[:, 1]
    return q_vals, r_vals


def compare_dataset(
    r_clean: np.ndarray,
    q_from_pickle: np.ndarray | None,
    curve_files: list[Path],
    r_atol: float,
    r_rtol: float,
    q_atol: float,
    q_rtol: float,
) -> list[CurveComparison]:
    comparisons: list[CurveComparison] = []

    if len(curve_files) > r_clean.shape[0]:
        raise ValueError(
            f"More curve files ({len(curve_files)}) than rows in R_clean "
            f"({r_clean.shape[0]}). Ensure the pickle and curve directory match."
        )

    for row_idx, curve_file in enumerate(curve_files):
        exp_id = _extract_experiment_id(curve_file)

        q_file, r_file = load_theoretical_curve(curve_file)
        r_ref = r_clean[row_idx]

        if r_file.shape[0] != r_ref.shape[0]:
            raise ValueError(
                f"Point count mismatch for {exp_id}: "
                f"file has {r_file.shape[0]}, R_clean has {r_ref.shape[0]}."
            )

        if q_from_pickle is not None and q_file.shape[0] != q_from_pickle.shape[0]:
            raise ValueError(
                f"Q-grid point count mismatch for {exp_id}: "
                f"file has {q_file.shape[0]}, pickle q has {q_from_pickle.shape[0]}."
            )

        q_match = True
        if q_from_pickle is not None:
            q_match = bool(
                np.allclose(q_file, q_from_pickle, atol=q_atol, rtol=q_rtol)
            )

        diff = np.abs(r_file - r_ref)
        r_match = bool(
            np.allclose(r_file, r_ref, atol=r_atol, rtol=r_rtol, equal_nan=True)
        )

        comparisons.append(
            CurveComparison(
                experiment_id=exp_id,
                q_match=q_match,
                r_match=r_match,
                max_abs_diff=float(np.nanmax(diff)),
                mean_abs_diff=float(np.nanmean(diff)),
                points=int(r_file.shape[0]),
            )
        )

    return comparisons


def print_report(
    comparisons: list[CurveComparison],
    total_clean_rows: int,
    total_curve_files: int,
    show_mismatches: int,
) -> None:
    q_mismatch = [c for c in comparisons if not c.q_match]
    r_mismatch = [c for c in comparisons if not c.r_match]
    any_mismatch = [c for c in comparisons if (not c.q_match) or (not c.r_match)]

    global_max_abs = max((c.max_abs_diff for c in comparisons), default=float("nan"))
    global_mean_abs = float(np.mean([c.mean_abs_diff for c in comparisons])) if comparisons else float("nan")

    print("=" * 80)
    print("R_CLEAN VS THEORETICAL CURVES COMPARISON")
    print("=" * 80)
    print(f"R_clean rows in pickle:      {total_clean_rows}")
    print(f"Theoretical files compared:  {total_curve_files}")
    print(f"Curve-level q mismatches:    {len(q_mismatch)}")
    print(f"Curve-level R mismatches:    {len(r_mismatch)}")
    print(f"Global max |delta R|:        {global_max_abs:.6e}")
    print(f"Global mean |delta R|:       {global_mean_abs:.6e}")

    if any_mismatch:
        any_mismatch = sorted(any_mismatch, key=lambda c: c.max_abs_diff, reverse=True)
        print("\nResult: DATASETS ARE NOT IDENTICAL within the selected tolerances.")
        print(f"Showing up to {show_mismatches} mismatches:")
        for comp in any_mismatch[:show_mismatches]:
            print(
                f"  {comp.experiment_id}: q_match={comp.q_match}, r_match={comp.r_match}, "
                f"max_abs_diff={comp.max_abs_diff:.6e}, mean_abs_diff={comp.mean_abs_diff:.6e}"
            )
    else:
        print("\nResult: DATASETS MATCH within the selected tolerances.")


def main() -> int:
    args = parse_args()

    pickle_file = Path(args.pickle_file)
    curves_dir = Path(args.curves_dir)

    if not pickle_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    if not curves_dir.exists():
        raise FileNotFoundError(f"Curves directory not found: {curves_dir}")

    curve_files = sorted(curves_dir.glob(args.pattern), key=lambda p: _extract_experiment_index(_extract_experiment_id(p)))
    if not curve_files:
        raise FileNotFoundError(
            f"No curve files found in {curves_dir} with pattern {args.pattern}"
        )

    r_clean, q_from_pickle = load_pickle_curves(pickle_file)

    comparisons = compare_dataset(
        r_clean=r_clean,
        q_from_pickle=q_from_pickle,
        curve_files=curve_files,
        r_atol=args.r_atol,
        r_rtol=args.r_rtol,
        q_atol=args.q_atol,
        q_rtol=args.q_rtol,
    )

    print_report(
        comparisons=comparisons,
        total_clean_rows=r_clean.shape[0],
        total_curve_files=len(curve_files),
        show_mismatches=args.show_mismatches,
    )

    if len(curve_files) != r_clean.shape[0]:
        print(
            "\nWarning: Number of compared files differs from number of R_clean rows. "
            "Only matched file set was compared."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
