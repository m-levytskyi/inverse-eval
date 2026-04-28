#!/usr/bin/env python3
"""
Aggregate statistics across batch sweep results.

For each model category and each setup (prior width × sld-fix mode × prominent),
computes overall MAPE and per-parameter MAPE (mean + median) from the individual
experiment entries stored in batch_results.json files.

Outputs:
  - aggregate_stats.json   machine-readable summary
  - aggregate_stats.md     human-readable report
"""

import json
import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Batch catalogue  (mirrors replot_batch_results.py)
# ---------------------------------------------------------------------------

BATCH_CATALOGUE = {
    "NF_baseline": list(range(269, 287)),
    "NF_qweighted": list(range(233, 251)),
    "reflectorch": list(range(302, 320)),
    "NF_qweighted_exp1_alpha2_beta2_sweep": list(range(324, 342)),
    "NF_qweighted_exp2_alpha4_beta4_sweep": list(range(342, 360)),
    "NF_mean_conditioned_sweep": list(range(360, 378)),
}

# ---------------------------------------------------------------------------
# Setup-label parsing from directory name
# ---------------------------------------------------------------------------


def parse_setup(dir_name: str) -> dict:
    """
    Extract structured setup metadata from a batch directory name.

    Directory names follow the pattern:
      <id>_<N>exps_<L>layers_<W>constraint[_PROMINENT][_backSLDfix|_allSLDfix][_<date>]

    Returns a dict with keys:
      prior_width     : int  (5 | 30 | 99)
      fix_sld         : str  ("none" | "back" | "all")
      prominent       : bool
    """
    parts = dir_name.split("_")

    # prior width: the token that ends with "constraint"
    prior_width = None
    fix_sld = "none"
    prominent = False

    for p in parts:
        if p.endswith("constraint"):
            try:
                prior_width = int(p.replace("constraint", ""))
            except ValueError:
                pass
        elif p == "PROMINENT":
            prominent = True
        elif p == "backSLDfix":
            fix_sld = "back"
        elif p == "allSLDfix":
            fix_sld = "all"

    return {
        "prior_width": prior_width,
        "fix_sld": fix_sld,
        "prominent": prominent,
    }


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def load_batch(batch_dir: Path) -> dict:
    """Load batch_results.json from a batch directory."""
    path = batch_dir / "batch_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def collect_mape_arrays(batch_results: dict) -> dict:
    """
    Extract per-experiment MAPE values from a batch_results dict.

    Returns:
        {
            "overall":    [float, ...],
            "thickness":  [float, ...],
            "roughness":  [float, ...],
            "sld":        [float, ...],
            # individual params as they appear under by_parameter:
            "thickness":  [...],
            "amb_rough":  [...],
            "sub_rough":  [...],
            "layer_sld":  [...],
            "sub_sld":    [...],
            ...
        }
    """
    arrays: dict[str, list] = {}

    for exp_id, result in batch_results.items():
        if not result.get("success"):
            continue

        pm = result.get("param_metrics", {})
        if not pm:
            continue

        # overall constraint MAPE
        overall = pm.get("overall", {})
        if "constraint_mape" in overall:
            arrays.setdefault("overall", []).append(overall["constraint_mape"])

        # by_type constraint MAPEs  (thickness / roughness / sld)
        for ptype, metrics in pm.get("by_type", {}).items():
            if "constraint_mape" in metrics:
                arrays.setdefault(f"type:{ptype}", []).append(
                    metrics["constraint_mape"]
                )

        # individual parameter constraint MAPEs
        for pname, metrics in pm.get("by_parameter", {}).items():
            if "constraint_percentage_error" in metrics:
                arrays.setdefault(f"param:{pname}", []).append(
                    metrics["constraint_percentage_error"]
                )

    return arrays


def stat_dict(values: list) -> dict:
    """Return mean, median, std, min, max, n for a list of floats."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def aggregate_category(
    batch_ids: list[int],
    base_dir: Path,
) -> dict:
    """
    Aggregate statistics for all batches in a category.

    Returns a dict:
        {
            "by_setup": {
                "<setup_label>": {
                    "setup": {...},
                    "batch_id": int,
                    "dir": str,
                    "n_total": int,
                    "n_success": int,
                    "mape": { "overall": stat_dict, "type:thickness": ..., "param:thickness": ..., ... }
                },
                ...
            },
            "aggregate": {
                "n_total": int,
                "n_success": int,
                "mape": { ... }
            }
        }
    """
    by_setup = {}
    all_arrays: dict[str, list] = {}

    for batch_id in batch_ids:
        pattern = f"{batch_id:03d}_*"
        matches = list(base_dir.glob(pattern))
        if not matches:
            print(f"  [warn] batch {batch_id} not found in {base_dir}")
            continue

        batch_dir = matches[0]
        setup = parse_setup(batch_dir.name)
        setup_label = (
            f"width{setup['prior_width']}"
            f"_sld{setup['fix_sld']}"
            f"{'_prominent' if setup['prominent'] else ''}"
        )

        batch_results = load_batch(batch_dir)
        n_total = len(batch_results)
        n_success = sum(1 for v in batch_results.values() if v.get("success"))

        arrays = collect_mape_arrays(batch_results)
        mape_stats = {key: stat_dict(vals) for key, vals in arrays.items()}

        by_setup[setup_label] = {
            "setup": setup,
            "batch_id": batch_id,
            "dir": batch_dir.name,
            "n_total": n_total,
            "n_success": n_success,
            "mape": mape_stats,
        }

        # accumulate for category-level aggregate
        for key, vals in arrays.items():
            all_arrays.setdefault(key, []).extend(vals)

    agg_n_total = sum(v["n_total"] for v in by_setup.values())
    agg_n_success = sum(v["n_success"] for v in by_setup.values())
    aggregate = {
        "n_total": agg_n_total,
        "n_success": agg_n_success,
        "mape": {key: stat_dict(vals) for key, vals in all_arrays.items()},
    }

    return {"by_setup": by_setup, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def fmt(val, decimals=2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def mape_row(label: str, sd: dict) -> str:
    """Single markdown table row for a MAPE stat_dict."""
    return (
        f"| {label} "
        f"| {fmt(sd['mean'])} "
        f"| {fmt(sd['median'])} "
        f"| {fmt(sd['std'])} "
        f"| {fmt(sd['min'])} "
        f"| {fmt(sd['max'])} "
        f"| {sd['n']} |"
    )


MAPE_TABLE_HEADER = (
    "| Metric | Mean % | Median % | Std % | Min % | Max % | N |\n"
    "|--------|-------:|---------:|------:|------:|------:|--:|"
)


def setup_section(setup_label: str, data: dict) -> str:
    lines = []
    setup = data["setup"]
    lines.append(
        f"#### Setup: prior width={setup['prior_width']}%, "
        f"SLD fix={setup['fix_sld']}, "
        f"prominent={'yes' if setup['prominent'] else 'no'}"
    )
    lines.append(
        f"Batch {data['batch_id']} — `{data['dir']}`  "
        f"({data['n_success']}/{data['n_total']} successful experiments)"
    )
    lines.append("")
    lines.append(MAPE_TABLE_HEADER)

    mape = data["mape"]
    # overall first
    if "overall" in mape:
        lines.append(mape_row("**Overall**", mape["overall"]))
    # by_type
    for key in sorted(k for k in mape if k.startswith("type:")):
        label = key.replace("type:", "type: ")
        lines.append(mape_row(label, mape[key]))
    # individual params
    for key in sorted(k for k in mape if k.startswith("param:")):
        label = key.replace("param:", "param: ")
        lines.append(mape_row(label, mape[key]))

    lines.append("")
    return "\n".join(lines)


def aggregate_section(data: dict) -> str:
    lines = []
    lines.append(
        f"**Aggregate across all setups** "
        f"({data['n_success']}/{data['n_total']} successful experiments)"
    )
    lines.append("")
    lines.append(MAPE_TABLE_HEADER)

    mape = data["mape"]
    if "overall" in mape:
        lines.append(mape_row("**Overall**", mape["overall"]))
    for key in sorted(k for k in mape if k.startswith("type:")):
        lines.append(mape_row(key.replace("type:", "type: "), mape[key]))
    for key in sorted(k for k in mape if k.startswith("param:")):
        lines.append(mape_row(key.replace("param:", "param: "), mape[key]))

    lines.append("")
    return "\n".join(lines)


def build_markdown(results: dict) -> str:
    lines = []
    lines.append("# Batch Sweep Aggregate Statistics")
    lines.append("")
    lines.append(
        "Statistics computed over all successful experiments in each batch. "
        "MAPE = constraint-based Mean Absolute Percentage Error "
        "(error relative to the full parameter constraint range, %)."
    )
    lines.append("")

    for category, cat_data in results.items():
        lines.append("---")
        lines.append(f"## {category}")
        lines.append("")
        lines.append(aggregate_section(cat_data["aggregate"]))

        lines.append("### Per-setup breakdown")
        lines.append("")
        for setup_label, setup_data in sorted(cat_data["by_setup"].items()):
            lines.append(setup_section(setup_label, setup_data))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    base_dir: str = "batch_inference_results",
    output_dir: str = ".",
    categories: list[str] | None = None,
):
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    catalogue = BATCH_CATALOGUE
    if categories:
        catalogue = {k: v for k, v in catalogue.items() if k in categories}

    results = {}
    for category, batch_ids in catalogue.items():
        print(f"\nAggregating {category} ({len(batch_ids)} batches) …")
        results[category] = aggregate_category(batch_ids, base_path)
        agg = results[category]["aggregate"]
        overall = agg["mape"].get("overall", {})
        print(
            f"  → {agg['n_success']}/{agg['n_total']} successful, "
            f"overall MAPE mean={fmt(overall.get('mean'))}%, "
            f"median={fmt(overall.get('median'))}%"
        )

    json_path = out_path / "aggregate_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    md_path = out_path / "aggregate_stats.md"
    with open(md_path, "w") as f:
        f.write(build_markdown(results))
    print(f"Saved Markdown → {md_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate MAPE statistics across batch sweep results"
    )
    parser.add_argument(
        "--base-dir",
        default="batch_inference_results",
        help="Directory containing numbered batch folders (default: batch_inference_results)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write aggregate_stats.json and aggregate_stats.md (default: .)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help=(
            "Restrict to specific model categories. "
            f"Available: {', '.join(BATCH_CATALOGUE)}"
        ),
    )
    args = parser.parse_args()
    run(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
