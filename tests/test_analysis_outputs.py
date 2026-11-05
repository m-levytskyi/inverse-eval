import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import aggregate_batch_stats as aggregate
import compare_model_versions as comparison
import evaluate_random_guessing as random_baseline
import plotting_utils as plots


NAMES = ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]


def _record(score=2.0):
    truth = [100.0, 3.0, 4.0, 5.0, 6.0]
    return {
        "success": True,
        "layer_count": 1,
        "param_metrics": {
            "overall": {"constraint_mape": score},
            "by_parameter": {
                name: {"constraint_percentage_error": score} for name in NAMES
            },
        },
        "true_params_dict": {"1_layer": {"params": truth, "param_names": NAMES}},
        "prediction_dict": {"param_names": NAMES + ["r_scale", "log10_background"]},
        "prior_bounds": [[value, value] for value in truth],
    }


def _batch(root, name, records):
    path = root / name
    path.mkdir()
    (path / "batch_results.json").write_text(json.dumps(records))
    return path


def test_current_batch_records_exclude_failures_and_invalid_metrics(tmp_path):
    batch = _batch(
        tmp_path,
        "001_Noneexps_1layers_30constraint",
        {
            "valid": _record(120.0),
            "failed": {**_record(), "success": False},
            "invalid": _record(float("nan")),
        },
    )
    overall, parameters = comparison.extract_mapes_from_batch(batch)
    assert overall == [120.0]
    assert parameters == {name: [120.0] for name in NAMES}


def test_setup_metadata_and_legacy_names(tmp_path):
    assert (
        comparison.parse_batch_config("001_Noneexps_2layers_30constraint")["num_exps"]
        is None
    )
    assert (
        comparison.parse_batch_config("001_custom_2exps_1layers_5priors")["priors_type"]
        == "narrow"
    )
    batch = _batch(tmp_path, "arbitrary-name", {})
    config = {
        "layer_count": 2,
        "priors_deviation": 0.3,
        "priors_type": "constraint_based",
        "fix_sld_mode": "backing",
        "use_prominent_features": True,
        "use_theoretical": True,
    }
    (batch / "run_metadata.json").write_text(json.dumps({"config": config}))
    loaded = comparison.load_batch_config(batch)
    assert (
        loaded["layer_count"],
        loaded["sld_fix_mode"],
        loaded["use_theoretical"],
    ) == (2, "backing", True)


def test_comparison_pairs_experiments_and_preserves_each_output(tmp_path, monkeypatch):
    seen = []
    original = plots.plot_model_comparison_histogram

    def capture(**kwargs):
        seen.append((kwargs["baseline_mapes"], kwargs["comparison_mapes"]))
        return original(**kwargs)

    monkeypatch.setattr(comparison, "plot_model_comparison_histogram", capture)
    for number, prior in [(1, 5), (2, 30)]:
        _batch(
            tmp_path,
            f"{number:03d}_Noneexps_1layers_{prior}constraint",
            {
                "paired": _record(0.0),
                "baseline_only": _record(90.0),
            },
        )
        _batch(
            tmp_path,
            f"{number + 2:03d}_custom_2exps_1layers_{prior}constraint",
            {
                "paired": _record(120.0),
                "comparison_only": _record(80.0),
            },
        )
    paths = comparison.generate_comparison_plots(
        (1, 2), (3, 4), tmp_path, tmp_path / "plots"
    )
    assert seen == [([0.0], [120.0]), ([0.0], [120.0])]
    assert len(paths) == len(set(paths)) == 4
    assert len({path.parent for path in paths}) == 2
    assert all(path.exists() for path in paths)
    for parent in {path.parent for path in paths}:
        manifest = json.loads((parent / "comparison_manifest.json").read_text())
        assert manifest["experiment_ids"] == ["paired"]
        assert manifest["unmatched_baseline"] == ["baseline_only"]


def test_duplicate_setups_and_batch_directories_are_rejected(tmp_path, monkeypatch):
    first = _batch(tmp_path, "001_Noneexps_1layers_30constraint", {"s1": _record()})
    second = _batch(tmp_path, "002_Noneexps_1layers_30constraint", {"s2": _record()})
    with pytest.raises(ValueError, match="duplicate setup"):
        comparison.find_matching_batches([first, second], [])
    with pytest.raises(ValueError, match="Duplicate setup"):
        aggregate.aggregate_category([1, 2], tmp_path)
    _batch(tmp_path, "001_custom_1exps_1layers_5constraint", {"s3": _record()})
    with pytest.raises(ValueError, match="Ambiguous batch"):
        aggregate.aggregate_category([1], tmp_path)
    with pytest.raises(ValueError, match="Ambiguous batch"):
        comparison.generate_comparison_plots(
            (1, 1), (2, 2), tmp_path, tmp_path / "plots"
        )

    batches = tmp_path / "batch_inference_results"
    batches.mkdir()
    _batch(batches, "001_first", {})
    _batch(batches, "1_second", {})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Ambiguous batch"):
        random_baseline.process_batches([1])


def test_layer_counts_are_part_of_comparison_matching(tmp_path):
    first = _batch(tmp_path, "001_Noneexps_1layers_30constraint", {})
    second = _batch(tmp_path, "002_Noneexps_2layers_30constraint", {})
    assert comparison.find_matching_batches([first], [second]) == []


def test_parameter_scores_use_the_same_experiment_ids(tmp_path, monkeypatch):
    left = {"a": _record(1.0), "b": _record(2.0)}
    right = {"a": _record(3.0), "b": _record(4.0)}
    del right["a"]["param_metrics"]["by_parameter"]["thickness"]
    _batch(tmp_path, "001_Noneexps_1layers_30constraint", left)
    _batch(tmp_path, "002_Noneexps_1layers_30constraint", right)
    captured = []
    monkeypatch.setattr(
        comparison, "plot_model_comparison_histogram", lambda **kwargs: None
    )
    monkeypatch.setattr(
        comparison,
        "plot_parameter_comparison_grid",
        lambda **kwargs: captured.append(kwargs),
    )
    comparison.generate_comparison_plots((1, 1), (2, 2), tmp_path, tmp_path / "plots")
    assert captured[0]["baseline_per_param"]["thickness"] == [2.0]
    assert captured[0]["comparison_per_param"]["thickness"] == [4.0]


def test_random_baseline_uses_physical_truth_and_keeps_scores_paired(tmp_path):
    missing_truth = _record(90.0)
    del missing_truth["true_params_dict"]
    malformed = _record(80.0)
    malformed["prior_bounds"] = [[1, 2]]
    batch = _batch(
        tmp_path,
        "batch",
        {
            "valid_nf": _record(),
            "missing": missing_truth,
            "malformed": malformed,
        },
    )
    model, random = random_baseline.evaluate_batch_against_random(batch)
    assert model == [2.0]
    assert random == [0.0]


def test_random_comparisons_keep_separate_batch_artifacts(tmp_path, monkeypatch):
    batches = tmp_path / "batch_inference_results"
    batches.mkdir()
    names = ["001_Noneexps_1layers_5constraint", "002_Noneexps_1layers_30constraint"]
    for name in names:
        _batch(batches, name, {"paired": _record()})
    monkeypatch.chdir(tmp_path)
    random_baseline.process_batches([1, 2])
    for name in names:
        assert (
            tmp_path
            / "random_guessing_evaluation"
            / name
            / "random_comparison_1layer.pdf"
        ).exists()


def test_histogram_bins_include_exactly_100_and_overflow(tmp_path, monkeypatch):
    ranges, labels = plots._get_mape_ranges()
    counts = plots._count_mapes_in_ranges([0, 99.9, 100, 120], ranges)
    assert sum(counts) == 4
    assert counts[-1] == 2
    assert labels[-1] == "≥100"
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    plots.plot_batch_mape_distribution(
        {str(i): _record(value) for i, value in enumerate([0, 100, 120])}, save=False
    )
    assert sum(bar.get_height() for bar in shown[0].axes[0].patches) == 3
    plt.close("all")


def test_parameter_grid_shows_all_eight_parameters(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    parameters = {f"param{i}": [0.0] for i in range(8)}
    plots.plot_parameter_comparison_grid(
        parameters, parameters, {"deviation": 0.3}, save=False
    )
    visible = [ax for ax in shown[0].axes if ax.axison]
    assert len(visible) == 8
    assert {ax.get_title() for ax in visible} == set(parameters)
    assert all(np.isfinite(bar.get_height()) for ax in visible for bar in ax.patches)
    plt.close("all")


def test_synthetic_baseline_labels_truth_conditioning_and_saves_all_parameters(
    tmp_path, monkeypatch
):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    output = tmp_path / "baseline.png"
    result = random_baseline.generate_synthetic_random_evaluation(
        2, layer_count=2, output_path=output
    )
    assert len(result["by_parameter"]) == 8
    assert output.exists() and output.with_name("baseline_per_parameter.png").exists()
    assert len([ax for ax in shown[1].axes if ax.get_visible()]) == 8
    texts = [text.get_text() for fig in shown for ax in fig.axes for text in ax.texts]
    assert not any("Theoretical" in text or "33.33" in text for text in texts)
    assert "truth-conditioned" in shown[0].axes[0].get_title()
