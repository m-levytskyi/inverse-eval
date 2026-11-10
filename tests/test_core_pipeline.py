"""Cheap integration checks for discovery, priors and the inference boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

import simple_pipeline
from error_calculation import calculate_parameter_metrics
from multimodal_posterior_mape import (
    select_current_ml_sample,
    select_multimodal_ml_sample,
)
from nf_statistics import compute_nf_sample_statistics, validate_nf_samples
from parameter_discovery import (
    discover_batch_experiments,
    discover_experiment_files,
    generate_true_sld_profile,
    get_prior_bounds_for_experiment,
    parse_true_parameters_from_model_file,
)
from simple_pipeline import (
    _nf_prediction_to_single_prediction_dict,
    align_samples_to_true,
    get_inference_prior_limits,
    resolve_inference_config,
    run_nf_inference,
    run_single_experiment,
)


PHYSICAL_NAMES = ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]
LABELS = [
    "Thickness L1",
    "Roughness L1",
    "Roughness Substrate",
    "SLD L1",
    "SLD Substrate",
]
TRUTH = np.array([100.0, 2.0, 3.0, 4.0, 8.0])


def write_experiment(directory, exp_id="s000001", layer_count=1, ambient=2.0):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{exp_id}_model.txt"
    rows = [f"fronting {ambient * 1e-6} inf 2", "layer1 4e-6 100 3"]
    if layer_count == 2:
        rows.append("layer2 6e-6 50 4")
    model.write_text("\n".join(rows + ["backing 8e-6 inf none"]) + "\n")
    curve = directory / f"{exp_id}_experimental_curve.dat"
    np.savetxt(
        curve,
        np.column_stack(
            (np.linspace(0.01, 0.2, 8), np.ones(8), np.full(8, 0.1), np.zeros(8))
        ),
        header="Q R dR dQ",
    )
    return curve, model


class FakeModel:
    def __init__(self, nf=True, sld_width=24.0):
        self.device = "cpu"
        self.config_name = "fake.yaml"
        self.calls = []
        names = LABELS + (["r_scale", "log10_background"] if nf else [])
        lower = [1.0, 0.0, 0.0, -8.0, -8.0] + ([0.9, -10.0] if nf else [])
        upper = [1500.0, 60.0, 60.0, 16.0, 16.0] + ([1.1, -4.0] if nf else [])
        widths = [1500.0, 60.0, 60.0, sld_width, sld_width] + ([0.2, 6.0] if nf else [])
        sampler = SimpleNamespace(
            max_num_layers=1,
            param_dim=len(names),
            min_bounds=np.array(lower),
            max_bounds=np.array(upper),
            min_delta=np.full(len(names), 0.01),
            max_delta=np.array(widths),
            param_model=SimpleNamespace(get_param_labels=lambda **kwargs: names),
        )
        self.trainer = SimpleNamespace(loader=SimpleNamespace(prior_sampler=sampler))

    def interpolate_data_to_model_q(self, q, curve, sigmas_exp=None):
        self.calls.append(("interpolate", {}))
        return (q, curve) if sigmas_exp is None else (q, curve, sigmas_exp)

    def preprocess_and_sample(self, **kwargs):
        self.calls.append(("sample", kwargs))
        count = kwargs["num_samples"]
        samples = np.tile(np.r_[TRUTH, 1.0, -7.0], (count, 1))
        return {
            "predicted_params_array": samples.squeeze() if count == 1 else samples,
            "log_likelihoods": np.arange(count, dtype=float),
            "sampled_curves": np.ones((count, 3)),
            "q_plot_pred": np.array([0.01, 0.1, 0.2]),
            "sampled_sld_profiles": np.ones((count, 4)),
            "sampled_sld_xaxis": np.arange(4.0),
            "param_names": LABELS + ["r_scale", "log10_background"],
            "ess": 1.5,
        }

    def predict(self, **kwargs):
        self.calls.append(("predict", kwargs))
        # Equal-length reordered output must still be aligned by name.
        return {
            "predicted_params_array": TRUTH[::-1].copy(),
            "polished_params_array": TRUTH[::-1].copy(),
            "param_names": LABELS[::-1],
        }


def test_nonzero_ambient_reaches_both_inference_backends_and_metrics_stay_absolute(
    tmp_path, monkeypatch
):
    write_experiment(tmp_path)
    monkeypatch.setattr(
        simple_pipeline,
        "EasyInferenceModel",
        lambda **kwargs: pytest.fail("injected model must be reused"),
    )
    for backend in ("predict", "nf"):
        model = FakeModel(nf=backend == "nf")
        for _ in range(2):
            result = run_single_experiment(
                "s000001",
                data_directory=str(tmp_path),
                inference_backend=backend,
                inference_model=model,
                nf_num_samples=2,
            )
            assert result["ambient_sld"] == pytest.approx(2.0)
            assert result["param_metrics"]["overall"][
                "constraint_mape"
            ] == pytest.approx(0.0)
            assert model.calls[-1][1]["ambient_sld"] == pytest.approx(2.0)
        if backend == "nf":
            assert [name for name, _ in model.calls] == ["sample", "sample"]


def test_true_sld_uses_parsed_ambient_and_explicit_override(tmp_path):
    _, model_path = write_experiment(tmp_path, ambient=3.5)
    truth = parse_true_parameters_from_model_file(model_path)
    z = np.array([-100.0, 50.0, 300.0])
    assert truth["1_layer"]["ambient_sld"] == pytest.approx(3.5)
    assert generate_true_sld_profile(truth, x_axis=z)[1][0] == pytest.approx(3.5)
    assert generate_true_sld_profile(truth, x_axis=z, ambient_sld=0.0)[1][
        0
    ] == pytest.approx(0.0)


@pytest.mark.parametrize("fraction, expected", [(0.05, 1.2), (0.3, 7.2), (0.99, 23.76)])
def test_prior_percentages_use_fixed_spans_and_selected_width_limits(
    fraction, expected
):
    truth = {"1_layer": {"params": TRUTH, "param_names": PHYSICAL_NAMES}}
    for cap in (5.0, 24.0):
        ranges, widths = get_inference_prior_limits(FakeModel(sld_width=cap), 1)
        bounds = get_prior_bounds_for_experiment(
            "x", truth, deviation=fraction, model_ranges=ranges, width_limits=widths
        )
        assert bounds[3][1] - bounds[3][0] == pytest.approx(min(expected, cap))
        assert bounds[0][1] - bounds[0][0] == pytest.approx(fraction * 999.0)
    assert calculate_parameter_metrics([101.0], [100.0], ["thickness"])["overall"][
        "constraint_mape"
    ] == pytest.approx(100.0 / 999.0)


def test_shifted_bounds_and_minimum_widths_preserve_truth():
    values = [100.0, 0.0, 0.0, -4.5, 16.0]
    truth = {"1_layer": {"params": values}}
    ranges, widths = get_inference_prior_limits(FakeModel(), 1, ambient_sld=3.5)
    bounds = get_prior_bounds_for_experiment(
        "x",
        truth,
        priors_type="narrow",
        deviation=0.0,
        fix_sld_mode="all",
        model_ranges=ranges,
        width_limits=widths,
    )
    for value, (lo, hi) in zip(values, bounds):
        assert lo <= value <= hi
        assert hi - lo == pytest.approx(0.01)
    assert bounds[3][0] >= -4.5
    assert bounds[4][1] <= 16.0
    truth["1_layer"]["params"][3] = -5.0
    with pytest.raises(ValueError, match="outside supported inference range"):
        get_prior_bounds_for_experiment(
            "x", truth, model_ranges=ranges, width_limits=widths
        )


def test_discovery_uses_complete_matching_layer_pairs(tmp_path):
    two = tmp_path / "2"
    write_experiment(two, layer_count=2)
    assert discover_experiment_files("s000001", tmp_path, 1) == (None, None, None)
    assert discover_experiment_files("s000001", tmp_path, 2)[2] == 2
    flat = tmp_path / "flat"
    write_experiment(flat, exp_id="s000003")
    write_experiment(flat, exp_id="s000002")
    assert discover_batch_experiments(tmp_path, 1, num_experiments=1) == ["s000002"]
    assert discover_experiment_files("s000002", flat, 1)[2] == 1
    (flat / "s000002_model.txt").rename(two / "s000002_model.txt")
    assert discover_experiment_files("s000002", tmp_path, 1) == (None, None, None)


def test_discovery_rejects_ambiguity_and_path_model_disagreement(tmp_path):
    write_experiment(tmp_path / "a")
    write_experiment(tmp_path / "b")
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_experiment_files("s000001", tmp_path, 1)
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_batch_experiments(tmp_path, 1)
    write_experiment(tmp_path / "1", exp_id="s000004", layer_count=2)
    with pytest.raises(ValueError, match="disagree"):
        discover_experiment_files("s000004", tmp_path, 1)


def test_explicit_discovery_keeps_requested_order_and_missing_ids(tmp_path):
    assert discover_batch_experiments(
        tmp_path, 1, experiment_ids=["s000003", "s000001", "s000003"]
    ) == ["s000003", "s000001"]


def test_nf_uses_raw_input_once_and_records_legacy_noise_and_proposal_stats():
    model = FakeModel()
    q = np.linspace(0.01, 0.2, 8)
    q_out, prediction = run_nf_inference(
        model, q, np.ones(8), [(1.0, 2.0)] * 5, nf_num_samples=1
    )
    assert [name for name, _ in model.calls] == ["sample"]
    kwargs = model.calls[0][1]
    assert kwargs["q_values"] is q
    assert kwargs["enable_error_bars_filtering"] is False
    assert kwargs["rel_err_factor"] == 0.2
    assert kwargs["sigmas"] is None
    assert len(kwargs["prior_bounds"]) == 7
    assert len(q_out) == 3
    assert prediction["nf_params_std"] is None
    assert prediction["nf_log_prob_std"] is None
    assert prediction["nf_ess"] == 1.5
    assert prediction["nf_statistics_kind"] == "unweighted_proposal"
    assert prediction["nf_likelihood_noise"] == {
        "source": "relative",
        "relative_error": 0.2,
        "absolute_floor": 1e-12,
    }


def test_provided_sigmas_are_forwarded_without_vendor_error_filtering():
    model = FakeModel()
    sigmas = np.full(8, 0.4)
    _, prediction = run_nf_inference(
        model,
        np.arange(8.0),
        np.ones(8),
        [(1.0, 2.0)] * 5,
        nf_num_samples=2,
        sigmas_exp=sigmas,
    )
    assert model.calls[0][1]["sigmas"] is sigmas
    assert prediction["nf_likelihood_noise"] == {"source": "provided_sigmas"}


def test_production_and_diagnostics_share_full_sample_validity_and_original_indices():
    posterior = FakeModel().preprocess_and_sample(num_samples=3)
    posterior["log_likelihoods"] = np.array([10.0, np.nan, 1.0])
    posterior["predicted_params_array"][0, -1] = np.nan
    prediction = _nf_prediction_to_single_prediction_dict(posterior)
    raw, likelihoods, mask = validate_nf_samples(
        posterior["predicted_params_array"], posterior["log_likelihoods"]
    )
    aligned = align_samples_to_true(raw, posterior["param_names"], PHYSICAL_NAMES, 1)
    current = select_current_ml_sample(
        aligned, likelihoods, TRUTH, PHYSICAL_NAMES, valid_mask=mask
    )
    oracle = select_multimodal_ml_sample(
        aligned, likelihoods, TRUTH, PHYSICAL_NAMES, valid_mask=mask
    )
    assert (
        prediction["nf_best_idx"]
        == current["selected_index"]
        == oracle["selected_index"]
        == 2
    )
    assert oracle["labels"].tolist() == [-1, -1, 0]
    assert oracle["selected_silhouette"] == 0.0


@pytest.mark.parametrize("per_sample_depths", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_nf_adapter_keeps_selected_sample_and_depth_axis_aligned(
    per_sample_depths, count
):
    posterior = FakeModel().preprocess_and_sample(num_samples=count)
    if count == 1:
        # The vendor may squeeze the sample dimension for singleton outputs.
        posterior["sampled_curves"] = posterior["sampled_curves"][0]
        posterior["sampled_sld_profiles"] = posterior["sampled_sld_profiles"][0]
    if per_sample_depths:
        posterior["sampled_sld_xaxis"] = np.arange(count * 4.0).reshape(count, 4)
    prediction = _nf_prediction_to_single_prediction_dict(posterior)
    best = count - 1
    expected_depths = posterior["sampled_sld_xaxis"]
    if per_sample_depths:
        expected_depths = expected_depths[best]
    assert prediction["nf_best_idx"] == best
    assert prediction["nf_best_log_likelihood"] == float(best)
    np.testing.assert_array_equal(prediction["predicted_sld_xaxis"], expected_depths)
    assert prediction["predicted_sld_xaxis"].ndim == 1
    for key in ("predicted_params_array", "polished_params_array"):
        np.testing.assert_array_equal(prediction[key], np.r_[TRUTH, 1.0, -7.0])
    for key, length in (
        ("predicted_curve", 3),
        ("polished_curve", 3),
        ("predicted_sld_profile", 4),
        ("sld_profile_polished", 4),
    ):
        np.testing.assert_array_equal(prediction[key], np.ones(length))


@pytest.mark.parametrize(
    "field",
    [
        "predicted_params_array",
        "log_likelihoods",
        "sampled_curves",
        "sampled_sld_profiles",
        "sampled_sld_xaxis",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("per_sample_depths", [False, True])
def test_nf_adapter_selects_best_complete_row_at_original_index(
    field, invalid, per_sample_depths
):
    posterior = FakeModel().preprocess_and_sample(num_samples=3)
    if per_sample_depths:
        posterior["sampled_sld_xaxis"] = np.arange(12.0).reshape(3, 4)
    posterior["log_likelihoods"] = np.array([50.0, 10.0, 100.0])
    # Both higher-ranked rows are invalid; the surviving row keeps index 1.
    posterior["sampled_curves"][0, 0] = np.nan
    if field == "sampled_sld_xaxis" and not per_sample_depths:
        # A non-finite shared axis invalidates every sample.
        posterior[field][2] = invalid
        with pytest.raises(ValueError, match="no finite complete sample rows"):
            _nf_prediction_to_single_prediction_dict(posterior)
        return
    posterior[field][2] = invalid
    prediction = _nf_prediction_to_single_prediction_dict(posterior)
    assert prediction["nf_best_idx"] == 1
    assert prediction["nf_best_log_likelihood"] == 10.0
    np.testing.assert_array_equal(
        prediction["predicted_sld_xaxis"],
        [4, 5, 6, 7] if per_sample_depths else [0, 1, 2, 3],
    )


@pytest.mark.parametrize(
    "field",
    [
        "sampled_curves",
        "sampled_sld_profiles",
        "sampled_sld_xaxis",
        "q_plot_pred",
    ],
)
def test_nf_adapter_rejects_posterior_without_complete_finite_rows(field):
    posterior = FakeModel().preprocess_and_sample(num_samples=2)
    posterior[field][...] = np.nan
    with pytest.raises(ValueError, match="no finite complete sample rows"):
        _nf_prediction_to_single_prediction_dict(posterior)


@pytest.mark.parametrize(
    "field, shape",
    [
        ("sampled_sld_xaxis", (1, 4)),
        ("sampled_sld_xaxis", (2, 3)),
        ("sampled_sld_xaxis", (2, 1, 4)),
        ("q_plot_pred", (2, 3)),
        ("sampled_sld_profiles", (2, 0)),
    ],
)
def test_nf_adapter_rejects_malformed_axes_and_empty_profiles(field, shape):
    posterior = FakeModel().preprocess_and_sample(num_samples=2)
    posterior[field] = np.ones(shape)
    with pytest.raises(ValueError):
        _nf_prediction_to_single_prediction_dict(posterior)


def test_nf_pipeline_accepts_per_sample_depths_and_skips_invalid_top_curve(tmp_path):
    write_experiment(tmp_path)

    class PerSampleModel(FakeModel):
        def preprocess_and_sample(self, **kwargs):
            posterior = super().preprocess_and_sample(**kwargs)
            posterior["sampled_sld_xaxis"] = np.arange(8.0).reshape(2, 4)
            posterior["sampled_curves"][1, 0] = np.nan
            return posterior

    result = run_single_experiment(
        "s000001",
        data_directory=str(tmp_path),
        inference_backend="nf",
        inference_model=PerSampleModel(),
        nf_num_samples=2,
    )
    assert result["experiment_id"] == "s000001"
    prediction = result["prediction_dict"]
    assert prediction["nf_best_idx"] == 0
    np.testing.assert_array_equal(prediction["predicted_sld_xaxis"], [0, 1, 2, 3])
    assert result["param_metrics"]["overall"]["constraint_mape"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mutation", ["all_invalid", "count", "names", "curve_shape", "profile_shape"]
)
def test_nf_rejects_invalid_posterior_contracts(mutation):
    posterior = FakeModel().preprocess_and_sample(num_samples=2)
    if mutation == "all_invalid":
        posterior["log_likelihoods"][:] = np.nan
    elif mutation == "count":
        posterior["log_likelihoods"] = np.array([0.0])
    elif mutation == "names":
        posterior["param_names"] = []
    elif mutation == "curve_shape":
        posterior["sampled_curves"] = np.ones((1, 3))
    else:
        posterior["sampled_sld_profiles"] = np.ones((2, 3))
    with pytest.raises(ValueError):
        _nf_prediction_to_single_prediction_dict(posterior)


def test_finite_statistics_exclude_invalid_rows_without_importance_weighting():
    statistics = compute_nf_sample_statistics(
        [[0.0], [100.0], [np.nan]], [0.0, -100.0, 0.0]
    )
    assert statistics["nf_params_mean"].tolist() == [50.0]
    assert statistics["nf_valid_sample_count"] == 2
    assert statistics["nf_invalid_sample_count"] == 1


def test_default_model_and_injected_layer_validation():
    assert (
        resolve_inference_config(2, "predict")
        == "b_mc_point_neutron_conv_standard_L2_InputQDq.yaml"
    )
    with pytest.raises(ValueError, match="explicit NF config"):
        resolve_inference_config(2, "nf")
    with pytest.raises(ValueError, match="Model has 1 layers"):
        get_inference_prior_limits(FakeModel(), 2)


def test_two_layer_vendor_labels_match_truth_order_without_loading_weights(tmp_path):
    import torch
    from reflectorch.data_generation.priors.parametric_subpriors import (
        SubpriorParametricSampler,
    )

    write_experiment(tmp_path, layer_count=2, ambient=2.0)
    sampler = SubpriorParametricSampler(
        param_ranges={
            "thicknesses": (1.0, 1000.0),
            "roughnesses": (0.0, 60.0),
            "slds": (-8.0, 16.0),
        },
        bound_width_ranges={
            "thicknesses": (0.01, 1000.0),
            "roughnesses": (0.01, 60.0),
            "slds": (0.01, 24.0),
        },
        max_num_layers=2,
        model_name="standard_model",
        device=torch.device("cpu"),
    )
    truth = np.array([100.0, 50.0, 2.0, 3.0, 4.0, 4.0, 6.0, 8.0])

    def sample(**kwargs):
        assert kwargs["kwargs_param_labels"] == {"number_top_to_bottom": True}
        return {
            "predicted_params_array": np.tile(truth, (2, 1)),
            "log_likelihoods": np.array([0.0, 1.0]),
            "sampled_curves": np.ones((2, 3)),
            "q_plot_pred": np.array([0.01, 0.1, 0.2]),
            "sampled_sld_profiles": np.ones((2, 4)),
            "sampled_sld_xaxis": np.arange(4.0),
            "param_names": sampler.param_model.get_param_labels(
                **kwargs["kwargs_param_labels"]
            ),
        }

    model = SimpleNamespace(
        device="cpu",
        trainer=SimpleNamespace(loader=SimpleNamespace(prior_sampler=sampler)),
        preprocess_and_sample=sample,
    )
    result = run_single_experiment(
        "s000001",
        layer_count=2,
        data_directory=str(tmp_path),
        inference_backend="nf",
        inference_model=model,
        nf_num_samples=2,
    )
    assert result["param_metrics"]["overall"]["constraint_mape"] == pytest.approx(0.0)
    assert result["prediction_dict"]["param_names"][:2] == [
        "Thickness L1",
        "Thickness L2",
    ]


def test_singleton_oracle_has_finite_fallback_score_when_k_one_is_omitted():
    result = select_multimodal_ml_sample(
        [[100.0]], [0.0], [100.0], ["thickness"], k_values=(2, 3)
    )
    assert result["selected_k"] == 1
    assert result["selected_silhouette"] == 0.0
