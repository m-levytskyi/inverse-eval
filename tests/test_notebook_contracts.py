import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np

from error_calculation import calculate_parameter_metrics
from nf_statistics import validate_nf_samples
from parameter_discovery import get_prior_bounds_for_experiment
from simple_pipeline import _extend_prior_bounds_for_nf, align_samples_to_true


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/05_multimodal_posterior_mape.ipynb"
)


def notebook_helpers(namespace):
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][5]["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_notebook_metric_helper_matches_current_api():
    namespace: dict[str, Any] = notebook_helpers(
        {
            "contextlib": contextlib,
            "io": io,
            "calculate_parameter_metrics": calculate_parameter_metrics,
        }
    )
    result = namespace["quiet_parameter_metrics"]([101.0], [100.0], ["thickness"])
    assert np.isclose(result["overall"]["constraint_mape"], 100 / 999)


def test_notebook_sampler_uses_raw_grid_ambient_and_full_sample_validity():
    names = ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]
    truth = {"params": [100, 5, 5, 4, 8], "param_names": names, "ambient_sld": 2.0}
    q, curve, errors = np.array([0.1, 0.2]), np.array([1.0, 0.2]), np.array([0.1, 0.02])
    sampler = Mock(
        return_value={
            "predicted_params_array": [
                [100, 5, 5, 4, 8, 1, -6],
                [100, 5, 5, 4, 8, np.nan, -6],
            ],
            "log_likelihoods": [0, 10],
            "param_names": names + ["r_scale", "log10_background"],
        }
    )
    namespace: dict[str, Any] = {
        "np": np,
        "DATA_DIRECTORY": "unused",
        "LAYER_COUNT": 1,
        "USE_THEORETICAL": False,
        "ENABLE_PREPROCESSING": True,
        "PREPROCESSING_THRESHOLD": 0.5,
        "PREPROCESSING_CONSECUTIVE": 3,
        "PREPROCESSING_REMOVE_SINGLES": False,
        "PRIORS_TYPE": "constraint_based",
        "PRIORS_DEVIATION": 0.3,
        "FIX_SLD_MODE": "none",
        "NF_NUM_SAMPLES": 2,
        "Q_RESOLUTION": 0.1,
        "NF_ENABLE_IMPORTANCE_SAMPLING": False,
        "CLIP_PREDICTION": True,
        "USE_SIGMAS_INPUT": False,
        "discover_experiment_files": lambda *args, **kwargs: ("curve", "model", 1),
        "load_experimental_data": lambda *args, **kwargs: (q, curve, errors),
        "parse_true_parameters_from_model_file": lambda *args: {"1_layer": truth},
        "get_inference_prior_limits": lambda *args: (None, None),
        "get_prior_bounds_for_experiment": get_prior_bounds_for_experiment,
        "_extend_prior_bounds_for_nf": _extend_prior_bounds_for_nf,
        "validate_nf_samples": validate_nf_samples,
        "align_samples_to_true": align_samples_to_true,
    }
    notebook_helpers(namespace)
    prior_sampler = SimpleNamespace(
        param_dim=7,
        max_num_layers=1,
        min_bounds=np.array([1, 0, 0, -8, -8, 0.9, -10]),
        max_bounds=np.array([1500, 60, 60, 16, 16, 1.1, -4]),
    )
    model = SimpleNamespace(
        preprocess_and_sample=sampler,
        trainer=SimpleNamespace(loader=SimpleNamespace(prior_sampler=prior_sampler)),
    )
    result = namespace["posterior_for_experiment"]("s000000", model)
    options = sampler.call_args.kwargs
    assert options["q_values"] is q
    assert options["reflectivity_curve"] is curve
    assert options["ambient_sld"] == 2.0
    assert options["enable_error_bars_filtering"] is False
    assert options["calc_sampled_curves"] is True
    assert options["kwargs_param_labels"] == {"number_top_to_bottom": True}
    assert "sigmas" not in options
    np.testing.assert_array_equal(result["valid_mask"], [True, False])
