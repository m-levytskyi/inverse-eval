"""Execute the single-run notebook's workflow cells with a fake sampler."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

import plotting_utils
import simple_pipeline


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/01_single_experiment.ipynb"


@pytest.mark.parametrize("sample_count", [1, 2])
def test_notebook_single_uses_paired_data_model_limits_and_finite_absolute_samples(
    tmp_path, monkeypatch, capsys, sample_count
):
    data_dir = tmp_path / "dataset/test/1"
    data_dir.mkdir(parents=True)
    (data_dir / "s000001_model.txt").write_text(
        "fronting 2e-6 inf 2\nlayer1 4e-6 100 3\nbacking 8e-6 inf none\n"
    )
    np.savetxt(
        data_dir / "s000001_experimental_curve.dat",
        np.column_stack(
            (np.linspace(0.01, 0.2, 8), np.ones(8), np.full(8, 0.1), np.zeros(8))
        ),
        header="Q R dR dQ",
    )
    names = [
        "Thickness L1",
        "Roughness L1",
        "Roughness sub",
        "SLD L1",
        "SLD sub",
        "r_scale",
        "log10_background",
    ]
    parameters = np.tile([100.0, 2.0, 3.0, 4.0, 8.0, 1.0, -7.0], (sample_count, 1))
    likelihoods = np.arange(sample_count, dtype=float)
    if sample_count > 1:
        parameters[1, -1] = np.nan
    sample = Mock(
        return_value={
            "predicted_params_array": parameters.squeeze()
            if sample_count == 1
            else parameters,
            "log_likelihoods": likelihoods,
            "param_names": names,
            "sampled_curves": np.ones((sample_count, 3)),
            "q_plot_pred": np.array([0.01, 0.1, 0.2]),
            "sampled_sld_profiles": np.ones((sample_count, 4)),
            "sampled_sld_xaxis": np.array([-100.0, 0.0, 100.0, 200.0]),
            "ess": 1.0,
        }
    )
    prior_sampler = SimpleNamespace(
        max_num_layers=1,
        param_dim=7,
        min_bounds=np.array([1.0, 0.0, 0.0, -8.0, -8.0, 0.9, -10.0]),
        max_bounds=np.array([1500.0, 60.0, 60.0, 16.0, 16.0, 1.1, -4.0]),
        min_delta=np.full(7, 0.01),
        max_delta=np.array([1500.0, 60.0, 60.0, 24.0, 24.0, 0.2, 6.0]),
        param_model=SimpleNamespace(get_param_labels=lambda **kwargs: names),
    )
    model = SimpleNamespace(
        device="cpu",
        preprocess_and_sample=sample,
        trainer=SimpleNamespace(loader=SimpleNamespace(prior_sampler=prior_sampler)),
        interpolate_data_to_model_q=Mock(
            side_effect=AssertionError("duplicate interpolation")
        ),
    )
    factory = Mock(return_value=model)
    monkeypatch.setattr(simple_pipeline, "create_inference_model", factory)
    figure = Mock()
    plot = Mock(return_value=figure)
    monkeypatch.setattr(plotting_utils, "plot_simple_comparison", plot)
    plt = SimpleNamespace(rcParams={}, show=Mock())
    namespace: dict[str, Any] = {
        "PROJECT_ROOT": tmp_path,
        "np": np,
        "torch": SimpleNamespace(manual_seed=Mock()),
        "plt": plt,
        "device": "cpu",
        "EXPERIMENT_ID": "s000001",
        "LAYER_COUNT": 1,
        "NF_CONFIG": "fake.yaml",
        "NF_SAMPLES": sample_count,
        "PRIORS_DEV": 30,
        "ENABLE_PREPROCESSING": True,
        "ERROR_THRESHOLD": 0.75,
        "CONSECUTIVE": 5,
    }
    notebook = json.loads(NOTEBOOK.read_text())
    for index in (8, 10, 12, 14, 16, 18, 20):
        exec(
            compile(
                "".join(notebook["cells"][index]["source"]),
                f"{NOTEBOOK}:cell{index}",
                "exec",
            ),
            namespace,
        )

    assert namespace["data_file"].parent == data_dir
    assert factory.call_count == 1
    assert sample.call_count == 1
    options = sample.call_args.kwargs
    assert options["q_values"] is namespace["q_exp"]
    assert options["sigmas"] is namespace["sigmas_exp"]
    assert options["ambient_sld"] == 2.0
    assert options["kwargs_param_labels"] == {"number_top_to_bottom": True}
    assert options["enable_error_bars_filtering"] is False
    assert options["prior_bounds"][3][1] - options["prior_bounds"][3][
        0
    ] == pytest.approx(7.2)
    assert namespace["best_idx"] == 0
    assert namespace["stats"]["nf_params_std"] is None
    np.testing.assert_array_equal(
        namespace["map_params_structural"], [100.0, 2.0, 3.0, 4.0, 8.0]
    )
    assert namespace["param_metrics"]["overall"]["constraint_mape"] == pytest.approx(
        0.0
    )
    assert plot.call_args.kwargs["true_sld_y"][0] == pytest.approx(2.0)
    figure.savefig.assert_called_once()
    plt.show.assert_called_once()
    assert "N/A" in capsys.readouterr().out
