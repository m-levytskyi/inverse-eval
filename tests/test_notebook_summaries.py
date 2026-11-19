import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nf_statistics import (
    compute_nf_sample_statistics,
    select_global_ml_index,
    validate_nf_samples,
)
from utils import to_numpy


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def execute_cell(filename, index, namespace):
    notebook = json.loads((NOTEBOOKS / filename).read_text())
    namespace.update(
        np=np,
        to_numpy=to_numpy,
        compute_nf_sample_statistics=compute_nf_sample_statistics,
        select_global_ml_index=select_global_ml_index,
        validate_nf_samples=validate_nf_samples,
    )
    exec("".join(notebook["cells"][index]["source"]), namespace)


@pytest.mark.parametrize("singleton", [False, True])
def test_orso_summary_preserves_indices_and_handles_single_valid_sample(
    singleton, capsys
):
    namespace = {
        "prediction": {
            "predicted_params_array": [4, 100] if singleton else [[0, 100], [4, 100]],
            "log_likelihoods": [3] if singleton else [np.nan, 3],
            "param_names": ["SLD layer", "thickness"],
        },
        "AMBIENT_SLD": 2,
        "PRIOR_BOUNDS": [(0, 16), (1, 1000)],
        "NF_SAMPLES": 1 if singleton else 2,
        "expected_dim": 2,
    }
    execute_cell("03_orso_l2_nf_inference.ipynb", 10, namespace)
    assert namespace["best_idx"] == (0 if singleton else 1)
    np.testing.assert_array_equal(namespace["map_params"], [4, 100])
    assert "N/A" in capsys.readouterr().out


def test_pnr_summary_handles_one_valid_row_without_losing_curve_index(capsys):
    namespace: dict[str, Any] = {
        "results": {},
        "PARAM_ORDER": ["thickness", "roughness"],
        "FIT_REFERENCE": {"test": {"thickness": 100, "roughness": 5}},
    }
    execute_cell("04_pnr_polarisation_l2_nf_inference.ipynb", 12, namespace)
    result = {
        "prediction": {
            "predicted_params_array": [[0, 0], [100, 5]],
            "log_likelihoods": [np.nan, 3],
            "reflectivity_curve_interp": [1, 1],
            "sampled_curves": [[9, 9], [1, 1]],
        },
        "prior_bounds": [(1, 1000), (0, 60)],
    }
    namespace["summarize_result"]("test", result)
    assert result["best_idx"] == 1
    assert result["weighted_mse"] == 0
    assert "N/A" in capsys.readouterr().out
