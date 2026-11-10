import argparse
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch

import build_denoised_experimental_dataset as denoising
from compare_clean_vs_theoretical import compare_dataset
from sld_profile_utils import finite_sld_profiles
from utils import to_numpy, validate_positional_experiment_ids


def test_tensor_conversion_detaches_gradients():
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    np.testing.assert_array_equal(to_numpy(tensor), [1.0, 2.0])


@pytest.mark.parametrize(
    "ids", [["s000001"], ["s000000"], ["s000000", "s000000"], ["s000001", "s000000"]]
)
def test_positional_rows_reject_missing_duplicate_or_reordered_ids(ids):
    with pytest.raises(ValueError, match="complete ordered experiment sequence"):
        validate_positional_experiment_ids(ids, 2)


def test_complete_positional_comparison_matches_rows(tmp_path):
    files = [tmp_path / f"s{i:06d}_theoretical_curve.dat" for i in range(2)]
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    for file, row in zip(files, values):
        np.savetxt(file, np.column_stack(([0.1, 0.2], row)))
    results = compare_dataset(values, None, files, 1e-6, 1e-9, 1e-7, 1e-9)
    assert all(result.r_match for result in results)
    with pytest.raises(ValueError, match="complete ordered experiment sequence"):
        compare_dataset(values, None, files[1:], 1e-6, 1e-9, 1e-7, 1e-9)


def test_denoising_rejects_subset_before_writing_curves(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.savetxt(source / "s000001_experimental_curve.dat", [[0.1, 3], [0.2, 4]])
    pickle_file = tmp_path / "curves.pkl"
    with pickle_file.open("wb") as handle:
        pickle.dump(
            {"q": np.array([0.1, 0.2]), "R_denoised": np.array([[1, 2], [3, 4]])},
            handle,
        )
    output = tmp_path / "output"
    args = argparse.Namespace(
        input_dir=str(source),
        pickle_file=str(pickle_file),
        output_dir=str(output),
        plot_count=0,
        pattern="s*_experimental_curve.dat",
    )
    with patch.object(denoising, "parse_args", return_value=args):
        with pytest.raises(ValueError, match="complete ordered experiment sequence"):
            denoising.main()
    assert not list(output.glob("*.dat"))


@pytest.mark.parametrize("depths", [np.array([0, 1]), np.array([[0, 1], [2, 3]])])
def test_finite_sld_rows_keep_depth_and_likelihood_alignment(depths):
    x, profiles, likelihoods = finite_sld_profiles(
        {
            "sampled_sld_xaxis": depths,
            "sampled_sld_profiles": [[1, 2], [np.nan, 4]],
            "log_likelihoods": [3, 4],
        }
    )
    np.testing.assert_array_equal(profiles, [[1, 2]])
    np.testing.assert_array_equal(likelihoods, [3])
    np.testing.assert_array_equal(x, [0, 1] if depths.ndim == 1 else [[0, 1]])


def test_finite_sld_profiles_rejects_empty_valid_set():
    with pytest.raises(ValueError, match="No finite"):
        finite_sld_profiles(
            {
                "sampled_sld_xaxis": [0, 1],
                "sampled_sld_profiles": [1, 2],
                "log_likelihoods": [np.nan],
            }
        )
