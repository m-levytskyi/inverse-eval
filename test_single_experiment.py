#!/usr/bin/env python3
"""
Test case for single experiment analysis using s005888.
This test validates that the inference pipeline produces consistent results
by comparing predicted parameters against expected values with tolerance thresholds.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Import our modular utilities
from parameter_discovery import discover_experiment_files
from error_calculation import calculate_fit_metrics
from reflectorch import EasyInferenceModel

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ParameterValidationError(Exception):
    """Custom exception for parameter validation failures."""

    pass


class ReflectometryTest:
    """Test class for validating reflectometry inference results."""

    def __init__(
        self, experiment_id="s005888", tolerance=0.10, fit_quality_threshold=0.99
    ):
        """Initialize the test case."""
        self.experiment_id = experiment_id
        self.tolerance = tolerance
        self.fit_threshold = fit_quality_threshold
        self.layer_count = 1  # s005888 is a 1-layer experiment

        # Expected parameter values (polished values from reference script)
        self.expected_params = {
            "Thickness L1": 282.34,
            "Roughness L1": 5.66,
            "Roughness sub": 30.22,
            "SLD L1": 11.00,
            "SLD sub": 14.71,
        }

        self.test_results = {}

    def load_experimental_data(self, data_file):
        """Load experimental data from file."""
        print(f"Loading test data from: {data_file}")
        data = np.loadtxt(data_file)
        q_exp = data[:, 0]
        curve_exp = data[:, 1]
        sigmas_exp = data[:, 2] if data.shape[1] > 2 else None
        print(f"Loaded {len(q_exp)} data points")
        return q_exp, curve_exp, sigmas_exp

    def run_inference(self, q_exp, curve_exp, prior_bounds):
        """Run inference on experimental data."""
        print("Running inference for test...")

        # Detect CI environment and force CPU usage
        import os

        is_ci = os.getenv("CI", "").lower() in ("true", "1", "yes") or os.getenv(
            "GITHUB_ACTIONS", ""
        ).lower() in ("true", "1", "yes")
        device = (
            "cpu" if is_ci else "cpu"
        )  # Always use CPU for now to ensure compatibility

        print(f"Using device: {device}")
        if is_ci:
            print("CI environment detected - forcing CPU usage")

        # Setup inference model with explicit device
        inference_model = EasyInferenceModel(
            "b_mc_point_neutron_conv_standard_L1_InputQDq", device=device
        )

        # Interpolate data
        q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
            q_exp, curve_exp
        )

        # Run prediction
        prediction_dict = inference_model.predict(
            reflectivity_curve=exp_curve_interp,
            prior_bounds=prior_bounds,
            q_values=q_model,
            q_resolution=0.1,
            polish_prediction=True,
            calc_pred_curve=True,
            calc_pred_sld_profile=True,
            calc_polished_sld_profile=True,
        )

        return q_model, prediction_dict

    def validate_parameters(self, prediction_dict):
        """Validate predicted parameters against expected values."""
        print(f"\nValidating parameters (tolerance: {self.tolerance * 100:.0f}%)")
        print("-" * 60)

        polished_params = prediction_dict["polished_params_array"]
        param_names = prediction_dict["param_names"]

        failed_validations = []

        for expected_name, expected_value in self.expected_params.items():
            # Find exact match in parameter names
            param_index = None
            if expected_name in param_names:
                param_index = param_names.index(expected_name)

            if param_index is None:
                print(f"WARNING: Could not find parameter '{expected_name}' in results")
                continue

            predicted_value = polished_params[param_index]

            # Calculate relative error
            relative_error = (
                abs(predicted_value - expected_value) / abs(expected_value)
                if expected_value != 0
                else abs(predicted_value)
            )

            # Check if within tolerance
            within_tolerance = relative_error <= self.tolerance
            status = "PASS" if within_tolerance else "FAIL"

            print(
                f"{expected_name:15} | Pred: {predicted_value:8.2f} | "
                f"Exp: {expected_value:8.2f} | Error: {relative_error * 100:6.2f}% | {status}"
            )

            if not within_tolerance:
                failed_validations.append(expected_name)

        if failed_validations:
            error_msg = (
                f"Parameter validation failed for: {', '.join(failed_validations)}"
            )
            raise ParameterValidationError(error_msg)

        print("All parameters within tolerance")

    def run_test(self):
        """Run the complete test case."""
        print("=" * 80)
        print(f"RUNNING TEST CASE: {self.experiment_id}")
        print("=" * 80)

        try:
            # 1. Discover experiment files
            print("1. Discovering experiment files...")
            data_file, model_file, detected_layer_count = discover_experiment_files(
                self.experiment_id, "data", layer_count=self.layer_count
            )

            if not data_file:
                raise FileNotFoundError(
                    f"Could not find data file for {self.experiment_id}"
                )

            print(f"Found data file: {data_file}")
            if model_file:
                print(f"Found model file: {model_file}")

            # 2. Load experimental data
            print("\n2. Loading experimental data...")
            q_exp, curve_exp, sigmas_exp = self.load_experimental_data(data_file)

            # 3. Set up specific prior bounds for s005888 (from notebook)
            print("\n3. Setting up prior bounds...")
            prior_bounds = [
                (142.0, 427.0),  # layer thicknesses (top to bottom)
                (3.5, 10.5),  # ambient roughness
                (14.0, 42.0),  # substrate roughness
                (11, 16),  # layer SLD
                (11, 16),  # substrate SLD
            ]
            print("Using specific prior bounds for s005888 (from notebook)")

            # 4. Run inference
            print("\n4. Running inference...")
            q_model, prediction_dict = self.run_inference(
                q_exp, curve_exp, prior_bounds
            )
            print("Inference completed successfully")

            # 5. Validate parameters
            print("\n5. Validating parameters...")
            self.validate_parameters(prediction_dict)

            print("\n" + "=" * 80)
            print("ALL TESTS PASSED!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\n" + "=" * 80)
            print(f"TEST FAILED: {str(e)}")
            print("=" * 80)
            return False


def main():
    """Run the test suite."""
    print("REFLECTOMETRY INFERENCE TEST SUITE")
    print("=" * 80)

    test = ReflectometryTest()
    success = test.run_test()

    if success:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
