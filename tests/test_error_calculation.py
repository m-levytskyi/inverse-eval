import unittest

import numpy as np

import error_calculation


def _make_perfect_data(n=20):
    """Return q_exp/q_model that match exactly, with flat y values."""
    q = np.linspace(0.01, 0.5, n)
    y = np.ones(n) * 0.5
    sigma = np.ones(n) * 0.01
    return q, y, sigma


class CalculateFitMetricsTests(unittest.TestCase):
    def test_perfect_prediction_r_squared_is_one(self) -> None:
        q = np.linspace(0.01, 0.5, 20)
        # Use non-constant y so ss_tot > 0, enabling a meaningful R² of 1
        y = np.linspace(1.0, 0.1, 20)
        sigma = np.ones(20) * 0.01
        metrics = error_calculation.calculate_fit_metrics(y, y, sigma, q, q)
        self.assertAlmostEqual(metrics["r_squared"], 1.0, places=6)

    def test_returns_required_keys(self) -> None:
        q, y, sigma = _make_perfect_data()
        metrics = error_calculation.calculate_fit_metrics(y, y, sigma, q, q)
        for key in ("r_squared", "mse", "l1_loss", "chi_squared",
                    "reduced_chi_squared", "mean_relative_error", "max_relative_error"):
            self.assertIn(key, metrics)

    def test_all_values_are_floats(self) -> None:
        q, y, sigma = _make_perfect_data()
        metrics = error_calculation.calculate_fit_metrics(y, y, sigma, q, q)
        for key, val in metrics.items():
            self.assertIsInstance(val, float, f"{key} should be float")

    def test_perfect_prediction_mse_is_zero(self) -> None:
        q, y, sigma = _make_perfect_data()
        metrics = error_calculation.calculate_fit_metrics(y, y, sigma, q, q)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=10)

    def test_prediction_offset_increases_mse(self) -> None:
        q, y, sigma = _make_perfect_data()
        y_shifted = y + 0.1
        metrics = error_calculation.calculate_fit_metrics(y, y_shifted, sigma, q, q)
        self.assertGreater(metrics["mse"], 0.0)

    def test_constant_y_exp_r_squared_is_zero(self) -> None:
        """When y_exp is constant, ss_tot == 0 so r_squared should be 0."""
        q = np.linspace(0.01, 0.5, 20)
        y_exp = np.ones(20)  # constant → ss_tot == 0
        y_pred = y_exp + 0.1
        sigma = np.ones(20) * 0.01
        metrics = error_calculation.calculate_fit_metrics(y_exp, y_pred, sigma, q, q)
        self.assertEqual(metrics["r_squared"], 0.0)

    def test_interpolation_to_different_q_grid(self) -> None:
        q_exp = np.linspace(0.01, 0.5, 10)
        q_model = np.linspace(0.01, 0.5, 50)
        y_exp = np.ones(10)
        y_model = np.ones(50)
        sigma = np.ones(10) * 0.01
        metrics = error_calculation.calculate_fit_metrics(y_exp, y_model, sigma, q_exp, q_model)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=10)


class CalculateParameterMetricsTests(unittest.TestCase):
    def _basic_call(self, pred, true, names):
        return error_calculation.calculate_parameter_metrics(pred, true, names)

    def test_identical_params_zero_mape(self) -> None:
        pred = [100.0, 5.0, 2.5e-6]
        true = [100.0, 5.0, 2.5e-6]
        names = ["thickness", "sub_rough", "layer_sld"]
        metrics = self._basic_call(pred, true, names)
        self.assertAlmostEqual(metrics["overall"]["mape"], 0.0, places=8)

    def test_mismatch_length_returns_error_sentinel(self) -> None:
        metrics = self._basic_call([1.0, 2.0], [1.0], ["thickness", "sub_rough"])
        self.assertEqual(metrics["overall"]["mape"], -1)
        self.assertEqual(metrics["overall"]["mse"], -1)

    def test_returns_required_keys(self) -> None:
        pred = [100.0, 5.0]
        true = [80.0, 4.0]
        names = ["thickness", "sub_rough"]
        metrics = self._basic_call(pred, true, names)
        self.assertIn("overall", metrics)
        self.assertIn("by_type", metrics)
        self.assertIn("by_parameter", metrics)

    def test_by_type_thickness(self) -> None:
        pred = [100.0]
        true = [80.0]
        names = ["thickness"]
        metrics = self._basic_call(pred, true, names)
        self.assertIn("thickness", metrics["by_type"])

    def test_by_type_roughness(self) -> None:
        pred = [5.0]
        true = [4.0]
        names = ["sub_rough"]
        metrics = self._basic_call(pred, true, names)
        self.assertIn("roughness", metrics["by_type"])

    def test_by_type_sld(self) -> None:
        pred = [2.5e-6]
        true = [2.0e-6]
        names = ["layer_sld"]
        metrics = self._basic_call(pred, true, names)
        self.assertIn("sld", metrics["by_type"])

    def test_zero_true_value_uses_absolute_error(self) -> None:
        pred = [0.5]
        true = [0.0]
        names = ["sub_rough"]
        metrics = self._basic_call(pred, true, names)
        # Should not raise; absolute error is used
        self.assertAlmostEqual(
            metrics["by_parameter"]["sub_rough"]["percentage_error"], 0.5
        )

    def test_constraint_based_mape_computed(self) -> None:
        pred = [120.0, 5.0]
        true = [100.0, 4.0]
        names = ["thickness", "sub_rough"]
        prior_bounds = np.array([[70.0, 130.0], [0.0, 7.0]])
        metrics = error_calculation.calculate_parameter_metrics(
            pred, true, names,
            prior_bounds=prior_bounds,
            priors_type="constraint_based",
        )
        self.assertIn("constraint_mape", metrics["overall"])
        self.assertIn("constraint_mape", metrics["by_type"]["thickness"])

    def test_mse_calculation(self) -> None:
        pred = [110.0]
        true = [100.0]
        names = ["thickness"]
        metrics = self._basic_call(pred, true, names)
        expected_mse = 100.0  # (110-100)^2
        self.assertAlmostEqual(metrics["overall"]["mse"], expected_mse)

    def test_by_parameter_contains_all_names(self) -> None:
        pred = [100.0, 5.0]
        true = [90.0, 4.5]
        names = ["thickness", "sub_rough"]
        metrics = self._basic_call(pred, true, names)
        for name in names:
            self.assertIn(name, metrics["by_parameter"])


class CalculateResidualsTests(unittest.TestCase):
    def test_returns_required_keys(self) -> None:
        q, y, sigma = _make_perfect_data()
        result = error_calculation.calculate_residuals(y, y, sigma, q, q)
        for key in ("q_exp", "residuals", "standardized_residuals",
                    "residual_stats", "standardized_stats"):
            self.assertIn(key, result)

    def test_perfect_prediction_zero_residuals(self) -> None:
        q, y, sigma = _make_perfect_data()
        result = error_calculation.calculate_residuals(y, y, sigma, q, q)
        np.testing.assert_allclose(result["residuals"], 0.0, atol=1e-10)

    def test_residual_stats_keys(self) -> None:
        q, y, sigma = _make_perfect_data()
        result = error_calculation.calculate_residuals(y, y, sigma, q, q)
        for key in ("mean", "std", "max_abs", "rms"):
            self.assertIn(key, result["residual_stats"])
            self.assertIn(key, result["standardized_stats"])

    def test_nonzero_residuals(self) -> None:
        q, y, sigma = _make_perfect_data()
        y_pred = y + 0.1
        result = error_calculation.calculate_residuals(y, y_pred, sigma, q, q)
        np.testing.assert_allclose(result["residuals"], -0.1, atol=1e-10)


class SummaryStatisticsTests(unittest.TestCase):
    def _fit_metrics(self, r_sq):
        return {
            "r_squared": r_sq,
            "reduced_chi_squared": 1.0,
            "mean_relative_error": 0.01,
        }

    def test_excellent_fit_quality(self) -> None:
        summary = error_calculation.summary_statistics(self._fit_metrics(0.96))
        self.assertTrue(summary["fit_quality"]["excellent"])
        self.assertTrue(summary["fit_quality"]["good"])
        self.assertTrue(summary["fit_quality"]["acceptable"])
        self.assertFalse(summary["fit_quality"]["poor"])

    def test_good_fit_quality(self) -> None:
        summary = error_calculation.summary_statistics(self._fit_metrics(0.92))
        self.assertFalse(summary["fit_quality"]["excellent"])
        self.assertTrue(summary["fit_quality"]["good"])

    def test_acceptable_fit_quality(self) -> None:
        summary = error_calculation.summary_statistics(self._fit_metrics(0.85))
        self.assertFalse(summary["fit_quality"]["good"])
        self.assertTrue(summary["fit_quality"]["acceptable"])

    def test_poor_fit_quality(self) -> None:
        summary = error_calculation.summary_statistics(self._fit_metrics(0.70))
        self.assertTrue(summary["fit_quality"]["poor"])

    def test_parameter_accuracy_excellent(self) -> None:
        param_metrics = {"overall": {"mape": 3.0, "mse": 0.01}}
        summary = error_calculation.summary_statistics(
            self._fit_metrics(0.96), param_metrics
        )
        self.assertTrue(summary["parameter_accuracy"]["excellent"])

    def test_parameter_accuracy_poor(self) -> None:
        param_metrics = {"overall": {"mape": 25.0, "mse": 100.0}}
        summary = error_calculation.summary_statistics(
            self._fit_metrics(0.96), param_metrics
        )
        self.assertTrue(summary["parameter_accuracy"]["poor"])

    def test_uses_constraint_mape_when_present(self) -> None:
        param_metrics = {"overall": {"mape": 30.0, "constraint_mape": 3.0, "mse": 1.0}}
        summary = error_calculation.summary_statistics(
            self._fit_metrics(0.96), param_metrics
        )
        self.assertTrue(summary["parameter_accuracy"]["excellent"])

    def test_no_param_metrics_no_accuracy_key(self) -> None:
        summary = error_calculation.summary_statistics(self._fit_metrics(0.96))
        self.assertNotIn("parameter_accuracy", summary)


if __name__ == "__main__":
    unittest.main()
