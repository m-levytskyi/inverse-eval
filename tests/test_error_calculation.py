import io
import unittest
from contextlib import redirect_stdout

import error_calculation


class CalculateParameterMetricsTests(unittest.TestCase):
    def _basic_call(self, pred, true, names):
        return error_calculation.calculate_parameter_metrics(pred, true, names)

    def test_identical_params_zero_mape(self) -> None:
        pred = [100.0, 5.0, 2.5e-6]
        true = [100.0, 5.0, 2.5e-6]
        names = ["thickness", "sub_rough", "layer_sld"]
        metrics = self._basic_call(pred, true, names)
        self.assertAlmostEqual(metrics["overall"]["mape"], 0.0, places=8)
        self.assertAlmostEqual(metrics["overall"]["constraint_mape"], 0.0, places=8)

    def test_mismatch_length_returns_error_sentinel(self) -> None:
        metrics = self._basic_call([1.0, 2.0], [1.0], ["thickness", "sub_rough"])
        self.assertEqual(metrics["overall"]["mape"], -1)
        self.assertEqual(metrics["overall"]["constraint_mape"], -1)

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
        metrics = error_calculation.calculate_parameter_metrics(pred, true, names)
        self.assertIn("constraint_mape", metrics["overall"])
        self.assertIn("constraint_mape", metrics["by_type"]["thickness"])

    def test_by_parameter_contains_all_names(self) -> None:
        pred = [100.0, 5.0]
        true = [90.0, 4.5]
        names = ["thickness", "sub_rough"]
        metrics = self._basic_call(pred, true, names)
        for name in names:
            self.assertIn(name, metrics["by_parameter"])


class PrintMetricsReportTests(unittest.TestCase):
    def test_reports_overall_and_per_parameter_constraint_mape(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            error_calculation.print_metrics_report(
                {
                    "overall": {"constraint_mape": 3.456, "mape": 30.0},
                    "by_parameter": {
                        "thickness": {"constraint_percentage_error": 2.5},
                        "sub_rough": {"constraint_percentage_error": 4.5},
                    },
                },
            )

        self.assertEqual(
            output.getvalue(),
            "Overall constraint-based MAPE: 3.46%\n"
            "  thickness: 2.50%\n"
            "  sub_rough: 4.50%\n",
        )

    def test_reports_constraint_mape_as_unavailable_when_absent(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            error_calculation.print_metrics_report(
                {
                    "overall": {"mape": 30.0},
                    "by_parameter": {"thickness": {}},
                }
            )

        self.assertEqual(
            output.getvalue(),
            "Overall constraint-based MAPE: N/A\n  thickness: N/A\n",
        )


if __name__ == "__main__":
    unittest.main()
