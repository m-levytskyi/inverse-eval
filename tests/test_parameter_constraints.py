import unittest

import numpy as np

import parameter_constraints


class ApplyPhysicalConstraintsTests(unittest.TestCase):
    def _make_prediction(self, pred, polished, names):
        return {
            "predicted_params_array": np.array(pred, dtype=float),
            "polished_params_array": np.array(polished, dtype=float),
            "param_names": names,
        }

    def test_no_violations_unchanged(self) -> None:
        pred = self._make_prediction([100.0, 5.0], [100.0, 5.0],
                                     ["thickness", "sub_rough"])
        result = parameter_constraints.apply_physical_constraints(pred)
        np.testing.assert_array_almost_equal(
            result["predicted_params_array"], [100.0, 5.0]
        )

    def test_negative_thickness_clamped_to_min(self) -> None:
        pred = self._make_prediction([-5.0], [-5.0], ["thickness"])
        result = parameter_constraints.apply_physical_constraints(pred)
        self.assertGreaterEqual(result["predicted_params_array"][0], 0.1)
        self.assertGreaterEqual(result["polished_params_array"][0], 0.1)

    def test_negative_roughness_clamped_to_zero(self) -> None:
        pred = self._make_prediction([100.0, -2.0], [100.0, -2.0],
                                     ["thickness", "sub_rough"])
        result = parameter_constraints.apply_physical_constraints(pred)
        self.assertGreaterEqual(result["predicted_params_array"][1], 0.0)
        self.assertGreaterEqual(result["polished_params_array"][1], 0.0)

    def test_missing_arrays_returns_unchanged(self) -> None:
        pred = {"param_names": ["thickness"]}
        result = parameter_constraints.apply_physical_constraints(pred)
        self.assertEqual(result, pred)

    def test_sld_params_not_constrained(self) -> None:
        # SLD values can be negative and should NOT be clamped
        pred = self._make_prediction([-5.0], [-5.0], ["layer_sld"])
        result = parameter_constraints.apply_physical_constraints(pred)
        np.testing.assert_array_almost_equal(
            result["predicted_params_array"], [-5.0]
        )

    def test_returns_expected_values_for_valid_input(self) -> None:
        pred = self._make_prediction([100.0], [100.0], ["thickness"])
        result = parameter_constraints.apply_physical_constraints(pred)
        self.assertEqual(result["param_names"], ["thickness"])
        np.testing.assert_array_almost_equal(
            result["predicted_params_array"], [100.0]
        )
        np.testing.assert_array_almost_equal(
            result["polished_params_array"], [100.0]
        )

    def test_polished_also_clamped_independently(self) -> None:
        pred = {
            "predicted_params_array": np.array([100.0]),
            "polished_params_array": np.array([-3.0]),
            "param_names": ["thickness"],
        }
        result = parameter_constraints.apply_physical_constraints(pred)
        # predicted was already fine
        np.testing.assert_almost_equal(result["predicted_params_array"][0], 100.0)
        # polished was negative and must be clamped
        self.assertGreaterEqual(result["polished_params_array"][0], 0.1)


class ValidatePhysicalParametersTests(unittest.TestCase):
    def test_all_valid_returns_true(self) -> None:
        params = [100.0, 5.0, 2.5e-6]
        names = ["thickness", "sub_rough", "layer_sld"]
        self.assertTrue(parameter_constraints.validate_physical_parameters(params, names))

    def test_negative_thickness_returns_false(self) -> None:
        params = [-1.0]
        names = ["thickness"]
        self.assertFalse(parameter_constraints.validate_physical_parameters(params, names))

    def test_negative_roughness_returns_false(self) -> None:
        params = [100.0, -0.5]
        names = ["thickness", "sub_rough"]
        self.assertFalse(
            parameter_constraints.validate_physical_parameters(params, names)
        )

    def test_negative_sld_still_valid(self) -> None:
        # SLD can be negative
        params = [-5.0]
        names = ["layer_sld"]
        self.assertTrue(parameter_constraints.validate_physical_parameters(params, names))

    def test_boundary_zero_roughness_valid(self) -> None:
        params = [0.0]
        names = ["sub_rough"]
        self.assertTrue(parameter_constraints.validate_physical_parameters(params, names))

    def test_boundary_zero_thickness_valid(self) -> None:
        # 0 is not < 0 so no violation
        params = [0.0]
        names = ["thickness"]
        self.assertTrue(parameter_constraints.validate_physical_parameters(params, names))

    def test_experiment_id_in_output(self) -> None:
        import contextlib
        import io
        params = [-1.0]
        names = ["thickness"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            parameter_constraints.validate_physical_parameters(
                params, names, experiment_id="exp_001"
            )
        self.assertIn("exp_001", captured.getvalue())


class GetParameterConstraintsInfoTests(unittest.TestCase):
    def test_returns_dict(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        self.assertIsInstance(info, dict)

    def test_has_thickness_key(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        self.assertIn("thickness", info)

    def test_has_roughness_key(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        self.assertIn("roughness", info)

    def test_thickness_min_value(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        self.assertAlmostEqual(info["thickness"]["min_value"], 0.1)

    def test_roughness_min_value(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        self.assertAlmostEqual(info["roughness"]["min_value"], 0.0)

    def test_each_entry_has_reason(self) -> None:
        info = parameter_constraints.get_parameter_constraints_info()
        for key, val in info.items():
            self.assertIn("reason", val, f"No 'reason' key for {key}")


if __name__ == "__main__":
    unittest.main()
