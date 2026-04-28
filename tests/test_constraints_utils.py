import unittest

import constraints_utils


class LoadConstraintsTests(unittest.TestCase):
    def test_returns_dict(self) -> None:
        data = constraints_utils.load_constraints()
        self.assertIsInstance(data, dict)

    def test_has_constraints_key(self) -> None:
        data = constraints_utils.load_constraints()
        self.assertIn("constraints", data)

    def test_has_constraint_widths_key(self) -> None:
        data = constraints_utils.load_constraints()
        self.assertIn("constraint_widths", data)

    def test_is_cached_on_second_call(self) -> None:
        first = constraints_utils.load_constraints()
        second = constraints_utils.load_constraints()
        self.assertEqual(first, second)


class GetConstraintRangesTests(unittest.TestCase):
    def test_returns_dict(self) -> None:
        ranges = constraints_utils.get_constraint_ranges()
        self.assertIsInstance(ranges, dict)

    def test_contains_known_parameters(self) -> None:
        ranges = constraints_utils.get_constraint_ranges()
        for name in ("thickness", "amb_rough", "sub_rough", "layer_sld"):
            self.assertIn(name, ranges, f"Expected '{name}' in ranges")

    def test_values_are_min_max_tuples(self) -> None:
        ranges = constraints_utils.get_constraint_ranges()
        for name, (lo, hi) in ranges.items():
            self.assertLessEqual(lo, hi, f"min <= max for {name}")


class GetConstraintWidthsTests(unittest.TestCase):
    def test_returns_dict(self) -> None:
        widths = constraints_utils.get_constraint_widths()
        self.assertIsInstance(widths, dict)

    def test_contains_known_parameters(self) -> None:
        widths = constraints_utils.get_constraint_widths()
        for name in ("thickness", "amb_rough", "layer_sld"):
            self.assertIn(name, widths)

    def test_all_widths_positive(self) -> None:
        widths = constraints_utils.get_constraint_widths()
        for name, width in widths.items():
            self.assertGreater(width, 0, f"Width for {name} must be positive")

    def test_thickness_width_is_999(self) -> None:
        # The model_constraints.json defines thickness constraint as [1.0, 1000.0],
        # giving a width of 1000.0 - 1.0 = 999.0 Å.
        widths = constraints_utils.get_constraint_widths()
        self.assertAlmostEqual(widths["thickness"], 999.0)


class GetConstraintRangeTests(unittest.TestCase):
    def test_known_parameter_returns_tuple(self) -> None:
        lo, hi = constraints_utils.get_constraint_range("thickness")
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1000.0)

    def test_roughness_range(self) -> None:
        lo, hi = constraints_utils.get_constraint_range("amb_rough")
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 60.0)

    def test_unknown_parameter_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            constraints_utils.get_constraint_range("nonexistent_param")


class GetConstraintWidthTests(unittest.TestCase):
    def test_known_parameter_returns_float(self) -> None:
        width = constraints_utils.get_constraint_width("amb_rough")
        self.assertAlmostEqual(width, 60.0)

    def test_sld_width(self) -> None:
        width = constraints_utils.get_constraint_width("layer_sld")
        self.assertAlmostEqual(width, 24.0)

    def test_unknown_parameter_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            constraints_utils.get_constraint_width("nonexistent_param")


class GetParameterInfoTests(unittest.TestCase):
    def test_returns_dict_with_min_max(self) -> None:
        info = constraints_utils.get_parameter_info("thickness")
        self.assertIn("min", info)
        self.assertIn("max", info)

    def test_returns_copy_not_original(self) -> None:
        info1 = constraints_utils.get_parameter_info("thickness")
        info2 = constraints_utils.get_parameter_info("thickness")
        self.assertIsNot(info1, info2)

    def test_unknown_parameter_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            constraints_utils.get_parameter_info("nonexistent_param")

    def test_roughness_info_has_unit(self) -> None:
        info = constraints_utils.get_parameter_info("sub_rough")
        self.assertIn("unit", info)


if __name__ == "__main__":
    unittest.main()
