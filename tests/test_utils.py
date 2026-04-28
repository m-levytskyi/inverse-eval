import json
import unittest
from pathlib import Path

import numpy as np

import utils


class ConvertToJsonSerializableTests(unittest.TestCase):
    def test_plain_dict_passthrough(self) -> None:
        data = {"a": 1, "b": "hello"}
        result = utils.convert_to_json_serializable(data)
        self.assertEqual(result, {"a": 1, "b": "hello"})
        json.dumps(result)  # Should not raise

    def test_nested_dict(self) -> None:
        data = {"outer": {"inner": 42}}
        result = utils.convert_to_json_serializable(data)
        self.assertEqual(result, {"outer": {"inner": 42}})

    def test_list_passthrough(self) -> None:
        data = [1, 2, 3]
        result = utils.convert_to_json_serializable(data)
        self.assertEqual(result, [1, 2, 3])

    def test_tuple_converted_to_list(self) -> None:
        result = utils.convert_to_json_serializable((1, 2, 3))
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])

    def test_numpy_array_converted_to_list(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = utils.convert_to_json_serializable(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_numpy_integer_converted(self) -> None:
        val = np.int64(7)
        result = utils.convert_to_json_serializable(val)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7)

    def test_numpy_floating_converted(self) -> None:
        val = np.float32(3.14)
        result = utils.convert_to_json_serializable(val)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=5)

    def test_object_with_dict_attr_converted(self) -> None:
        class Dummy:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = utils.convert_to_json_serializable(Dummy())
        self.assertEqual(result, {"x": 1, "y": "hello"})
        json.dumps(result)

    def test_non_serializable_falls_back_to_string(self) -> None:
        # An object with no __dict__ and not JSON-serializable → converted to string
        class Unserializable:
            __slots__ = ()

        result = utils.convert_to_json_serializable(Unserializable())
        self.assertIsInstance(result, str)

    def test_deeply_nested_mixed_structure(self) -> None:
        data = {"arr": np.array([1, 2]), "nested": {"tup": (3, 4)}}
        result = utils.convert_to_json_serializable(data)
        self.assertEqual(result["arr"], [1, 2])
        self.assertEqual(result["nested"]["tup"], [3, 4])
        json.dumps(result)


class ValidateLayerCountTests(unittest.TestCase):
    def test_valid_zero(self) -> None:
        self.assertTrue(utils.validate_layer_count(0))

    def test_valid_one(self) -> None:
        self.assertTrue(utils.validate_layer_count(1))

    def test_valid_two(self) -> None:
        self.assertTrue(utils.validate_layer_count(2))

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            utils.validate_layer_count(-1)

    def test_too_large_raises(self) -> None:
        with self.assertRaises(ValueError):
            utils.validate_layer_count(3)

    def test_float_raises(self) -> None:
        with self.assertRaises(ValueError):
            utils.validate_layer_count(1.0)

    def test_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            utils.validate_layer_count("1")


class FormatParameterValueTests(unittest.TestCase):
    def test_sld_with_units(self) -> None:
        result = utils.format_parameter_value("layer_sld", 2.5e-6)
        self.assertIn("Å⁻²", result)
        self.assertIn("2.50e-06", result)

    def test_sld_without_units(self) -> None:
        result = utils.format_parameter_value("layer_sld", 2.5e-6, units=False)
        self.assertNotIn("Å", result)

    def test_thickness_with_units(self) -> None:
        result = utils.format_parameter_value("thickness1", 100.0)
        self.assertIn("Å", result)
        self.assertIn("100.0", result)

    def test_thickness_without_units(self) -> None:
        result = utils.format_parameter_value("thickness1", 100.0, units=False)
        self.assertNotIn("Å", result)
        self.assertIn("100.0", result)

    def test_roughness_with_units(self) -> None:
        result = utils.format_parameter_value("sub_rough", 5.0)
        self.assertIn("Å", result)
        self.assertIn("5.0", result)

    def test_roughness_without_units(self) -> None:
        result = utils.format_parameter_value("sub_rough", 5.0, units=False)
        self.assertNotIn("Å", result)

    def test_unknown_parameter_default_format(self) -> None:
        result = utils.format_parameter_value("some_param", 1.234)
        self.assertIn("1.234", result)

    def test_unknown_parameter_no_units(self) -> None:
        result = utils.format_parameter_value("some_param", 1.234, units=False)
        self.assertIn("1.234", result)


class EnsureDirectoryExistsTests(unittest.TestCase):
    def test_creates_new_directory(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as base:
            new_dir = Path(base) / "subdir" / "nested"
            result = utils.ensure_directory_exists(new_dir)
            self.assertTrue(result.exists())
            self.assertTrue(result.is_dir())

    def test_returns_path_object(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as base:
            result = utils.ensure_directory_exists(base)
            self.assertIsInstance(result, Path)

    def test_existing_directory_is_idempotent(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as base:
            utils.ensure_directory_exists(base)
            # Calling again should not raise
            result = utils.ensure_directory_exists(base)
            self.assertTrue(result.exists())


class FormatTimeDurationTests(unittest.TestCase):
    def test_seconds_format(self) -> None:
        result = utils.format_time_duration(45.0)
        self.assertIn("seconds", result)
        self.assertIn("45.0", result)

    def test_minutes_format(self) -> None:
        result = utils.format_time_duration(120.0)
        self.assertIn("minutes", result)
        self.assertIn("2.0", result)

    def test_hours_format(self) -> None:
        result = utils.format_time_duration(7200.0)
        self.assertIn("hours", result)
        self.assertIn("2.0", result)

    def test_boundary_exactly_60_seconds(self) -> None:
        result = utils.format_time_duration(60.0)
        self.assertIn("minutes", result)

    def test_boundary_exactly_3600_seconds(self) -> None:
        result = utils.format_time_duration(3600.0)
        self.assertIn("hours", result)


if __name__ == "__main__":
    unittest.main()
