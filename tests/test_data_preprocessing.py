import unittest

import numpy as np

import data_preprocessing


def _make_arrays(values, sigmas=None):
    q = np.linspace(0.01, 0.5, len(values))
    y = np.array(values, dtype=float)
    if sigmas is None:
        sigmas = np.ones(len(values)) * 0.01
    else:
        sigmas = np.array(sigmas, dtype=float)
    return q, y, sigmas


class RemoveNegativeValuesTests(unittest.TestCase):
    def test_all_positive_unchanged(self) -> None:
        q, y, sigma = _make_arrays([1.0, 2.0, 3.0])
        q2, y2, s2 = data_preprocessing.remove_negative_values(q, y, sigma)
        np.testing.assert_array_equal(q2, q)
        np.testing.assert_array_equal(y2, y)
        np.testing.assert_array_equal(s2, sigma)

    def test_removes_negative_intensity(self) -> None:
        q, y, sigma = _make_arrays([1.0, -0.5, 2.0])
        q2, y2, s2 = data_preprocessing.remove_negative_values(q, y, sigma)
        self.assertEqual(len(y2), 2)
        self.assertTrue(np.all(y2 > 0))

    def test_removes_zero_intensity(self) -> None:
        q, y, sigma = _make_arrays([1.0, 0.0, 2.0])
        q2, y2, s2 = data_preprocessing.remove_negative_values(q, y, sigma)
        # Zero is NOT positive so it should be removed
        self.assertEqual(len(y2), 2)

    def test_all_negative_returns_empty(self) -> None:
        q, y, sigma = _make_arrays([-1.0, -2.0])
        q2, y2, s2 = data_preprocessing.remove_negative_values(q, y, sigma)
        self.assertEqual(len(y2), 0)

    def test_arrays_remain_aligned(self) -> None:
        q, y, sigma = _make_arrays([1.0, -0.5, 3.0], sigmas=[0.01, 0.99, 0.03])
        q2, y2, s2 = data_preprocessing.remove_negative_values(q, y, sigma)
        self.assertEqual(len(q2), len(y2))
        self.assertEqual(len(q2), len(s2))
        # Verify correct values survived
        self.assertAlmostEqual(y2[0], 1.0)
        self.assertAlmostEqual(y2[1], 3.0)
        self.assertAlmostEqual(s2[1], 0.03)


class FilterHighErrorPointsTests(unittest.TestCase):
    def test_low_error_data_unchanged(self) -> None:
        q, y, sigma = _make_arrays([1.0] * 10, sigmas=[0.01] * 10)
        q2, y2, s2 = data_preprocessing.filter_high_error_points(q, y, sigma)
        np.testing.assert_array_equal(q2, q)

    def test_truncation_at_consecutive_high_error_region(self) -> None:
        # 5 good points then 3 consecutive high-error points
        y_vals = [1.0] * 5 + [1.0] * 3
        sigma_vals = [0.01] * 5 + [0.9] * 3  # last 3 have relative error 0.9 > 0.5
        q, y, sigma = _make_arrays(y_vals, sigmas=sigma_vals)
        q2, y2, s2 = data_preprocessing.filter_high_error_points(
            q, y, sigma, error_threshold=0.5, consecutive_threshold=3
        )
        # Should truncate before the first of the 3 consecutive high-error points
        self.assertLessEqual(len(y2), 5)

    def test_isolated_high_error_removed_when_requested(self) -> None:
        # Good, one spike, good, good
        y_vals = [1.0, 1.0, 1.0, 1.0, 1.0]
        sigma_vals = [0.01, 0.9, 0.01, 0.01, 0.01]  # index 1 is an isolated spike
        q, y, sigma = _make_arrays(y_vals, sigmas=sigma_vals)
        q2, y2, s2 = data_preprocessing.filter_high_error_points(
            q, y, sigma, error_threshold=0.5, consecutive_threshold=10,
            remove_singles=True,
        )
        self.assertEqual(len(y2), 4)

    def test_isolated_high_error_kept_when_not_requested(self) -> None:
        y_vals = [1.0, 1.0, 1.0, 1.0, 1.0]
        sigma_vals = [0.01, 0.9, 0.01, 0.01, 0.01]
        q, y, sigma = _make_arrays(y_vals, sigmas=sigma_vals)
        q2, y2, s2 = data_preprocessing.filter_high_error_points(
            q, y, sigma, error_threshold=0.5, consecutive_threshold=10,
            remove_singles=False,
        )
        # Isolated spike is kept
        self.assertEqual(len(y2), 5)

    def test_output_arrays_aligned(self) -> None:
        q, y, sigma = _make_arrays([1.0] * 8, sigmas=[0.01] * 8)
        q2, y2, s2 = data_preprocessing.filter_high_error_points(q, y, sigma)
        self.assertEqual(len(q2), len(y2))
        self.assertEqual(len(q2), len(s2))


class PreprocessExperimentalDataTests(unittest.TestCase):
    def test_clean_data_passes_through(self) -> None:
        q, y, sigma = _make_arrays([1.0] * 10, sigmas=[0.01] * 10)
        q2, y2, s2 = data_preprocessing.preprocess_experimental_data(q, y, sigma)
        np.testing.assert_array_equal(q2, q)

    def test_negative_values_removed(self) -> None:
        q, y, sigma = _make_arrays([1.0, -1.0, 2.0], sigmas=[0.01, 0.01, 0.01])
        q2, y2, s2 = data_preprocessing.preprocess_experimental_data(q, y, sigma)
        self.assertTrue(np.all(y2 > 0))

    def test_result_shorter_or_equal_than_input(self) -> None:
        y_vals = [1.0] * 5 + [-1.0] + [1.0] * 3 + [1.0] * 3
        sigma_vals = [0.01] * 5 + [0.01] + [0.01] * 3 + [0.9] * 3
        q, y, sigma = _make_arrays(y_vals, sigmas=sigma_vals)
        q2, y2, s2 = data_preprocessing.preprocess_experimental_data(q, y, sigma)
        self.assertLessEqual(len(y2), len(y))

    def test_output_arrays_aligned(self) -> None:
        q, y, sigma = _make_arrays([1.0, 0.5, 2.0], sigmas=[0.01, 0.01, 0.01])
        q2, y2, s2 = data_preprocessing.preprocess_experimental_data(q, y, sigma)
        self.assertEqual(len(q2), len(y2))
        self.assertEqual(len(q2), len(s2))


if __name__ == "__main__":
    unittest.main()
