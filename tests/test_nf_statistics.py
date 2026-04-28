import unittest

import numpy as np

import nf_statistics


def _make_samples(num_samples=100, num_params=3, seed=42):
    rng = np.random.default_rng(seed)
    params = rng.standard_normal((num_samples, num_params))
    log_prob = rng.standard_normal(num_samples)
    return params, log_prob


class ComputeNfSampleStatisticsTests(unittest.TestCase):
    def test_returns_required_keys(self) -> None:
        params, log_prob = _make_samples()
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        for key in ("nf_params_mean", "nf_params_std", "nf_params_percentiles",
                    "nf_log_prob_mean", "nf_log_prob_std"):
            self.assertIn(key, result)

    def test_mean_shape(self) -> None:
        num_params = 4
        params, log_prob = _make_samples(num_params=num_params)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        self.assertEqual(result["nf_params_mean"].shape, (num_params,))

    def test_std_shape(self) -> None:
        num_params = 4
        params, log_prob = _make_samples(num_params=num_params)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        self.assertEqual(result["nf_params_std"].shape, (num_params,))

    def test_percentiles_shape(self) -> None:
        num_params = 3
        params, log_prob = _make_samples(num_params=num_params)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        self.assertEqual(result["nf_params_percentiles"].shape, (5, num_params))

    def test_log_prob_stats_are_scalars(self) -> None:
        params, log_prob = _make_samples()
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        self.assertIsInstance(result["nf_log_prob_mean"], float)
        self.assertIsInstance(result["nf_log_prob_std"], float)

    def test_constant_params_zero_std(self) -> None:
        params = np.ones((50, 2))
        log_prob = np.zeros(50)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        np.testing.assert_allclose(result["nf_params_std"], 0.0, atol=1e-10)

    def test_constant_params_mean_equals_value(self) -> None:
        params = np.full((50, 3), fill_value=7.0)
        log_prob = np.zeros(50)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        np.testing.assert_allclose(result["nf_params_mean"], 7.0, atol=1e-10)

    def test_percentile_ordering(self) -> None:
        params, log_prob = _make_samples(num_samples=200)
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        # Each successive percentile row should be >= the previous for every parameter
        p = result["nf_params_percentiles"]
        for row in range(p.shape[0] - 1):
            self.assertTrue(np.all(p[row] <= p[row + 1]),
                            f"Percentile row {row} > row {row + 1}")

    def test_1d_params_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            nf_statistics.compute_nf_sample_statistics(
                np.ones(10), np.ones(10)
            )

    def test_2d_log_prob_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            nf_statistics.compute_nf_sample_statistics(
                np.ones((10, 3)), np.ones((10, 1))
            )

    def test_mismatched_sample_count_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            nf_statistics.compute_nf_sample_statistics(
                np.ones((10, 3)), np.ones(5)
            )

    def test_accepts_lists_as_input(self) -> None:
        params = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        log_prob = [0.1, 0.2, 0.3]
        result = nf_statistics.compute_nf_sample_statistics(params, log_prob)
        self.assertEqual(result["nf_params_mean"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
