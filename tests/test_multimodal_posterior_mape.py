import unittest

import numpy as np

from constraints_utils import get_constraint_width
from multimodal_posterior_mape import (
    constraint_mape,
    select_current_ml_sample,
    select_multimodal_ml_sample,
)


class MultimodalPosteriorMapeTests(unittest.TestCase):
    def test_current_ml_uses_global_max_log_likelihood(self) -> None:
        samples = np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
            ]
        )
        log_likelihoods = np.array([-3.0, -1.0, -2.0])

        selected = select_current_ml_sample(
            samples,
            log_likelihoods,
            true_params=np.array([20.0, 2.0]),
            param_names=["thickness", "sub_rough"],
        )

        self.assertEqual(selected["selected_index"], 1)
        np.testing.assert_allclose(selected["selected_params"], samples[1])

    def test_multimodal_selects_nearest_cluster_ml_sample(self) -> None:
        samples = np.array(
            [
                [100.0, 10.0],
                [102.0, 11.0],
                [98.0, 9.0],
                [300.0, 20.0],
                [302.0, 19.5],
                [298.0, 20.5],
            ]
        )
        log_likelihoods = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        true_params = np.array([300.0, 20.0])
        param_names = ["thickness", "sub_rough"]

        current = select_current_ml_sample(
            samples, log_likelihoods, true_params, param_names
        )
        multimodal = select_multimodal_ml_sample(
            samples,
            log_likelihoods,
            true_params,
            param_names,
            k_values=(1, 2),
            random_state=0,
        )

        self.assertEqual(current["selected_index"], 0)
        self.assertEqual(multimodal["selected_index"], 3)
        self.assertLess(
            multimodal["selected_constraint_mape"],
            current["constraint_mape"],
        )
        self.assertEqual(multimodal["selected_k"], 2)

    def test_degenerate_samples_fall_back_to_k_one(self) -> None:
        samples = np.ones((5, 2))
        log_likelihoods = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        true_params = np.ones(2)
        param_names = ["thickness", "sub_rough"]

        multimodal = select_multimodal_ml_sample(
            samples,
            log_likelihoods,
            true_params,
            param_names,
            k_values=(1, 2, 3),
        )

        self.assertEqual(multimodal["selected_k"], 1)
        self.assertEqual(multimodal["selected_index"], 4)
        self.assertEqual(multimodal["selected_constraint_mape"], 0.0)

    def test_constraint_mape_uses_constraint_widths(self) -> None:
        result = constraint_mape(
            pred_params=np.array([110.0]),
            true_params=np.array([100.0]),
            param_names=["thickness"],
        )

        expected = 10.0 / get_constraint_width("thickness") * 100.0
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
