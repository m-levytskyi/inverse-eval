import unittest

import numpy as np

import sld_profile_utils


class SldProfileTests(unittest.TestCase):
    def test_single_interface_step_shape(self) -> None:
        z = np.linspace(-10, 30, 200)
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        self.assertEqual(profile.shape, z.shape)

    def test_far_left_equals_ambient_sld(self) -> None:
        z = np.linspace(-100, 100, 500)
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        # Far to the left of the interface → should approach slds[0]
        self.assertAlmostEqual(float(profile[0]), slds[0], places=3)

    def test_far_right_equals_substrate_sld(self) -> None:
        z = np.linspace(-100, 100, 500)
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        # Far to the right of the interface → should approach slds[1]
        self.assertAlmostEqual(float(profile[-1]), slds[1], places=3)

    def test_midpoint_is_average_of_two_slds(self) -> None:
        z = np.array([0.0])  # exactly at the interface
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        # erf(0) == 0, so profile = slds[0] + (slds[1]-slds[0])/2 * (1 + 0) = 3.0
        self.assertAlmostEqual(float(profile[0]), 3.0, places=10)

    def test_two_layer_system_shape(self) -> None:
        z = np.linspace(-10, 200, 500)
        slds = [0.0, 4.0, 2.0, 6.0]  # ambient + 2 layers + substrate
        interfaces = [0.0, 100.0, 150.0]
        roughnesses = [1.0, 1.0, 1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        self.assertEqual(profile.shape, z.shape)

    def test_two_layer_far_left_is_ambient(self) -> None:
        z = np.linspace(-100, 300, 1000)
        slds = [0.0, 4.0, 2.0, 6.0]
        interfaces = [0.0, 100.0, 150.0]
        roughnesses = [1.0, 1.0, 1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        self.assertAlmostEqual(float(profile[0]), slds[0], places=3)

    def test_two_layer_far_right_is_substrate(self) -> None:
        z = np.linspace(-100, 300, 1000)
        slds = [0.0, 4.0, 2.0, 6.0]
        interfaces = [0.0, 100.0, 150.0]
        roughnesses = [1.0, 1.0, 1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        self.assertAlmostEqual(float(profile[-1]), slds[-1], places=3)

    def test_zero_roughness_produces_sharp_step(self) -> None:
        """With very small roughness the step should be very sharp."""
        z = np.array([-1.0, -0.001, 0.001, 1.0])
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1e-6]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        # Left of interface ≈ 0, right of interface ≈ 6
        self.assertAlmostEqual(float(profile[0]), 0.0, places=3)
        self.assertAlmostEqual(float(profile[-1]), 6.0, places=3)

    def test_monotone_increase_across_positive_step(self) -> None:
        z = np.linspace(-10, 10, 100)
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        # Profile should be non-decreasing for a positive SLD step
        self.assertTrue(np.all(np.diff(profile) >= 0))

    def test_output_dtype_float(self) -> None:
        z = np.linspace(-10, 10, 50)
        slds = [0.0, 6.0]
        interfaces = [0.0]
        roughnesses = [1.0]
        profile = sld_profile_utils.sld_profile(z, slds, interfaces, roughnesses)
        self.assertTrue(np.issubdtype(profile.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
