from types import SimpleNamespace
from unittest import TestCase, mock

import device_utils


class DetectTorchDeviceTests(TestCase):
    def test_auto_prefers_cuda(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            with mock.patch.object(
                device_utils.torch.backends,
                "mps",
                new=SimpleNamespace(is_available=lambda: True),
                create=True,
            ):
                self.assertEqual(device_utils.detect_torch_device(), "cuda")

    def test_auto_uses_mps_when_cuda_unavailable(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with mock.patch.object(
                device_utils.torch.backends,
                "mps",
                new=SimpleNamespace(is_available=lambda: True),
                create=True,
            ):
                self.assertEqual(device_utils.detect_torch_device(), "mps")

    def test_auto_falls_back_to_cpu(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with mock.patch.object(
                device_utils.torch.backends,
                "mps",
                new=SimpleNamespace(is_available=lambda: False),
                create=True,
            ):
                self.assertEqual(device_utils.detect_torch_device(), "cpu")

    def test_explicit_mps_requires_availability(self) -> None:
        with mock.patch.object(
            device_utils.torch.backends,
            "mps",
            new=SimpleNamespace(is_available=lambda: False),
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "Requested device 'mps'"):
                device_utils.detect_torch_device("mps")

    def test_explicit_device_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            device_utils.detect_torch_device("tpu")
