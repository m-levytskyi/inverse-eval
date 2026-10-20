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

    def test_auto_uses_cpu_even_when_mps_available(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with mock.patch.object(
                device_utils.torch.backends,
                "mps",
                new=SimpleNamespace(is_available=lambda: True),
                create=True,
            ):
                self.assertEqual(device_utils.detect_torch_device(), "cpu")

    def test_auto_falls_back_to_cpu(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with mock.patch.object(
                device_utils.torch.backends,
                "mps",
                new=SimpleNamespace(is_available=lambda: False),
                create=True,
            ):
                self.assertEqual(device_utils.detect_torch_device(), "cpu")

    def test_explicit_mps_is_unsupported(self) -> None:
        with mock.patch.object(
            device_utils.torch.backends,
            "mps",
            new=SimpleNamespace(is_available=lambda: True),
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported device 'mps'"):
                device_utils.detect_torch_device("mps")

    def test_explicit_device_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            device_utils.detect_torch_device("tpu")

    def test_explicit_cpu_overrides_available_cuda(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(device_utils.detect_torch_device(" CPU "), "cpu")

    def test_explicit_cuda_requires_availability(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Requested device 'cuda'"):
                device_utils.detect_torch_device("cuda")

    def test_explicit_cuda_when_available(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(device_utils.detect_torch_device("cuda"), "cuda")

    def test_summary_only_includes_supported_backends(self) -> None:
        for cuda_available in (False, True):
            with self.subTest(cuda_available=cuda_available):
                with mock.patch(
                    "device_utils.torch.cuda.is_available", return_value=cuda_available
                ):
                    self.assertEqual(
                        device_utils.summarize_torch_backends(),
                        {"cuda": cuda_available, "cpu": True},
                    )
