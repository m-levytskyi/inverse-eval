from unittest import TestCase, mock

import device_utils


class DetectTorchDeviceTests(TestCase):
    def test_default_is_cpu_even_when_cuda_is_available(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(device_utils.detect_torch_device(), "cpu")

    def test_explicit_cuda_when_available(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(device_utils.detect_torch_device(" CUDA "), "cuda")

    def test_explicit_cuda_requires_availability(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Requested device 'cuda'"):
                device_utils.detect_torch_device("cuda")

    def test_backend_summary_reports_cuda_availability(self) -> None:
        with mock.patch("device_utils.torch.cuda.is_available", return_value=True):
            self.assertEqual(device_utils.summarize_torch_backends(), {"cuda": True})

    def test_explicit_device_rejects_unknown_values(self) -> None:
        for device in ("mps", "tpu"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "Expected one of: cpu, cuda"):
                    device_utils.detect_torch_device(device)
