import unittest

import bootstrap
import bootstrap_windows


NVIDIA_SMI_118 = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03      CUDA Version: 11.8     |
+-----------------------------------------------------------------------------------------+
""".strip()

NVIDIA_SMI_121 = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.1     |
+-----------------------------------------------------------------------------------------+
""".strip()

NVIDIA_SMI_126 = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.28.03              Driver Version: 560.28.03      CUDA Version: 12.6     |
+-----------------------------------------------------------------------------------------+
""".strip()

NVIDIA_SMI_128 = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.42.01              Driver Version: 570.42.01      CUDA Version: 12.8     |
+-----------------------------------------------------------------------------------------+
""".strip()


class ResolveTorchBackendTests(unittest.TestCase):
    def test_auto_on_macos_uses_default_wheel(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="darwin",
            machine="arm64",
            nvidia_smi_output="",
        )

        self.assertEqual(selection.resolved, "cpu")
        self.assertIn("macOS uses the default PyTorch wheel", selection.reason)

    def test_auto_without_nvidia_smi_falls_back_to_cpu(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="linux",
            machine="x86_64",
            nvidia_smi_output="",
        )

        self.assertEqual(selection.resolved, "cpu")
        self.assertIn("fell back to CPU", selection.reason)

    def test_auto_maps_cuda_118(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="linux",
            machine="x86_64",
            nvidia_smi_output=NVIDIA_SMI_118,
        )

        self.assertEqual(selection.resolved, "cu118")

    def test_auto_maps_cuda_121(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="linux",
            machine="x86_64",
            nvidia_smi_output=NVIDIA_SMI_121,
        )

        self.assertEqual(selection.resolved, "cu121")

    def test_auto_maps_cuda_126(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="linux",
            machine="x86_64",
            nvidia_smi_output=NVIDIA_SMI_126,
        )

        self.assertEqual(selection.resolved, "cu126")

    def test_auto_maps_cuda_128(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="linux",
            machine="x86_64",
            nvidia_smi_output=NVIDIA_SMI_128,
        )

        self.assertEqual(selection.resolved, "cu128")

    def test_auto_with_unparseable_output_falls_back_to_cpu(self) -> None:
        selection = bootstrap.resolve_torch_backend(
            "auto",
            platform_name="win32",
            machine="amd64",
            nvidia_smi_output="driver output without a CUDA version",
        )

        self.assertEqual(selection.resolved, "cpu")
        self.assertIn("did not include a parseable CUDA version", selection.reason)

    def test_explicit_cuda_selection_errors_on_macos(self) -> None:
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.resolve_torch_backend(
                "cu121",
                platform_name="darwin",
                machine="arm64",
                nvidia_smi_output="",
            )


class ParserSmokeTests(unittest.TestCase):
    def test_bootstrap_parser_defaults_to_auto(self) -> None:
        parser = bootstrap.build_parser()
        args = parser.parse_args(["torch"])

        self.assertEqual(args.torch_wheel, "auto")

    def test_bootstrap_parser_accepts_auto(self) -> None:
        parser = bootstrap.build_parser()
        args = parser.parse_args(["torch", "--torch-wheel", "auto"])

        self.assertEqual(args.action, "torch")
        self.assertEqual(args.torch_wheel, "auto")

    def test_windows_parser_defaults_to_auto(self) -> None:
        parser = bootstrap_windows.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.torch_wheel, "auto")

    def test_windows_parser_accepts_auto(self) -> None:
        parser = bootstrap_windows.build_parser()
        args = parser.parse_args(["--torch-wheel", "auto"])

        self.assertEqual(args.torch_wheel, "auto")


if __name__ == "__main__":
    unittest.main()
