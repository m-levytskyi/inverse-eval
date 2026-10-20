"""Check platform selection without downloading wheels for other platforms."""

from pathlib import Path
from unittest import TestCase

from packaging.requirements import Requirement


class ScientificRequirementsTests(TestCase):
    def test_scipy_selection_by_platform(self) -> None:
        requirements = [
            Requirement(line)
            for line in (Path(__file__).resolve().parents[1] / "requirements.txt")
            .read_text()
            .splitlines()
            if line.startswith("scipy")
        ]
        for system in ("darwin", "linux", "win32"):
            for python in ("3.10", "3.11", "3.12"):
                for arch in ("arm64", "x86_64"):
                    with self.subTest(system=system, python=python, arch=arch):
                        selected = [
                            req
                            for req in requirements
                            if req.marker is None
                            or req.marker.evaluate(
                                {
                                    "sys_platform": system,
                                    "python_version": python,
                                    "platform_machine": arch,
                                }
                            )
                        ]
                        self.assertEqual(len(selected), 1)
                        spec = selected[0].specifier
                        self.assertIn("1.14.1", spec)
                        if system == "darwin":
                            self.assertNotIn("1.15.3", spec)
                        else:
                            self.assertIn("1.15.3", spec)
