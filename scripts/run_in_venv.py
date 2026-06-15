#!/usr/bin/env python3
"""Run a command through the repository-local virtual environment.

This keeps Git hooks cross-platform while still forcing project commands to use
the dependencies installed by this repository's bootstrap flow.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def main() -> int:
    root = repo_root()
    python = venv_python(root)
    if not python.exists():
        print(
            f"Missing repository virtual environment Python: {python}\n"
            "Run setup from the inverse-eval repo root, then retry.",
            file=sys.stderr,
        )
        return 1

    if not sys.argv[1:]:
        print("Usage: run_in_venv.py <python-args...>", file=sys.stderr)
        return 2

    return subprocess.run([str(python), *sys.argv[1:]], cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
