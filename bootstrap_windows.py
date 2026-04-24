#!/usr/bin/env python3
"""
Windows entrypoint for the shared repository bootstrap flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap import BootstrapError, run_setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up this repository on Windows with the shared bootstrap flow."
    )
    parser.add_argument(
        "--torch-wheel",
        default="cpu",
        choices=["cpu", "cu118", "cu121"],
        help="PyTorch wheel variant to install. CPU is the safest default for workshop machines.",
    )
    args = parser.parse_args()

    try:
        run_setup(args.torch_wheel)
    except BootstrapError as exc:
        print(exc.format(), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
