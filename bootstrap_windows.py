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

from bootstrap import BootstrapError, SUPPORTED_TORCH_WHEELS, run_setup  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up this repository on Windows with the shared bootstrap flow."
    )
    parser.add_argument(
        "--torch-wheel",
        default="auto",
        choices=list(SUPPORTED_TORCH_WHEELS),
        help="PyTorch wheel variant to install. `auto` detects the best pinned backend for the current machine.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_setup(args.torch_wheel)
    except BootstrapError as exc:
        print(exc.format(), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
