"""Helpers for selecting the best available PyTorch inference device."""

from __future__ import annotations

import torch


def detect_torch_device(preferred_device: str | None = None) -> str:
    """Return the best available torch device.

    If ``preferred_device`` is provided, it is validated first. Otherwise the
    function prefers CUDA, then CPU.
    """
    if preferred_device is not None:
        device = preferred_device.strip().lower()
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise ValueError("Requested device 'cuda' is not available.")
        if device == "cpu":
            return "cpu"
        raise ValueError(
            f"Unsupported device '{preferred_device}'. Expected one of: cpu, cuda."
        )

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def summarize_torch_backends() -> dict[str, bool]:
    """Return availability for the supported inference backends."""
    return {"cuda": torch.cuda.is_available(), "cpu": True}
