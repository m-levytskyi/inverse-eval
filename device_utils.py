"""Helpers for selecting the best available PyTorch inference device."""

from __future__ import annotations

import torch


def detect_torch_device(preferred_device: str | None = None) -> str:
    """Return the best available torch device.

    If ``preferred_device`` is provided, it is validated first. Otherwise the
    function prefers CUDA, then MPS, then CPU.
    """
    if preferred_device is not None:
        device = preferred_device.strip().lower()
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise ValueError("Requested device 'cuda' is not available.")
        if device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            raise ValueError("Requested device 'mps' is not available.")
        if device == "cpu":
            return "cpu"
        raise ValueError(
            f"Unsupported device '{preferred_device}'. Expected one of: cpu, cuda, mps."
        )

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def summarize_torch_backends() -> dict[str, bool]:
    """Return a simple availability summary for common torch backends."""
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "cuda": torch.cuda.is_available(),
        "mps": bool(mps_backend is not None and mps_backend.is_available()),
    }
