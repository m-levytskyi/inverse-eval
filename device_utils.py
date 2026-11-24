"""Helpers for selecting a supported PyTorch inference device."""

from __future__ import annotations

import torch


INFERENCE_DEVICES = ("cpu", "cuda")


def detect_torch_device(preferred_device: str | None = None) -> str:
    """Validate an explicit inference device or return the CPU default."""
    if preferred_device is None:
        return "cpu"
    device = (
        preferred_device.strip().lower() if isinstance(preferred_device, str) else ""
    )
    if device not in INFERENCE_DEVICES:
        raise ValueError(
            f"Unsupported device '{preferred_device}'. Expected one of: cpu, cuda."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' is not available.")
    return device


def summarize_torch_backends() -> dict[str, bool]:
    """Report CUDA availability. CPU is always available."""
    return {"cuda": torch.cuda.is_available()}
