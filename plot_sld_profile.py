#!/usr/bin/env python3
"""
Utility for plotting SLD (Scattering Length Density) profiles.

Supports both regular and paper/publication styling.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_sld_profile(
    depths,
    sld_profile,
    label=None,
    ax=None,
    color=None,
    title=None,
    xlabel="z [Å]",
    ylabel=r"SLD [$10^{-6}$ Å$^{-2}$]",
    save_path=None,
    paper_mode=False,
):
    """
    Plot a single SLD profile (or multiple if given as a list).

    Args:
        depths: 1D array of depth values (Å)
        sld_profile: 1D array of SLD values ($10^{-6}$ Å$^{-2}$), or list of such arrays
        label: Label for the profile (or list of labels)
        ax: Optional matplotlib axis to plot on
        color: Optional color or list of colors
        title: Optional plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
        paper_mode: Use paper styling if True

    Returns:
        The matplotlib axis with the plot.
    """
    ax = _create_sld_plot(
        depths, sld_profile, label, ax, color, title, xlabel, ylabel
    )
    
    if save_path and ax is not None:
        if paper_mode:
            save_path = Path(save_path).with_suffix(".pdf")
            plt.savefig(save_path)
            print(f"Paper figure saved to: {save_path}")
        else:
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")

    return ax


def _create_sld_plot(
    depths,
    sld_profile,
    label=None,
    ax=None,
    color=None,
    title=None,
    xlabel="z [Å]",
    ylabel=r"SLD [$10^{-6}$ Å$^{-2}$]",
):
    """Internal function to create SLD plot."""
    print(f"\nDEBUG [_create_sld_plot]: Creating SLD plot")
    print(f"  depths shape: {np.array(depths).shape if hasattr(depths, '__len__') else 'scalar'}")
    print(f"  sld_profile type: {type(sld_profile)}")
    if isinstance(sld_profile, (list, tuple)):
        print(f"  sld_profile is list/tuple with {len(sld_profile)} elements")
        for i, prof in enumerate(sld_profile):
            print(f"    Profile {i} shape: {np.array(prof).shape if hasattr(prof, '__len__') else 'scalar'}")
    else:
        print(f"  sld_profile shape: {np.array(sld_profile).shape if hasattr(sld_profile, '__len__') else 'scalar'}")
    print(f"  label: {label}")
    print(f"  color: {color}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Support plotting multiple profiles
    if isinstance(sld_profile, (list, tuple)) and hasattr(sld_profile[0], "__len__"):
        for i, prof in enumerate(sld_profile):
            lbl = label[i] if isinstance(label, (list, tuple)) else None
            clr = color[i] if isinstance(color, (list, tuple)) else None
            ax.plot(depths, prof, label=lbl, color=clr)
    else:
        ax.plot(depths, sld_profile, label=label, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    ax.tick_params(axis="both", which="both", length=0)
    return ax
