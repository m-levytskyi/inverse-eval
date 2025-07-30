import matplotlib.pyplot as plt
import numpy as np

def plot_sld_profile(depths, sld_profile, label=None, ax=None, color=None, title=None, xlabel='z [Å]', ylabel='SLD [10⁻⁶ Å⁻²]'):
    """
    Plot a single SLD profile (or multiple if given as a list).
    Args:
        depths: 1D array of depth values (Å)
        sld_profile: 1D array of SLD values (×10⁻⁶ Å⁻²), or list of such arrays
        label: Label for the profile (or list of labels)
        ax: Optional matplotlib axis to plot on
        color: Optional color or list of colors
        title: Optional plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    Returns:
        The matplotlib axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Support plotting multiple profiles
    if isinstance(sld_profile, (list, tuple)) and hasattr(sld_profile[0], '__len__'):
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
    ax.grid(True, alpha=0.3)
    return ax
