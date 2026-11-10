import numpy as np
from scipy.special import erf

from utils import to_numpy


def finite_sld_profiles(prediction):
    """Select matching finite profiles, likelihoods, and shared/per-sample depths."""
    sld_x = to_numpy(prediction["sampled_sld_xaxis"])
    profiles = np.atleast_2d(to_numpy(prediction["sampled_sld_profiles"]))
    loglike = to_numpy(prediction["log_likelihoods"]).reshape(-1)
    if profiles.ndim != 2 or len(loglike) != len(profiles):
        raise ValueError(
            "SLD profiles and likelihoods must have matching sample counts"
        )
    if sld_x.ndim == 1 and sld_x.size == profiles.shape[1]:
        finite_x = np.all(np.isfinite(sld_x))
    elif sld_x.shape == profiles.shape:
        finite_x = np.all(np.isfinite(sld_x), axis=1)
    else:
        raise ValueError("SLD depth coordinates must match the profile lengths")
    mask = np.isfinite(loglike) & np.all(np.isfinite(profiles), axis=1) & finite_x
    if not np.any(mask):
        raise ValueError("No finite sampled SLD profiles available to plot.")
    return sld_x[mask] if sld_x.ndim == 2 else sld_x, profiles[mask], loglike[mask]


def sld_profile(z, slds, interfaces, roughnesses):
    """
    Calculate SLD profile for multilayer system with error-function interfaces.
    Args:
        z: 1D numpy array of depth values (Å)
        slds: list of SLDs for each region (ambient, layer1, substrate, ...)
        interfaces: list of interface positions (Å), e.g. [0, thickness1, thickness1+thickness2, ...]
        roughnesses: list of roughnesses (Å) for each interface (len = len(slds)-1)
    Returns:
        profile: 1D numpy array of SLD values at each z
    """
    profile = np.zeros_like(z)
    for i in range(len(slds) - 1):
        z0 = interfaces[i]
        sigma = roughnesses[i]
        erf_arg = (z - z0) / (np.sqrt(2) * sigma)
        step = (slds[i + 1] - slds[i]) / 2 * (1 + erf(erf_arg))
        profile += step
    profile += slds[0]
    return profile
