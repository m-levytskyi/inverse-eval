import numpy as np
from scipy.special import erf


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
