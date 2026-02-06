import numpy as np


def filter_and_truncate(Q, R, dR, threshold=0.5, consecutive=3, remove_singles=False):
    """
    Filters and/or truncates neutron reflectivity data without modifying inputs.

    Parameters:
    - Q, R, dR: 1D NumPy arrays (same length)
    - threshold: float, relative error threshold (e.g., 0.5)
    - consecutive: int, number of consecutive high-error points to trigger truncation
    - remove_singles: bool, if True, removes isolated high-error points before truncating

    Returns:
    - New Q, R, dR arrays with applied filtering/truncation
    """
    Q = Q.copy()
    R = R.copy()
    dR = dR.copy()

    rel_error = dR / R

    if remove_singles:
        mask = rel_error < threshold
        Q = Q[mask]
        R = R[mask]
        dR = dR[mask]
        rel_error = rel_error[mask]

    count = 0
    for idx, err in enumerate(rel_error):
        if err >= threshold:
            count += 1
            if count >= consecutive:
                cutoff = idx - consecutive + 1
                return Q[:cutoff], R[:cutoff], dR[:cutoff]
        else:
            count = 0

    return Q, R, dR
