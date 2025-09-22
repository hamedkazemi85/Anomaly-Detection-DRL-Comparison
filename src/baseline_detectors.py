import numpy as np

def baseline_from_series(series, threshold=0.95):
    """
    Convert a 'thr-lev' series (probabilities or precomputed detector signal) to
    a detection vector similar to your notebook's CSV plots.
    This just returns the series as numpy and also a binary detection vector where
    signal >= threshold is 1.
    """
    arr = np.asarray(series)
    binary = (arr >= threshold).astype(int)
    return arr, binary
