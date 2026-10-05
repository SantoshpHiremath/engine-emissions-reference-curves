"""
Scores how far a new measurement deviates from its reference curve, as a
z-score relative to the nearest operating-point bin's mean and standard
deviation. This is the "Performance Score" half of the concept: the
reference curve alone is descriptive, but a live system needs to judge
whether a current reading is normal or anomalous for its current
operating point.
"""
import numpy as np
import pandas as pd

# Floor applied to a bin's std before dividing, to avoid a real
# ZeroDivisionError/inf/nan on a sparse bin with near-zero variance --
# a genuine edge case in real operating data, not a hypothetical one.
MIN_STD = 1e-6


def score_deviation(reference_curve: pd.DataFrame, x: float, y: float) -> float:
    """Return a z-score for measurement y at operating point x, relative
    to the nearest bin in reference_curve (nearest by x_bin_center).

    z = (y - bin_y_mean) / max(bin_y_std, MIN_STD)

    A z-score near 0 means the measurement matches its reference curve;
    large |z| means the measurement is unusual for that operating point.
    """
    distances = (reference_curve["x_bin_center"] - x).abs()
    nearest_idx = distances.idxmin()
    bin_row = reference_curve.loc[nearest_idx]

    std = max(bin_row["y_std"], MIN_STD)
    return (y - bin_row["y_mean"]) / std
