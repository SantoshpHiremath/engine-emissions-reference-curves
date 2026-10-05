"""
Reference-curve construction: bins a dataset by an operating parameter
(e.g. turbine load) and reports the mean, spread, and count of an output
metric (e.g. NOx emissions) within each bin -- the core building block
behind an emissions/performance "reference curve."

Deliberately simple and transparent (quantile binning + per-bin
statistics) rather than a fitted parametric curve or black-box model,
because a reference curve's purpose is to be inspectable and explainable
to engineers who will judge a live measurement against it -- a hidden
model fit would undermine that.
"""
import pandas as pd


def bin_by_quantile(series: pd.Series, n_bins: int) -> pd.Series:
    """Assign each value to one of n_bins roughly-equal-count quantile
    bins. Quantile (not equal-width) binning is used so that bins stay
    populated even when the underlying variable's distribution is skewed,
    which real operating data usually is.
    """
    return pd.qcut(series, q=n_bins, duplicates="drop")


def build_reference_curve(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Build a reference curve: bin df by x_col into n_bins quantile bins,
    and report each bin's center (mean x in that bin), mean y, std of y,
    and row count.

    The bin center is computed as the mean of x within the bin (not the
    bin edge) so the resulting curve reflects where the data actually
    sits, not an arbitrary boundary.
    """
    bins = bin_by_quantile(df[x_col], n_bins)
    grouped = df.groupby(bins, observed=True)

    curve = pd.DataFrame({
        "x_bin_center": grouped[x_col].mean(),
        "y_mean": grouped[y_col].mean(),
        "y_std": grouped[y_col].std(),
        "count": grouped[y_col].count(),
    }).reset_index(drop=True).sort_values("x_bin_center").reset_index(drop=True)

    return curve
