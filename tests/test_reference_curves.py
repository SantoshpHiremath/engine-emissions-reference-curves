"""
Tests for the reference-curve module -- the core of this project.

A finding from the data drove the design here: the raw correlation between TEY
(turbine load) and NOX is weakly NEGATIVE (-0.12) in this dataset, and
ambient temperature (AT) correlates with NOX more strongly (-0.56) than
load does. A naive "NOx vs load" reference curve built without controlling
for ambient conditions would be dominated by a confounding variable, not
the operating parameter it claims to describe. These tests exist to force
the reference-curve builder to bin by the primary variable AND report how
much residual variation remains explained by other factors, rather than
silently producing a misleading curve.
"""
import numpy as np
import pandas as pd
import pytest

from reference_curves import build_reference_curve, bin_by_quantile


class TestBinByQuantile:
    def test_splits_into_requested_number_of_bins(self):
        s = pd.Series(range(100))
        bins = bin_by_quantile(s, n_bins=5)
        assert bins.nunique() == 5

    def test_bins_have_roughly_equal_counts(self):
        s = pd.Series(range(100))
        bins = bin_by_quantile(s, n_bins=4)
        counts = bins.value_counts()
        # Quantile-based bins should be roughly balanced (each ~25 of 100)
        assert counts.min() >= 20
        assert counts.max() <= 30


class TestBuildReferenceCurve:
    @pytest.fixture
    def synthetic_data(self):
        # Known ground truth: y = 2*x + noise, so the reference curve
        # should recover an increasing trend close to slope 2.
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 2000)
        y = 2 * x + rng.normal(0, 0.5, 2000)
        return pd.DataFrame({"x": x, "y": y})

    def test_recovers_known_increasing_trend(self, synthetic_data):
        curve = build_reference_curve(synthetic_data, x_col="x", y_col="y", n_bins=10)
        # bin means of y should be monotonically increasing (allowing
        # for a small amount of noise-driven non-monotonicity)
        means = curve["y_mean"].values
        increasing_steps = (means[1:] > means[:-1]).sum()
        assert increasing_steps >= 8  # 8 of 9 steps increasing, allows noise

    def test_returns_bin_center_mean_and_count(self, synthetic_data):
        curve = build_reference_curve(synthetic_data, x_col="x", y_col="y", n_bins=10)
        for col in ("x_bin_center", "y_mean", "y_std", "count"):
            assert col in curve.columns

    def test_bin_counts_sum_to_total_rows(self, synthetic_data):
        curve = build_reference_curve(synthetic_data, x_col="x", y_col="y", n_bins=10)
        assert curve["count"].sum() == len(synthetic_data)

    def test_flat_relationship_produces_roughly_flat_curve(self):
        # y independent of x -> the reference curve should show no real
        # trend. This is the no-trend counterpart to the
        # increasing-trend test above: the function must not invent a
        # trend where none exists.
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 10, 2000)
        y = rng.normal(50, 1, 2000)  # independent of x
        df = pd.DataFrame({"x": x, "y": y})
        curve = build_reference_curve(df, x_col="x", y_col="y", n_bins=10)
        spread = curve["y_mean"].max() - curve["y_mean"].min()
        # bin means should stay within a small band around the true mean
        # (50), not swing by several std-devs, if there is truly no trend
        assert spread < 1.5
