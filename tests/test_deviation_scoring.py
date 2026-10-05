"""
Tests for scoring how far a new measurement deviates from its reference
curve -- the natural second half of a "Performance Score" concept: not
just building the reference curve, but using it to judge whether a live
reading is normal or anomalous for its operating point.
"""
import numpy as np
import pandas as pd
import pytest

from deviation_scoring import score_deviation


class TestScoreDeviation:
    @pytest.fixture
    def reference_curve(self):
        return pd.DataFrame({
            "x_bin_center": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_mean": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y_std": [1.0, 1.0, 1.0, 1.0, 1.0],
            "count": [100, 100, 100, 100, 100],
        })

    def test_measurement_matching_reference_scores_near_zero(self, reference_curve):
        # x=3 is closest to bin center 3.0, whose y_mean is 30.0 -- a
        # measurement of y=30 at x=3 should be judged completely normal.
        z = score_deviation(reference_curve, x=3.0, y=30.0)
        assert abs(z) < 0.1

    def test_measurement_two_std_above_reference_scores_around_two(self, reference_curve):
        # y_std at the nearest bin (x=3) is 1.0, so y=32 is 2 std above
        # the reference mean of 30.0.
        z = score_deviation(reference_curve, x=3.0, y=32.0)
        assert 1.8 < z < 2.2

    def test_measurement_below_reference_scores_negative(self, reference_curve):
        z = score_deviation(reference_curve, x=3.0, y=28.0)
        assert z < 0

    def test_uses_nearest_bin_when_x_falls_between_bin_centers(self, reference_curve):
        # x=2.4 is closer to bin center 2.0 than 3.0, so it should use
        # that bin's mean (20.0) and std (1.0) as the reference.
        z = score_deviation(reference_curve, x=2.4, y=22.0)
        assert 1.8 < z < 2.2

    def test_zero_std_bin_does_not_divide_by_zero(self):
        # A bin with zero variance (e.g. only one data point, or all
        # identical values) is a real edge case in sparse regions of
        # real operating data -- must not raise ZeroDivisionError or
        # return inf/nan silently.
        curve = pd.DataFrame({
            "x_bin_center": [1.0],
            "y_mean": [10.0],
            "y_std": [0.0],
            "count": [1],
        })
        z = score_deviation(curve, x=1.0, y=15.0)
        assert np.isfinite(z)
