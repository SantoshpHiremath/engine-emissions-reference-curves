"""
Tests for loading the UCI Gas Turbine CO and NOx Emission dataset (real
hourly sensor measurements from a gas turbine in northwestern Turkey,
2011-2015, CC BY 4.0). Unlike the SECOM/Air Quality datasets used in the
industrial-sensor-fault-detection project, this one is clean: zero missing values across all 36,733 rows.
"""
import pandas as pd
import pytest

from data_loading import load_all_years, EXPECTED_COLUMNS


class TestLoadAllYears:
    def test_loads_all_five_years(self):
        df = load_all_years("data")
        assert set(df["year"].unique()) == {2011, 2012, 2013, 2014, 2015}

    def test_total_row_count_matches_known_dataset_size(self):
        # 36,733 total rows across all 5 years, matching the dataset's
        # documented size.
        df = load_all_years("data")
        assert len(df) == 36733

    def test_no_missing_values(self):
        # This dataset has zero missing cells, unlike SECOM/Air Quality.
        df = load_all_years("data")
        assert df.isna().sum().sum() == 0

    def test_has_expected_sensor_columns(self):
        df = load_all_years("data")
        for col in EXPECTED_COLUMNS:
            assert col in df.columns

    def test_nox_and_co_are_positive(self):
        # Emissions concentrations cannot be physically negative; this
        # is a basic sanity check on the real data, not a synthetic
        # assumption.
        df = load_all_years("data")
        assert (df["NOX"] > 0).all()
        assert (df["CO"] > 0).all()
