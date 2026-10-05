"""
Loading utilities for the UCI Gas Turbine CO and NOx Emission dataset.

Real hourly sensor measurements from a gas turbine power plant in
northwestern Turkey, 2011-2015 (36,733 rows, CC BY 4.0, DOI
10.24432/C5WC95). Columns: AT (ambient temperature, C), AP (ambient
pressure, mbar), AH (ambient humidity, %), AFDP (air filter pressure
drop, mbar), GTEP (gas turbine exhaust pressure, mbar), TIT (turbine
inlet temperature, C), TAT (turbine after temperature, C), TEY (turbine
energy yield, MWH -- the load/output proxy), CDP (compressor discharge
pressure, mbar), CO and NOX (emissions concentrations, mg/m3).

This is gas-turbine data. It serves as a real, measured combustion-
emissions-vs-operating-condition dataset for the analytical method
(deriving reference/performance curves from test-bench-style sensor
data), which is independent of engine type.
"""
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]
YEARS = [2011, 2012, 2013, 2014, 2015]


def load_all_years(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all five yearly CSVs, tagging each row with
    its source year.
    """
    frames = []
    for year in YEARS:
        path = Path(data_dir) / f"gt_{year}.csv"
        df = pd.read_csv(path)
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
