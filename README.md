# Engine Emissions Reference Curves

A real-data project on test-bench-style sensor data and combustion
emissions: I build **reference curves** (how an emissions output varies
across an operating parameter, binned and summarized) and a
**deviation/performance score** (how far a new reading sits from its
reference curve, as a z-score), applied to real gas turbine
combustion-emissions sensor data.

## What it does

A pipeline that:

- bins an operating parameter by quantile and summarizes an emissions
  output per bin (`build_reference_curve()`), and
- scores a new reading against its reference curve as a z-score
  (deviation/performance score).

`demo.py` walks through the analysis end to end on real data.

## Data

**UCI Gas Turbine CO and NOx Emission Data Set**: real hourly sensor
measurements from a gas turbine power plant in northwestern Turkey,
2011-2015; 36,733 rows; CC BY 4.0, DOI 10.24432/C5WC95. Columns: ambient
temperature/pressure/humidity, turbine inlet/after temperature, exhaust
pressure, compressor discharge pressure, turbine energy yield (the load
proxy), and CO/NOx emissions concentrations.

This is gas-turbine data; the analytical pattern (deriving a reference
curve relating an emissions output to an operating parameter, then
scoring new readings against it) is independent of engine type.

The dataset is clean: there are zero missing values across all 36,733
rows (`tests/test_data_loading.py::test_no_missing_values`). The data is
fetched by `download_data.py` from a GitHub mirror of the UCI files and
checked against the documented row counts and column layout.

## Results

Analysis of NOx against load (all figures from `demo.py`):

1. **Raw correlation.** TEY (turbine load) correlates with NOX at
   **-0.12**, weakly negative, whereas the naive expectation is that more
   load means more combustion and more NOx.
2. **Ambient temperature as a candidate confound.** AT (ambient
   temperature) has a much stronger raw correlation with NOX (-0.56), so
   I tested whether AT masks a positive load effect.
3. **Partial correlation.** Regressing AT out of both TEY and NOX and
   correlating the residuals gives **-0.20**, more negative than the raw
   correlation, so the masking hypothesis is not supported.
4. **Variable structure.** TEY and TIT (turbine inlet temperature) are
   nearly collinear (r = 0.91); both measure combustion/load intensity.
   This also explains why controlling for TIT instead of AT flips the
   sign unstably: "TEY's effect after removing TIT" is not a well-posed
   question with two nearly-collinear variables.
5. **Non-monotonic curve.** Even within a fixed ambient-temperature band,
   the NOx-vs-load curve is not monotonic (checked with `np.diff` on the
   bin means across 8 bins). That is why `build_reference_curve()` returns
   bin-by-bin statistics rather than fitting a single slope: a one-number
   slope would misrepresent the relationship.

## Tests

16 tests covering quantile binning, a zero-variance edge case in the
deviation scorer, a known-ground-truth recovery test, a flat-relationship
test (no trend is invented where none exists), and data-loading checks
(row count 36,733, no missing values, expected columns).

## Project structure

- `download_data.py`: fetches the dataset (~2.7 MB total).
- `src/data_loading.py`: loading utilities for the dataset.
- `src/reference_curves.py`: quantile binning and reference-curve
  construction.
- `src/deviation_scoring.py`: z-score deviation/performance scoring.
- `demo.py`: end-to-end demo of the analysis above.
- `tests/`: `test_data_loading.py`, `test_reference_curves.py`,
  `test_deviation_scoring.py`.
- `conftest.py`, `pytest.ini`, `requirements-dev.txt`: test configuration
  and dev dependencies.

## Running it

```bash
pip install -r requirements-dev.txt
python3 download_data.py    # fetches the real dataset (~2.7 MB total)
pytest tests/               # 16 tests
python3 demo.py
```

## Possible extensions

- Apply the same reference-curve and deviation-score pipeline to
  reciprocating-engine test-bench data.
- Add further conditioning variables to the reference curves beyond a
  fixed ambient-temperature band.
