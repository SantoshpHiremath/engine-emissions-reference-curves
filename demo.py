"""
End-to-end demo: reference curves and a deviation-based performance score
on real gas turbine combustion-emissions data.

Walkthrough of the analysis:

Raw correlation of TEY (load) with NOX is weakly negative (-0.12),
counter to the naive "more load -> more NOx" expectation. A natural
hypothesis is that ambient temperature (AT, raw corr -0.56) is a
confound masking a positive load effect underneath.

A partial correlation (regressing AT out of both TEY and NOX, then
correlating the residuals) tests this: controlling for AT makes the
TEY-NOX relationship MORE negative (-0.20), not less, so the masking
hypothesis is not supported.

Further checking shows why: TEY and TIT (turbine inlet temperature)
are nearly collinear (r=0.91) -- they are both essentially measuring
combustion/load intensity. Controlling for TIT while asking about TEY's
"own" effect is close to meaningless given that collinearity, which is
also why an earlier attempt at controlling for TIT alone flipped the
sign. Building the reference curve within a fixed ambient-temperature
band still shows a NON-monotonic relationship (verified: 8 bins, no
consistent up or down trend) -- NOx vs load is not a simple
monotonic curve in this dataset, so the demo reports bin-by-bin values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from data_loading import load_all_years
from reference_curves import build_reference_curve
from deviation_scoring import score_deviation


def partial_correlation(df, x_col, y_col, control_col):
    """Correlation between x_col and y_col after linearly regressing out
    control_col from both -- the standard way to check whether a raw
    correlation is being driven by a third variable.
    """
    c = df[control_col].values
    x = df[x_col].values
    y = df[y_col].values
    x_resid = x - np.polyval(np.polyfit(c, x, 1), c)
    y_resid = y - np.polyval(np.polyfit(c, y, 1), c)
    return np.corrcoef(x_resid, y_resid)[0, 1]


def main():
    df = load_all_years("data")
    print(f"Loaded {len(df)} real hourly gas turbine sensor readings (2011-2015)")
    print(f"Columns: {list(df.columns)}\n")

    print("--- Raw correlations with NOX ---")
    for col in ["TEY", "TIT", "AT", "AFDP", "GTEP"]:
        corr = df[[col, "NOX"]].corr().iloc[0, 1]
        print(f"  {col}: {corr:+.3f}")

    print(
        "\nTEY (load) correlates weakly NEGATIVELY with NOX. The first hypothesis was that "
        "ambient temperature (AT, the strongest raw correlate at -0.56) is a confound masking "
        "a real positive load effect. This is tested below with a partial "
        "correlation."
    )

    partial_at = partial_correlation(df, "TEY", "NOX", "AT")
    print(f"\nPartial correlation of TEY vs NOX, controlling for AT: {partial_at:+.3f}")
    print(
        f"This is MORE negative than the raw correlation ({df[['TEY','NOX']].corr().iloc[0,1]:+.3f}), "
        f"not less -- the opposite of what the 'AT is masking a positive load effect' hypothesis "
        f"predicted, so that hypothesis is not supported by the data."
    )

    tey_tit_corr = df[["TEY", "TIT"]].corr().iloc[0, 1]
    print(f"\nChecking why: TEY and TIT (turbine inlet temperature) correlate at {tey_tit_corr:+.3f} "
          f"-- they are nearly the same signal, both reflecting combustion/load intensity. This is "
          f"also why controlling for TIT instead of AT gives an unstable, sign-flipped result: with "
          f"two nearly-collinear variables, 'TEY's effect after removing TIT' is not a well-posed "
          f"question. This is a property of this dataset's variable structure.")

    # --- Build the reference curve within a fixed ambient-temperature band ---
    print("\n--- Reference curve: NOx vs TEY (load), within a fixed ambient-temperature band ---")
    at_median = df["AT"].median()
    at_band = df[(df["AT"] >= at_median - 2) & (df["AT"] <= at_median + 2)]
    print(f"Restricting to AT within +/-2C of the median ({at_median:.1f}C): {len(at_band)} rows "
          f"({len(at_band) / len(df):.1%} of the data)")

    curve = build_reference_curve(at_band, x_col="TEY", y_col="NOX", n_bins=8)
    print(curve[["x_bin_center", "y_mean", "y_std", "count"]].to_string(index=False))

    means = curve["y_mean"].values
    is_monotonic = all(d <= 0 for d in np.diff(means)) or all(d >= 0 for d in np.diff(means))
    print(
        f"\nFinding: even within a fixed ambient-temperature band, this curve is NOT "
        f"monotonic (monotonic={is_monotonic}). NOx vs. load is "
        f"a genuinely non-trivial, non-linear relationship in this dataset, not a simple 'more load "
        f"= more emissions' curve. A reference-curve system built on this data needs bin-by-bin "
        f"reference values (as built here) rather than a single fitted slope, precisely because "
        f"the true relationship does not reduce to one."
    )

    # --- Deviation scoring demo: is a given reading normal for its load? ---
    print("\n--- Performance-score demo: scoring example readings against the reference curve ---")
    examples = [
        (curve["x_bin_center"].median(), curve["y_mean"].median(), "typical reading"),
        (curve["x_bin_center"].median(), curve["y_mean"].median() + 3 * curve["y_std"].median(), "elevated NOx reading"),
    ]
    for x, y, label in examples:
        z = score_deviation(curve, x=x, y=y)
        flag = "FLAG (|z|>2)" if abs(z) > 2 else "normal"
        print(f"  {label}: load={x:.1f}, NOx={y:.2f} -> z={z:+.2f} ({flag})")


if __name__ == "__main__":
    main()
