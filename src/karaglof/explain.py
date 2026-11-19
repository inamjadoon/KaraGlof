"""Plain-language explanations of Tier 1 risk scores (notebook 04 -> shap_top.csv).

SHAP splits each lake-year's score into one contribution per feature. The web
app shows the 5 largest as sentences, e.g. "Last summer was +1.2 °C compared
with the 1980-1999 normal: raises susceptibility". Wording follows the project
rule: "susceptibility", never "predicts floods".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEMPLATES = {
    "area_km2_prev": "Lake area last year: {v:.3f} km²",
    "area_growth_3y_pct": "Lake area changed {v:+.0f}% over the 3 years before",
    "area_trend_km2_per_yr": "Lake area trend over the last 5 years: {v:+.4f} km²/year",
    "velocity_prev": "Nearby glacier ice moved {v:.0f} m/year last year",
    "velocity_change": "Nearby glacier speed changed {v:+.0f} m/year from the year before",
    "surge_ratio_prev": "Nearby glacier moved {v:.1f}× its usual speed last year",
    "surge_flag_prev": None,                                 # yes/no sentence, see driver_text
    "pdd_prev_summer": "Last summer had {v:.0f} positive degree-days (melt energy)",
    "heat_anomaly_prev": "Last summer was {v:+.1f} °C compared with the 1980–1999 normal",
    "precip_prev_summer_mm": "Last summer's precipitation: {v:.0f} mm",
    "snowmelt_prev_summer_mm": "Last summer's snowmelt: {v:.0f} mm",
    "elev_m": "Lake elevation: {v:.0f} m",
    "slope_around_deg": "Mean slope around the lake: {v:.0f}°",
    "dist_to_glacier_m": "Distance to the nearest glacier: {v:.0f} m",
    "parent_glacier_km2": "Parent glacier area: {v:.1f} km²",
}


def driver_text(feature: str, value, shap: float) -> str:
    """One sentence for one driver, with the direction of its effect."""
    effect = "raises susceptibility" if shap > 0 else "lowers susceptibility"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{feature.replace('_', ' ')}: no data for this year ({effect})"
    if feature == "surge_flag_prev":
        text = ("The nearby glacier was surging last year" if value >= 1
                else "No glacier surge last year")
    else:
        text = TEMPLATES[feature].format(v=value)
    return f"{text}: {effect}"


def top_drivers(shap_values: pd.DataFrame, X: pd.DataFrame, keys: pd.DataFrame,
                n: int = 5) -> pd.DataFrame:
    """shap_top.csv rows (SPECS §3): lake_id, year, feature, value, shap, text.

    `shap_values` and `X` share their index and feature columns; `keys` holds
    lake_id and year for the same index. The `n` largest contributions by size
    are kept per lake-year, positive or negative.
    """
    long = shap_values.stack().rename("shap").reset_index()
    long.columns = ["row", "feature", "shap"]
    long["value"] = [X.at[r, f] for r, f in zip(long["row"], long["feature"], strict=True)]
    long["size"] = long["shap"].abs()
    long = long.sort_values(["row", "size"], ascending=[True, False]).groupby("row").head(n)
    long["text"] = [driver_text(f, v, s) for f, v, s in
                    zip(long["feature"], long["value"], long["shap"], strict=True)]
    long = long.join(keys[["lake_id", "year"]], on="row")
    long["value"] = long["value"].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    long["shap"] = long["shap"].round(4)
    return long[["lake_id", "year", "feature", "value", "shap", "text"]].reset_index(drop=True)
