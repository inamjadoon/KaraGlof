"""Tests for the Tier 1 models (karaglof.models) and explanations (karaglof.explain)."""

import numpy as np
import pandas as pd

from karaglof import explain, models
from karaglof.features import FEATURES


def data(seed=0):
    """Small lake-year table where 'velocity_prev' carries the signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(60):
        for year in range(2000, 2026):
            row = {"lake_id": f"L{i}", "year": year}
            row.update({f: float(rng.normal()) for f in FEATURES})
            hot = i < 3 and year % 4 == 0
            row["velocity_prev"] = 5.0 + rng.normal() if hot else float(rng.normal())
            row["glof"] = int(hot)
            rows.append(row)
    return pd.DataFrame(rows)


def test_pos_weight_counts_negatives_per_positive():
    assert models.pos_weight([0, 0, 0, 1]) == 3.0


def test_models_learn_the_signal_and_tuning_picks_from_the_grid():
    d = data()
    train, val = d[d.year <= 2017], d[d.year.between(2018, 2019)]
    best, table = models.tune_lightgbm(train, d[d.year >= 2018], FEATURES)
    assert best in models.LGBM_GRID and len(table) == len(models.LGBM_GRID)
    for make in (models.logistic, models.random_forest,
                 lambda: models.lightgbm(models.pos_weight(train.glof), **best)):
        m = make().fit(train[FEATURES], train.glof)
        test = d[d.year >= 2020]
        s = models.score(m, test, FEATURES)
        assert s[test.glof.values == 1].mean() > s[test.glof.values == 0].mean()
    assert len(val)


def test_leave_one_lake_out_never_trains_on_that_lake():
    d = data()
    out = models.leave_one_lake_out(d, "L0", FEATURES, models.LGBM_GRID[0],
                                    (2000, 2019), (2020, 2025))
    assert set(out.year) == set(range(2020, 2026)) and "L0" in set(out.lake_id)
    # L0 only appears in the scored rows, never in training: the other hot lakes teach the signal
    l0 = out[(out.lake_id == "L0") & (out.glof == 1)]
    assert (l0.score > out[out.glof == 0].score.median()).all()


def test_every_feature_has_a_sentence_and_direction():
    for f in FEATURES:
        text = explain.driver_text(f, 1.0, 0.2)
        assert text.endswith("raises susceptibility")
    assert "no data" in explain.driver_text("velocity_prev", float("nan"), -0.1)
    surging = explain.driver_text("surge_flag_prev", 1.0, 0.3)
    assert surging.startswith("The nearby glacier was surging")
    for f in FEATURES:
        assert "predict" not in explain.driver_text(f, 1.0, 0.1).lower()


def test_top_drivers_keeps_the_largest_by_size():
    X = pd.DataFrame({f: [1.0, 2.0] for f in FEATURES})
    sv = pd.DataFrame({f: [0.0, 0.0] for f in FEATURES})
    sv.loc[0, "heat_anomaly_prev"] = -0.9
    sv.loc[0, "velocity_prev"] = 0.5
    keys = pd.DataFrame({"lake_id": ["A", "B"], "year": [2022, 2022]})
    t = explain.top_drivers(sv, X, keys, n=2)
    a = t[t.lake_id == "A"]
    assert a.feature.tolist() == ["heat_anomaly_prev", "velocity_prev"]
    assert list(t.columns) == ["lake_id", "year", "feature", "value", "shap", "text"]
