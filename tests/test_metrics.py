"""Tests for the research metrics (karaglof.metrics)."""

import numpy as np
import pandas as pd
import pytest

from karaglof import metrics as km


def table():
    # 2 years x 20 lakes; lake L0 floods in both years: rank 1 in 2020,
    # rank 12 in 2021 (11 lakes score higher)
    rows = []
    for year in (2020, 2021):
        for i in range(20):
            score = 1.0 - i / 20
            if year == 2021 and i == 0:
                score = 1.0 - 11.5 / 20
            rows.append({"lake_id": f"L{i}", "year": year, "glof": int(i == 0), "score": score})
    return pd.DataFrame(rows)


def test_split_is_by_year_only():
    s = km.split_of(pd.Series([2000, 2017, 2018, 2019, 2020, 2025]))
    assert s.tolist() == ["train", "train", "validation", "validation", "test", "test"]


def test_recall_at_top_k_and_event_ranks():
    df = table()
    assert km.recall_at_k(df, "score", k=10) == 0.5          # top-1 in 2020, rank 12 in 2021
    ranks = km.event_ranks(df, "score").set_index("year")["rank"]
    assert ranks.loc[2020] == 1 and ranks.loc[2021] == 12


def test_pr_auc_perfect_and_no_positives():
    assert km.pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert np.isnan(km.pr_auc([0, 0], [0.1, 0.2]))


def test_bootstrap_ci_contains_point_estimate_and_skips_empty_resamples():
    df = table()
    m = km.all_metrics(df, "score")
    ci = km.bootstrap_ci(df, "score", n=200)
    lo, hi = ci["pr_auc_ci95"]
    assert lo <= m["pr_auc"] <= hi
    assert ci["pr_auc_n_boot"] < 200          # resamples without the flooding lake are skipped
    assert ci["brier_n_boot"] == 200


def test_paired_delta_is_zero_for_identical_models():
    df = table().assign(score_b=lambda d: d.score)
    out = km.paired_delta_ci(df, "score", "score_b", n=100)
    assert out["delta_pr_auc"] == 0 and out["delta_pr_auc_ci95"] == [0.0, 0.0]
