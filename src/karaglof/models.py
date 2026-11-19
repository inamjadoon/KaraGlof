"""Tier 1 models (notebook 04): logistic baseline -> Random Forest -> LightGBM.

Rules (CLAUDE.md): time-based split only; class imbalance handled with class
weights (`class_weight`, `scale_pos_weight`), never synthetic samples (SMOTE);
seed 42. With ~10 positives, tuning is kept deliberately tiny: a handful of
LightGBM settings, chosen on the validation years, simplest first so that ties
go to the simpler model.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from karaglof import SEED
from karaglof.metrics import pr_auc

#: LightGBM settings tried on the validation years, simplest first.
LGBM_GRID = [
    {"num_leaves": nl, "min_child_samples": mcs, "n_estimators": ne}
    for nl in (3, 7) for mcs in (10, 3) for ne in (100, 300)
]


def pos_weight(y) -> float:
    """Negatives per positive: how much more a missed outburst counts."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def logistic():
    """Baseline: median-filled, standardised features, balanced class weights."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000,
                                            random_state=SEED))


def random_forest():
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED))


def lightgbm(scale_pos_weight: float, num_leaves=3, min_child_samples=10, n_estimators=100):
    """LightGBM reads missing values natively, so nothing is filled in."""
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, n_estimators=n_estimators,
        num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=SEED, deterministic=True,
        force_row_wise=True, verbose=-1)


def score(model, rows: pd.DataFrame, features) -> np.ndarray:
    return model.predict_proba(rows[features])[:, 1]


def tune_lightgbm(train: pd.DataFrame, val: pd.DataFrame, features) -> tuple[dict, pd.DataFrame]:
    """Try LGBM_GRID: fit on train, PR-AUC on validation. Returns the best
    settings (first one wins ties, i.e. the simplest) and the full table."""
    rows = []
    for params in LGBM_GRID:
        m = lightgbm(pos_weight(train["glof"]), **params).fit(train[features], train["glof"])
        rows.append({**params, "val_pr_auc": pr_auc(val["glof"], score(m, val, features))})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_pr_auc"].fillna(-1).idxmax()]
    return {k: int(best[k]) for k in LGBM_GRID[0]}, table


def leave_one_lake_out(data: pd.DataFrame, lake_id: str, features, params: dict,
                       fit_years: tuple[int, int], score_years: tuple[int, int]) -> pd.DataFrame:
    """Train LightGBM on `fit_years` **without** `lake_id`, then score every lake
    in `score_years`. If the left-out lake still ranks high in its outburst
    years, the model learned something general, not the lake itself."""
    fit = data[data["year"].between(*fit_years) & (data["lake_id"] != lake_id)]
    m = lightgbm(pos_weight(fit["glof"]), **params).fit(fit[features], fit["glof"])
    rows = data[data["year"].between(*score_years)].copy()
    rows["score"] = score(m, rows, features)
    return rows
