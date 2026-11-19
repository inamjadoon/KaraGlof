"""Research metrics for KaraGLOF (notebook 04).

The only metrics ever reported (accuracy is never reported: with ~0.1% positives
a model that says "no flood" everywhere would be 99.9% "accurate"):
    - PR-AUC (main metric; the no-skill value is the share of positives);
    - recall at top-10 per year: share of outbursts whose lake was in that
      year's 10 highest-ranked lakes;
    - Brier score (mean squared error of the risk score; lower is better);
    - 95% confidence intervals by bootstrap **over lakes** (1,000 resamples):
      whole lakes are resampled, because rows of the same lake are not
      independent.

Plus two honesty checks for few events: the rank of every test outburst's
lake in its year (`event_ranks`) and paired bootstrap differences between two
models scored on the same rows (`paired_delta_ci`, used for the ablation).

Everything is computed on the time-based split only:
train 2000-2017, validation 2018-2019, test 2020-2025, scored once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

SPLITS = {"train": (2000, 2017), "validation": (2018, 2019), "test": (2020, 2025)}
N_BOOT = 1000


def split_of(years: pd.Series) -> pd.Series:
    """'train' / 'validation' / 'test' for each year (time-based, never shuffled)."""
    out = pd.Series(pd.NA, index=years.index, dtype="object")
    for name, (a, b) in SPLITS.items():
        out[years.between(a, b)] = name
    return out


def pr_auc(y, score) -> float:
    """Area under the precision-recall curve (average precision). Empty if no positives."""
    y = np.asarray(y)
    return float(average_precision_score(y, score)) if y.sum() > 0 else np.nan


def brier(y, score) -> float:
    return float(brier_score_loss(np.asarray(y), np.clip(np.asarray(score, float), 0, 1)))


def recall_at_k(df: pd.DataFrame, score_col: str, k: int = 10) -> float:
    """Share of positive lake-years whose lake ranked in the top `k` of its year."""
    ranks = df.groupby("year")[score_col].rank(ascending=False, method="first")
    pos = df["glof"] == 1
    return float((ranks[pos] <= k).mean()) if pos.any() else np.nan


def all_metrics(df: pd.DataFrame, score_col: str, k: int = 10) -> dict:
    return {"pr_auc": pr_auc(df["glof"], df[score_col]),
            f"recall_at_top{k}": recall_at_k(df, score_col, k),
            "brier": brier(df["glof"], df[score_col])}


def _lake_rows(df: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each lake, computed once per bootstrap."""
    return list(df.reset_index(drop=True).groupby("lake_id").indices.values())


def _resample_lakes(df: pd.DataFrame, groups: list[np.ndarray], rng) -> pd.DataFrame:
    """Draw whole lakes with replacement. A lake drawn twice appears twice,
    as two separate members of each year's ranking."""
    pick = rng.integers(0, len(groups), len(groups))
    return df.iloc[np.concatenate([groups[i] for i in pick])]


def bootstrap_ci(df: pd.DataFrame, score_col: str, k: int = 10, n: int = N_BOOT,
                 seed: int = 42) -> dict:
    """95% CI of every metric by resampling whole lakes. Resamples without any
    positive are skipped for PR-AUC and recall (they are undefined there); the
    number of usable resamples is returned."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = _lake_rows(df)
    draws = [all_metrics(_resample_lakes(df, groups, rng), score_col, k) for _ in range(n)]
    out = {}
    for m in draws[0]:
        vals = np.array([d[m] for d in draws], float)
        vals = vals[~np.isnan(vals)]
        out[f"{m}_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out[f"{m}_n_boot"] = int(len(vals))
    return out


def paired_delta_ci(df: pd.DataFrame, score_a: str, score_b: str, n: int = N_BOOT,
                    seed: int = 42) -> dict:
    """PR-AUC(a) - PR-AUC(b) on the same rows, with a 95% CI from the same lake
    resamples for both models (paired), as used for the ablation."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = _lake_rows(df)
    deltas = []
    for _ in range(n):
        s = _resample_lakes(df, groups, rng)
        if s["glof"].sum() == 0:
            continue
        deltas.append(pr_auc(s["glof"], s[score_a]) - pr_auc(s["glof"], s[score_b]))
    d = np.array(deltas)
    return {"delta_pr_auc": pr_auc(df["glof"], df[score_a]) - pr_auc(df["glof"], df[score_b]),
            "delta_pr_auc_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_boot": int(len(d))}


def event_ranks(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """For every positive lake-year: its rank among all lakes that year (1 = highest)."""
    ranks = df.groupby("year")[score_col].rank(ascending=False, method="first")
    n = df.groupby("year")["lake_id"].transform("size")
    out = df.assign(rank=ranks.astype(int), lakes_that_year=n)
    return out.loc[out["glof"] == 1, ["lake_id", "year", score_col, "rank", "lakes_that_year"]]
