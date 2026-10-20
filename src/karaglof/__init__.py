"""KaraGLOF shared package.

Python code shared by the Colab notebooks (01-05). Logic lives here, not
inline in notebooks, so it can be tested and imported consistently.

Modules:
    features — Tier 1 feature engineering (one row per lake per year) and
        the leakage guard: features for year Y may only use data up to
        31 December of year Y-1.
    rules    — Tier 2 weekly warning-sign rules. Thresholds are tuned once
        on 2015-2021 and frozen before the Shisper 2022 replay.
    metrics  — Research metrics only: PR-AUC, recall at top-10, Brier score,
        bootstrap 95% CIs. Accuracy is never reported.
"""

SEED = 42

__all__ = ["SEED"]
__version__ = "0.1.0"
