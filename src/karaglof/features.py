"""Tier 1 feature engineering for the lake-year table (lake_year.parquet).

Built by notebook 03 and frozen afterwards — no new features after the gate.

Planned contents:
    - column definitions and one-year-lag joins (features for year Y use only
      data up to 31 December of year Y-1);
    - a leakage guard that asserts no feature column was computed from data
      on or after the prediction date (1 January of year Y);
    - a missing-value report per column.

This module is deliberately empty in v0.1; it is filled in Phase 2.
"""
