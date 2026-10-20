"""Research metrics for KaraGLOF.

The only metrics ever reported (accuracy is never reported):
    - PR-AUC (main metric);
    - recall at top-10 per year;
    - Brier score (with a reliability plot for calibration);
    - 95% confidence intervals by bootstrap over lakes (1,000 resamples).

Everything is computed on the time-based split only:
train 2000-2017, validation 2018-2019, test 2020-2025, scored once.

This module is deliberately empty in v0.1; it is filled in Phase 3.
"""
