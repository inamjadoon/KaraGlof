"""Tier 2 weekly warning-sign rules (alerts.csv).

Transparent rules, not a trained model — each Tier 2 lake has only a handful
of events. Starting thresholds (SPECS §5) are tuned once on 2015-2021 and
then frozen; they are never adjusted after looking at the Shisper 2022 replay.

Levels:
    green — none of the amber conditions hold;
    amber — area grew >15% in 2 weeks, or area above the lake's 90th
        percentile and 7-day temperature anomaly above +2 deg C;
    red   — amber conditions hold two weeks in a row, or area above the
        95th percentile with a heat anomaly above +3 deg C.

This module is deliberately empty in v0.1; it is filled in Phase 3.
"""
