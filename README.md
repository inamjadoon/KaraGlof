# KaraGLOF

A research-prototype web application for **Glacial Lake Outburst Flood (GLOF) susceptibility** in the
Hunza, Shimshal and Shigar basins of the Karakoram, Pakistan, covering 2000–2025.

> **Research prototype — not an official warning.** KaraGLOF ranks lake *susceptibility* and shows
> weekly *warning signs*. It never predicts floods and is not an official alert system.

It has two tiers:

- **Tier 1 — yearly susceptibility:** a LightGBM model ranks every mapped glacial lake each year,
  with SHAP explanations for the top drivers.
- **Tier 2 — weekly warning signs:** transparent rules (lake growth plus heat) are computed each week
  for three ice-dammed lakes: Shisper, Khurdopin and Passu.

## How it works

All satellite processing and modelling happens **once in Google Colab** (Earth Engine project
`glof-497313`). The deployed app reads only small released files, so hosting is free and fast.

1. **Colab notebooks** (`notebooks/01`–`05`): lakes and labels → lake area (yearly + weekly) →
   features (join + freeze) → Tier 1 model → Tier 2 rules. Each notebook reads the previous one's
   files from Google Drive (`Drive/KaraGlof/`) and must pass its checks before the next starts.
2. **Released files:** notebook outputs are copied into `api/data/`. That folder is the only data
   the web app reads (`ranking.csv`, `shap_top.csv`, `alerts.csv`, `metrics.json`, `lakes.geojson`).
3. **Web app:** a read-only FastAPI service (`api/`) serves the files over `/api/v1`, and a
   React + TypeScript + Leaflet single-page app (`web/`) displays them.

The full requirements are in [docs/PRD.md](docs/PRD.md) and the technical design in
[docs/SPECS.md](docs/SPECS.md) — read SPECS before writing any code.

## Repository layout

```
KaraGlof/
├── notebooks/            01–05 Colab notebooks (.ipynb)
├── src/karaglof/         shared Python code used by notebooks (features, rules, metrics)
├── data/                 raw/ interim/ processed/  (git-ignored except small CSVs)
├── api/
│   ├── app/              main.py, routers/, schemas.py, store.py
│   ├── data/             released output files the API serves
│   ├── tests/
│   └── Dockerfile
├── web/                  React app (src/pages, src/components, src/api)
├── reports/figures/      plots for the write-up and slides
├── docs/                 PRD.md, SPECS.md, write-up, data dictionary
├── .github/workflows/    ci.yml
├── README.md
└── LICENSE               MIT for code; CC BY 4.0 for data
```

## Setup (local)

Requires Python 3.11+ (and Node 20+ for the web app, once it exists).

```bash
pip install -r requirements.txt   # pinned versions (NFR-1)
pip install -e .                  # makes the karaglof package importable
```

Checks once code lands: `ruff check .` for linting, `pytest` for tests.

## Research integrity

The rules that make this research defensible (see AGENTS.md for the full list):

- Time-based split only: train 2000–2017, validate 2018–2019, test 2020–2025; the test set is
  scored once.
- No leakage: a Tier 1 feature for year Y uses only data up to 31 December of year Y−1, enforced
  by an automated test.
- Class imbalance is handled with class weights — never synthetic samples (SMOTE).
- Reported metrics are PR-AUC, recall at top-10, Brier score and bootstrap 95% CIs — never accuracy.
- No Sentinel-2 before 2017; Landsat 5/7/8/9 is used for 2000–2016.
- Tier 2 thresholds are tuned on 2015–2021 and frozen before the Shisper 2022 replay.

## Status

**v0.1 — scaffold only.** Planning (PRD + SPECS) is complete; notebooks, API and web app are not
written yet. See the milestone plan in PRD §7.

## Licence

Code under the [MIT License](LICENSE); derived datasets under CC BY 4.0. Every data source is
credited on the app's Data page.
