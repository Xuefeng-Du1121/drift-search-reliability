# drift-search-reliability

Core models, environment data and configuration for the manuscript

> *Sampling reliability of forecast-driven environmental decisions: sampling scheme,
> certification feasibility and action-set reduction* (under review).

This repository contains the components needed to build the decision problem studied in
the paper: empirical 24-hour drifter displacement forecasts, a library of 16 multi-aircraft
search plans, and the deterministic detection utility that scores a plan against a target
position or path. The experiment drivers, experiment parameters, result files, manuscript
source and plotting code will be added after the review process is complete.

## Contents

| Path | Content |
|---|---|
| `src/pevagen/drift_audit/baselines.py` | `TrainOnlyBaselines`: global and analog (32 nearest neighbours) residual forecasts in empirical, Gaussian and two-component-mixture form, fitted on training drifters only |
| `src/pevagen/task_prediction/plans.py`, `plan_geometry.py` | `build_library`: the 16 open-loop plans for six aircraft (four full-square sweeps, four central sweeps, eight focus plans); `score_tracks`: reference integrator |
| `src/pevagen/decision_audit/search.py` | `static_plan_utilities`: utility of every plan for fixed target endpoints; `localize_endpoints`: map forecasts to the search square |
| `src/pevagen/decision_audit/benchmark.py` | `dynamic_plan_utilities`: utility for moving targets; `terminal_linear_paths`, `interpolate_terminal_truth`: target paths in the search window |
| `configs/search_environment.json` | Search square, aircraft, sensor, time step, survival discount and plan-library geometry |
| `configs/forecast_models.json` | Forecast settings: ensemble size, number of analogs, covariance floor, seed |
| `data/gdp/*.npz` | Drifter events derived from the NOAA Global Drifter Program (below) |
| `examples/quickstart.py` | Forecast one event and evaluate the 16 plans |
| `tests/` | Consistency tests of the utility evaluators |

## Data

Prepared from the six-hourly interpolated product of the
[NOAA Global Drifter Program](https://www.aoml.noaa.gov/phod/gdp/). One event is a drifter
with a 24-hour history before the origin and an observed position 24 hours later.

| File | Events | Region | Origins |
|---|---|---|---|
| `train_north_atlantic.npz` | 192 | western North Atlantic | 2010 |
| `test_south_indian.npz` | 94 | South Indian Ocean | January 2021 |
| `test_south_atlantic.npz` | 109 | South Atlantic | January 2021 |

Arrays (one row per event; positions in km relative to the origin, east and north):

| Key | Shape | Meaning |
|---|---|---|
| `features` | (n, 9) | last six-hour velocity (2), mean 24-hour velocity (2), last-minus-previous velocity (2), all in km/h; latitude (deg); sine and cosine of longitude |
| `history_km` | (n, 5, 2) | positions at −24, −18, −12, −6 and 0 h |
| `kinematic_km` | (n, 2) | kinematic 24-hour endpoint (last velocity × 24 h) |
| `targets_km` | (n, 2) | observed 24-hour displacement |
| `future_18h_24h_km` | (n, 2, 2) | observed positions at 18 and 24 h (test files) |
| `groups`, `regions`, `origin_utc_seconds`, `expno_proxy`, `availability_known` | (n,) | drifter identifier, region label, origin time, deployment proxy, metadata flag |

## Installation

```bash
pip install -e .
```

Tested with Python 3.12, NumPy 2.2, SciPy 1.15 and scikit-learn 1.7.

## Quick start

```bash
python examples/quickstart.py
pytest
```

## Licence and citation

Code under the MIT licence. The drifter data are a derived product of the NOAA Global Drifter
Program; please acknowledge the GDP when using them. A citation will be added on publication.
