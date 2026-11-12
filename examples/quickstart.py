"""Forecast one drifter event and evaluate the 16 search plans under the forecast.

Run from the repository root: python examples/quickstart.py
"""
import json
from pathlib import Path

import numpy as np

from pevagen.decision_audit.search import localize_endpoints, static_plan_utilities
from pevagen.drift_audit.baselines import TrainOnlyBaselines
from pevagen.task_prediction.plans import build_library

ROOT = Path(__file__).resolve().parents[1]
env = json.loads((ROOT / "configs/search_environment.json").read_text(encoding="utf-8"))
forecast = json.loads((ROOT / "configs/forecast_models.json").read_text(encoding="utf-8"))
train = dict(np.load(ROOT / "data/gdp/train_north_atlantic.npz"))
test = dict(np.load(ROOT / "data/gdp/test_south_atlantic.npz"))

event = 0
query = {k: test[k][event : event + 1] for k in ("features", "kinematic_km")}
draws = TrainOnlyBaselines(train, forecast).predict(query, seed=1)["analog_empirical"]

# Search square of 40 km centred on the kinematic endpoint
local = localize_endpoints(draws, query["kinematic_km"], env["environment"]["region_size_km"])[0]
library = build_library(env)
utilities = static_plan_utilities(library, local, env)  # scenarios x plans
values = utilities.mean(axis=0)

best = int(np.argmax(values))
print(f"event {event}: {len(local)} forecast scenarios, {len(library.names)} plans")
for name, value in sorted(zip(library.names, values), key=lambda x: -x[1])[:5]:
    print(f"  {name:<20} {value:.4f}")
observed = localize_endpoints(test["targets_km"][event : event + 1], query["kinematic_km"],
                              env["environment"]["region_size_km"])
print(f"selected {library.names[best]}; utility at the observed endpoint "
      f"{static_plan_utilities(library, observed, env)[0, best]:.4f}")
