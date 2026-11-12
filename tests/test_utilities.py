import json
from pathlib import Path

import numpy as np

from pevagen.decision_audit.benchmark import (
    dynamic_plan_utilities,
    interpolate_terminal_truth,
    terminal_linear_paths,
)
from pevagen.decision_audit.search import localize_endpoints, static_plan_utilities
from pevagen.drift_audit.baselines import TrainOnlyBaselines
from pevagen.task_prediction.plans import build_library, score_tracks

ROOT = Path(__file__).resolve().parents[1]


def config(name="search_environment"):
    return json.loads((ROOT / f"configs/{name}.json").read_text(encoding="utf-8"))


def test_library_has_sixteen_plans():
    library = build_library(config())
    assert len(library.names) == 16
    assert library.actions.shape[:2] == (16, config()["environment"]["episode_steps"])


def test_static_endpoint_score_matches_original_track_integrator():
    cfg = config()
    library = build_library(cfg)
    endpoints = np.asarray([[20.0, 20.0], [12.5, 28.0], [-1.0, 41.0]])
    got = static_plan_utilities(library, endpoints, cfg, chunk_size=2)
    tracks = np.repeat(endpoints[:, None, :], cfg["environment"]["episode_steps"] + 1, axis=1)
    expected = score_tracks(library, tracks, cfg)["return"]
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_repeated_dynamic_path_matches_static_endpoint_evaluator():
    cfg = config()
    library = build_library(cfg)
    endpoints = np.array([[20.0, 20.0], [21.0, 19.0]])
    paths = np.repeat(endpoints[:, None, :], cfg["environment"]["episode_steps"], axis=1)
    np.testing.assert_allclose(
        dynamic_plan_utilities(library, paths, cfg),
        static_plan_utilities(library, endpoints, cfg),
        atol=1e-12,
    )


def test_terminal_paths_end_at_endpoint_and_truth_segment():
    endpoints = np.array([[5.0, 6.0], [2.0, 4.0]])
    velocity = np.array([[1.0, -2.0], [0.5, 0.25]])
    paths = terminal_linear_paths(endpoints, velocity, 3, 3600)
    np.testing.assert_allclose(paths[:, -1], endpoints)
    np.testing.assert_allclose(paths[:, 0], endpoints - 2 * velocity)
    truth = interpolate_terminal_truth(np.zeros((2, 2)), np.full((2, 2), 6.0), 2, 3600)
    np.testing.assert_allclose(truth[:, 0], 5.0)
    np.testing.assert_allclose(truth[:, 1], 6.0)


def test_local_mapping_centers_kinematic_forecast():
    kinematic = np.asarray([[3.0, -2.0], [7.0, 8.0]])
    draws = np.asarray([[[3.0, -2.0], [4.0, 0.0]], [[7.0, 8.0], [6.0, 7.0]]])
    got = localize_endpoints(draws, kinematic, 40.0)
    np.testing.assert_array_equal(got[:, 0], np.full((2, 2), 20.0))


def test_forecasts_use_training_residuals_only():
    train = dict(np.load(ROOT / "data/gdp/train_north_atlantic.npz"))
    test = dict(np.load(ROOT / "data/gdp/test_south_atlantic.npz"))
    query = {k: test[k][:3] for k in ("features", "kinematic_km")}
    draws = TrainOnlyBaselines(train, config("forecast_models")).predict(query, seed=1)
    residuals = train["targets_km"] - train["kinematic_km"]
    shifted = draws["global_empirical"] - query["kinematic_km"][:, None, :]
    assert all(any(np.allclose(r, x) for x in residuals) for r in shifted.reshape(-1, 2))
    assert draws["analog_empirical"].shape == (3, config("forecast_models")["samples"], 2)
