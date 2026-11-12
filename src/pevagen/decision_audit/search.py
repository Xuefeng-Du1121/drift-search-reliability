"""Exact endpoint-search utility under the frozen multi-UAV plan library."""

from __future__ import annotations

import numpy as np


def static_plan_utilities(library, endpoints: np.ndarray, config, chunk_size: int = 256):
    """Return endpoint x plan utilities, exactly matching a static repeated track."""
    # The original task integrator explicitly casts target tracks to float32.
    points = np.asarray(endpoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("finite endpoint matrix required")
    if chunk_size < 1:
        raise ValueError("positive chunk size required")
    env = config["environment"]
    steps = env["episode_steps"]
    time = (np.arange(steps) + 1) * env["dt_seconds"]
    phase = 2 * np.pi * np.arange(steps) * env["dt_seconds"] / (3 * 3600)
    survival = np.exp(-time / (env["survival_tau_hours"] * 3600))
    weather = (1 - 0.6 * env["fog_level"]) * (1 - 0.35 * env["sea_state"])
    radius = env["sensing_radius_km"]
    per_step_energy = (
        env["energy_penalty"] * np.linalg.norm(library.actions, axis=-1).sum(axis=-1)
    )
    result = np.empty((len(points), len(library.names)), dtype=float)
    for plan, positions in enumerate(library.positions):
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            batch = points[start:stop]
            spatial = 0.82 + 0.18 * np.sin(
                2 * np.pi * batch[:, None, 0] / env["region_size_km"] + phase
            ) * np.cos(2 * np.pi * batch[:, None, 1] / env["region_size_km"] - phase)
            rho = np.clip(env["base_detection_probability"] * weather * spatial, 0.02, 0.99)
            distances = np.linalg.norm(
                batch[:, None, None, :] - positions[None, :, :, :], axis=-1
            )
            sensor = np.where(
                distances <= radius,
                rho[:, :, None] * np.exp(-0.5 * (distances / (0.7 * radius)) ** 2),
                0.0,
            )
            hazard = 1 - np.prod(1 - sensor, axis=-1)
            before = np.concatenate(
                [np.ones((len(batch), 1)), np.cumprod(1 - hazard[:, :-1], axis=1)], axis=1
            )
            first = before * hazard
            reward = env["detection_reward"] * np.sum(first * survival, axis=1)
            energy = np.sum(before * per_step_energy[plan][None, :], axis=1)
            result[start:stop, plan] = reward - energy
    return result


def localize_endpoints(values: np.ndarray, kinematic: np.ndarray, region_size_km: float):
    values = np.asarray(values, dtype=float)
    kinematic = np.asarray(kinematic, dtype=float)
    if values.shape[-1] != 2 or kinematic.shape != (values.shape[0], 2):
        raise ValueError("forecast and kinematic endpoint dimensions disagree")
    center = region_size_km / 2
    if values.ndim == 3:
        return values - kinematic[:, None, :] + center
    return values - kinematic + center
