"""Dynamic terminal-window utilities for the drift-to-search benchmark."""

from __future__ import annotations

import numpy as np


def dynamic_plan_utilities(library, target_paths: np.ndarray, config, chunk_size: int = 128):
    """Evaluate one 2-D target position per search step against every fixed UAV plan."""
    paths = np.asarray(target_paths, dtype=np.float32)
    if paths.ndim != 3 or paths.shape[-1] != 2 or not np.isfinite(paths).all():
        raise ValueError("finite [path, step, xy] target paths required")
    if chunk_size < 1:
        raise ValueError("positive chunk size required")
    env = config["environment"]
    steps = int(env["episode_steps"])
    if paths.shape[1] != steps:
        raise ValueError("target path length must equal the environment episode steps")
    time = (np.arange(steps) + 1) * env["dt_seconds"]
    phase = 2 * np.pi * np.arange(steps) * env["dt_seconds"] / (3 * 3600)
    survival = np.exp(-time / (env["survival_tau_hours"] * 3600))
    weather = (1 - 0.6 * env["fog_level"]) * (1 - 0.35 * env["sea_state"])
    radius = env["sensing_radius_km"]
    per_step_energy = (
        env["energy_penalty"] * np.linalg.norm(library.actions, axis=-1).sum(axis=-1)
    )
    result = np.empty((len(paths), len(library.names)), dtype=float)
    for plan, positions in enumerate(library.positions):
        for start in range(0, len(paths), chunk_size):
            stop = min(start + chunk_size, len(paths))
            batch = paths[start:stop]
            spatial = 0.82 + 0.18 * np.sin(
                2 * np.pi * batch[:, :, 0] / env["region_size_km"] + phase
            ) * np.cos(2 * np.pi * batch[:, :, 1] / env["region_size_km"] - phase)
            rho = np.clip(env["base_detection_probability"] * weather * spatial, 0.02, 0.99)
            distances = np.linalg.norm(batch[:, :, None, :] - positions[None, :, :, :], axis=-1)
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


def terminal_linear_paths(
    endpoints: np.ndarray,
    hourly_velocity: np.ndarray,
    steps: int,
    dt_seconds: float,
) -> np.ndarray:
    """Backcast a common legal constant-velocity segment ending at each 24 h endpoint."""
    endpoints = np.asarray(endpoints, dtype=float)
    velocity = np.asarray(hourly_velocity, dtype=float)
    if endpoints.ndim not in (2, 3) or endpoints.shape[-1] != 2:
        raise ValueError("endpoint array must end in xy")
    if velocity.shape != (endpoints.shape[0], 2):
        raise ValueError("one hourly velocity is required per event")
    remaining_hours = (steps - 1 - np.arange(steps)) * dt_seconds / 3600
    if endpoints.ndim == 2:
        return endpoints[:, None, :] - remaining_hours[None, :, None] * velocity[:, None, :]
    return endpoints[:, :, None, :] - (
        remaining_hours[None, None, :, None] * velocity[:, None, None, :]
    )


def interpolate_terminal_truth(
    position_18h: np.ndarray,
    position_24h: np.ndarray,
    steps: int,
    dt_seconds: float,
) -> np.ndarray:
    """Interpolate the observed 18--24 h segment over the final two search hours."""
    start, end = np.asarray(position_18h, float), np.asarray(position_24h, float)
    if start.shape != end.shape or start.ndim < 2 or start.shape[-1] != 2:
        raise ValueError("paired [..., xy] positions required")
    elapsed_hours = 4 + (np.arange(steps) + 1) * dt_seconds / 3600
    if elapsed_hours[-1] > 6 + 1e-12:
        raise ValueError("search window extends beyond the observed 18--24 h segment")
    fraction = elapsed_hours / 6
    shape = (1,) * (start.ndim - 1) + (steps, 1)
    return start[..., None, :] + fraction.reshape(shape) * (end - start)[..., None, :]
