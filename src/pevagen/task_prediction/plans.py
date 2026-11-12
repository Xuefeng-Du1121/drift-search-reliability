"""Public geometric plan library and exact first-hit objective for fixed plans."""

from dataclasses import dataclass

import numpy as np

from pevagen.task_prediction.plan_geometry import cruise_actions, reflect, starting_positions


@dataclass(frozen=True)
class PlanLibrary:
    names: tuple
    actions: np.ndarray  # plan x step x agent x xy
    positions: np.ndarray  # post-move sensing positions, same shape


def _scan_waypoints(width, center, angle, env):
    agents = env["n_agents"]
    strip = width / agents
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotation[np.abs(rotation) < 1e-12] = 0
    paths = []
    for agent in range(agents):
        xs = np.arange(-width / 2 + agent * strip + 0.5, -width / 2 + (agent + 1) * strip, 1.0)
        if not len(xs):
            raise ValueError("plan window is too narrow for the declared strip spacing")
        points = np.asarray(
            [
                p
                for j, x in enumerate(xs)
                for p in (
                    (x, -width / 2 if j % 2 == 0 else width / 2),
                    (x, width / 2 if j % 2 == 0 else -width / 2),
                )
            ]
        )
        paths.append(points @ rotation.T + center)
    return paths


def build_library(config):
    """No cases, histories or outcomes enter library generation."""
    env = config["environment"]
    center = np.full(2, env["region_size_km"] / 2)
    specs = []
    for scale, label in ((1.0, "full"), (0.5, "central")):
        for turn in range(4):
            specs.append(
                (
                    f"{label}_rotation_{90 * turn}",
                    scale * env["region_size_km"],
                    center,
                    turn * np.pi / 2,
                )
            )
    for direction in range(8):
        angle = direction * np.pi / 4
        offset = config["plan_library"]["focus_offset_km"] * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        specs.append(
            (
                f"focus_{45 * direction}",
                config["plan_library"]["focus_width_km"],
                center + offset,
                0.0,
            )
        )
    action_set = cruise_actions(config)
    step = env["max_speed_mps"] * env["dt_seconds"] / 1000
    all_actions, all_positions = [], []
    for _, width, origin, angle in specs:
        paths = _scan_waypoints(width, origin, angle, env)
        if any(np.any((p < -1e-7) | (p > env["region_size_km"] + 1e-7)) for p in paths):
            raise ValueError("waypoints outside allowed search square")
        position = starting_positions(env)
        indices = np.zeros(env["n_agents"], dtype=int)
        actions, positions = [], []
        for _ in range(env["episode_steps"]):
            joint = []
            for agent, path in enumerate(paths):
                if np.linalg.norm(path[indices[agent]] - position[agent]) <= step:
                    indices[agent] = (indices[agent] + 1) % len(path)
                direction = path[indices[agent]] - position[agent]
                joint.append(action_set[np.argmax(action_set @ direction)])
            joint = np.asarray(joint)
            position = reflect(position + step * joint, env["region_size_km"])
            actions.append(joint.copy())
            positions.append(position.copy())
        all_actions.append(actions)
        all_positions.append(positions)
    return PlanLibrary(
        tuple(s[0] for s in specs), np.asarray(all_actions), np.asarray(all_positions)
    )


def tracks_from_cases(cases, config):
    return np.asarray(
        [
            np.linspace(
                c.hidden_search_start_world,
                c.hidden_target_world,
                config["environment"]["episode_steps"] + 1,
            )
            for c in cases
        ],
        dtype=np.float32,
    )


def hazards(library, tracks, config):
    """Return plan x target trajectory x time probabilities of any-agent detection."""
    env = config["environment"]
    targets = np.asarray(tracks, dtype=np.float32)
    steps = env["episode_steps"]
    if targets.ndim != 3 or targets.shape[1:] != (steps + 1, 2) or not np.isfinite(targets).all():
        raise ValueError("finite complete target tracks required")
    result = np.empty((len(library.names), len(targets), steps), dtype=float)
    phase = 2 * np.pi * np.arange(steps) * env["dt_seconds"] / (3 * 3600)
    points = targets[:, 1:]
    spatial = 0.82 + 0.18 * np.sin(
        2 * np.pi * points[..., 0] / env["region_size_km"] + phase
    ) * np.cos(2 * np.pi * points[..., 1] / env["region_size_km"] - phase)
    weather = (1 - 0.6 * env["fog_level"]) * (1 - 0.35 * env["sea_state"])
    rho = np.clip(env["base_detection_probability"] * weather * spatial, 0.02, 0.99)
    radius = env["sensing_radius_km"]
    for i, positions in enumerate(library.positions):
        distances = np.linalg.norm(points[:, :, None] - positions[None], axis=-1)
        p = np.where(
            distances <= radius,
            rho[..., None] * np.exp(-0.5 * (distances / (0.7 * radius)) ** 2),
            0,
        )
        result[i] = 1 - np.prod(1 - p, axis=-1)
    return result


def integrate_hazards(chances, actions, config):
    """Integrate original first-detection reward, stopping energy and restricted TTD."""
    env = config["environment"]
    h = np.asarray(chances, dtype=float)
    if h.ndim != 3 or not np.isfinite(h).all() or np.any((h < 0) | (h > 1)):
        raise ValueError("valid plan x target x time hazard array required")
    before = np.concatenate(
        [np.ones((*h.shape[:2], 1)), np.cumprod(1 - h[..., :-1], axis=-1)], axis=-1
    )
    first = before * h
    time = (np.arange(h.shape[-1]) + 1) * env["dt_seconds"]
    survival = np.exp(-time / (env["survival_tau_hours"] * 3600))
    reward = env["detection_reward"] * np.sum(first * survival, axis=-1)
    per_step_energy = env["energy_penalty"] * np.linalg.norm(actions, axis=-1).sum(axis=-1)
    energy = np.sum(before * per_step_energy[:, None], axis=-1)
    success = first.sum(axis=-1)
    ttd = (first * time).sum(axis=-1) + np.prod(1 - h, axis=-1) * time[-1]
    return {
        "return": (reward - energy).T,
        "success": success.T,
        "survival_reward": reward.T,
        "energy": energy.T,
        "restricted_ttd": ttd.T,
        "geometric_opportunity": np.any(h > 0, axis=-1).T.astype(float),
    }


def score_tracks(library, tracks, config):
    return integrate_hazards(hazards(library, tracks, config), library.actions, config)
