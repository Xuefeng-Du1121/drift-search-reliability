"""NumPy-only geometry shared by the fixed-plan benchmark.

These formulas retain the original planning_reference and learner conventions,
without importing their unrelated PyTorch training stack.
"""
import numpy as np


def cruise_actions(config):
    radians = np.deg2rad(config['action_headings_degrees'])
    vectors = np.c_[np.cos(radians), np.sin(radians)].astype(np.float32)
    vectors[np.abs(vectors) < 1e-7] = 0
    return vectors


def reflect(positions, width):
    wrapped = np.mod(positions, 2 * width)
    return np.where(wrapped <= width, wrapped, 2 * width - wrapped).astype(np.float32)


def starting_positions(config):
    width, agents = config['region_size_km'], config['n_agents']
    angles = np.linspace(0, 2 * np.pi, agents, endpoint=False)
    return (width / 2 + min(0.8, 0.04 * width) * np.c_[np.cos(angles), np.sin(angles)]).astype(np.float32)
