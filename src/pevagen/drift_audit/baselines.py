from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from sklearn.mixture import GaussianMixture

METHODS = (
    "global_gaussian",
    "global_mixture",
    "global_empirical",
    "analog_gaussian",
    "analog_mixture",
    "analog_empirical",
    "kinematic_quadrature",
)
SIMPLE_BASELINES = (
    "global_gaussian",
    "global_mixture",
    "analog_gaussian",
    "analog_mixture",
    "kinematic_quadrature",
)


class TrainOnlyBaselines:
    """Endpoint displacement baselines; fit and scaling never use held-out labels."""

    def __init__(self, train: dict, config: dict):
        self.config = config
        self.features = np.asarray(train["features"], dtype=np.float64)
        self.residuals = np.asarray(train["targets_km"]) - np.asarray(train["kinematic_km"])
        if len(self.features) < max(4, config["neighbors"]):
            raise ValueError("not enough training drifters for the fixed analog neighborhood")
        self.center = self.features.mean(axis=0)
        self.scale = np.maximum(self.features.std(axis=0), 1e-6)
        self.scaled = (self.features - self.center) / self.scale
        self.global_mixture = self._mixture(self.residuals)

    def _mixture(self, values):
        model = GaussianMixture(
            n_components=2,
            covariance_type="full",
            max_iter=200,
            n_init=1,
            reg_covar=self.config["covariance_floor_km2"],
            random_state=self.config["seed"],
        ).fit(values)
        if not model.converged_:
            raise ValueError("mixture fit did not converge; no silent Gaussian fallback")
        return model

    def _gaussian(self, values, noise, *, mean_zero=False):
        covariance = np.cov(values, rowvar=False) + np.eye(2) * self.config["covariance_floor_km2"]
        mean = np.zeros(2) if mean_zero else values.mean(axis=0)
        return mean + noise @ np.linalg.cholesky(covariance).T

    @staticmethod
    def _mixture_draws(model, noise, uniforms):
        indices = np.searchsorted(np.cumsum(model.weights_), uniforms, side="right")
        indices = np.minimum(indices, len(model.weights_) - 1)
        roots = np.linalg.cholesky(model.covariances_[indices])
        return model.means_[indices] + np.einsum("mij,mj->mi", roots, noise)

    def predict(self, query: dict, seed: int) -> dict[str, np.ndarray]:
        count, samples = len(query["features"]), self.config["samples"]
        rng = np.random.default_rng(seed)
        draws = {name: np.empty((count, samples, 2)) for name in METHODS}
        # Sobol is a deterministic kinematic proxy, not an ocean-physics ensemble.
        if samples & (samples - 1):
            raise ValueError("quadrature sample budget must be a power of two")
        uniforms = qmc.Sobol(2, scramble=True, seed=self.config["seed"]).random_base2(
            int(np.log2(samples)) - 1
        )
        positive = ndtri(np.clip(uniforms, 1e-10, 1 - 1e-10))
        quadrature = np.concatenate([positive, -positive])
        for i, feature in enumerate(query["features"]):
            distances = np.linalg.norm(self.scaled - (feature - self.center) / self.scale, axis=1)
            indices = np.argsort(distances, kind="stable")[: self.config["neighbors"]]
            local = self.residuals[indices]
            noise, uniform = rng.normal(size=(samples, 2)), rng.uniform(size=samples)
            empirical_global = self.residuals[rng.integers(len(self.residuals), size=samples)]
            empirical_local = local[rng.integers(len(local), size=samples)]
            outputs = {
                "global_gaussian": self._gaussian(self.residuals, noise),
                "global_mixture": self._mixture_draws(self.global_mixture, noise, uniform),
                "global_empirical": empirical_global,
                "analog_gaussian": self._gaussian(local, noise),
                "analog_mixture": self._mixture_draws(self._mixture(local), noise, uniform),
                "analog_empirical": empirical_local,
                "kinematic_quadrature": self._gaussian(self.residuals, quadrature, mean_zero=True),
            }
            for name in METHODS:
                draws[name][i] = query["kinematic_km"][i] + outputs[name]
        return draws
