"""Step 3: Estimate Ising model parameters from binarized brain states.

This script fits a pairwise maximum entropy (Ising) model to binarized
fMRI data by matching empirical moments and maximizing pseudo-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult:
    """Container for fitted Ising parameters and diagnostics."""

    h: np.ndarray  # Shape: (N,)
    J: np.ndarray  # Shape: (N, N), symmetric, diagonal = 0
    empirical_mean: np.ndarray  # Shape: (N,)
    empirical_pair: np.ndarray  # Shape: (N, N)
    model_mean: np.ndarray  # Shape: (N,)
    model_pair: np.ndarray  # Shape: (N, N)
    history: list[dict[str, float]]


def compute_empirical_moments(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical first- and second-order moments."""

    if sigma.ndim != 2:
        raise ValueError("sigma must have shape (T, N).")
    if not np.all(np.isin(sigma, [-1, 1])):
        raise ValueError("sigma must contain only -1 and +1 values.")

    mean = sigma.mean(axis=0)
    pair = (sigma.T @ sigma) / sigma.shape[0]
    return mean, pair


def enumerate_states(n_regions: int) -> np.ndarray:
    """Enumerate all possible spin states in {-1, +1}."""

    n_states = 2 ** n_regions
    states = np.zeros((n_states, n_regions), dtype=int)
    for idx in range(n_states):
        bits = (idx >> np.arange(n_regions)) & 1
        states[idx] = bits * 2 - 1
    return states


def ising_energy(states: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Ising energy for each state."""

    fields = -states @ h
    couplings = -0.5 * np.sum(states @ J * states, axis=1)
    return fields + couplings


def model_expectations(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute model moments via exact enumeration (small N)."""

    n_regions = h.shape[0]
    states = enumerate_states(n_regions)
    energies = ising_energy(states, h, J)
    weights = np.exp(-energies - energies.max())
    probabilities = weights / weights.sum()
    mean = probabilities @ states
    pair = (states.T * probabilities) @ states
    return mean, pair


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    return 1.0 / (1.0 + np.exp(-x))


def fit_ising_pseudolikelihood(
    sigma: np.ndarray,
    learning_rate: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> FitResult:
    """Fit Ising parameters via gradient ascent on pseudo-likelihood.

    This approximation scales to moderate N and yields parameters that
    can be validated via exact enumeration when N is small (<= 8).
    """

    n_timepoints, n_regions = sigma.shape
    empirical_mean, empirical_pair = compute_empirical_moments(sigma)

    h = np.zeros(n_regions, dtype=float)
    J = np.zeros((n_regions, n_regions), dtype=float)
    history: list[dict[str, float]] = []

    for iteration in range(max_iter):
        h_grad = np.zeros_like(h)
        J_grad = np.zeros_like(J)

        for i in range(n_regions):
            field = h[i] + sigma @ J[:, i] - sigma[:, i] * J[i, i]
            prob = sigmoid(2.0 * sigma[:, i] * field)
            residual = 1.0 - prob
            h_grad[i] = (2.0 * sigma[:, i] * residual).mean()

            for j in range(n_regions):
                if i == j:
                    continue
                J_grad[j, i] = (2.0 * sigma[:, i] * sigma[:, j] * residual).mean()

        h_update = learning_rate * h_grad
        J_update = learning_rate * J_grad
        h += h_update
        J += J_update
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        update_norm = np.linalg.norm(h_update) + np.linalg.norm(J_update)
        history.append({"iteration": iteration, "update_norm": update_norm})

        if update_norm < tol:
            break

    model_mean, model_pair = model_expectations(h, J)
    return FitResult(
        h=h,
        J=J,
        empirical_mean=empirical_mean,
        empirical_pair=empirical_pair,
        model_mean=model_mean,
        model_pair=model_pair,
        history=history,
    )


def main() -> None:
    """Example usage with synthetic spin data."""

    rng = np.random.default_rng(2)
    sigma = rng.choice([-1, 1], size=(800, 6))

    result = fit_ising_pseudolikelihood(sigma)
    print("Fitted h:", result.h)
    print("Fitted J (first row):", result.J[0])
    print("Empirical mean:", result.empirical_mean)
    print("Model mean:", result.model_mean)
    print("Empirical pair (0,1):", result.empirical_pair[0, 1])
    print("Model pair (0,1):", result.model_pair[0, 1])


if __name__ == "__main__":
    main()
