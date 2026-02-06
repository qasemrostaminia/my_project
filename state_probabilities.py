"""Step 2: Enumerate brain states and estimate empirical probabilities.

This script takes binarized fMRI data (sigma in {-1, +1}) and computes
the empirical distribution P_data(sigma) over observed brain states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StateProbabilityResult:
    """Container for observed states and their empirical probabilities."""

    states_binary: np.ndarray  # Shape: (n_observed, N), entries in {0, 1}
    states_spin: np.ndarray  # Shape: (n_observed, N), entries in {-1, +1}
    labels: np.ndarray  # Shape: (n_observed,), integer labels for each state
    counts: np.ndarray  # Shape: (n_observed,), occurrences of each state
    probabilities: np.ndarray  # Shape: (n_observed,), empirical probabilities
    n_possible_states: int  # Total possible states = 2**N


def sigma_to_binary(sigma: np.ndarray) -> np.ndarray:
    """Convert spin states {-1, +1} to binary {0, 1}."""

    if sigma.ndim != 2:
        raise ValueError("sigma must have shape (T, N).")
    if not np.all(np.isin(sigma, [-1, 1])):
        raise ValueError("sigma must contain only -1 and +1 values.")
    return ((sigma + 1) // 2).astype(int)


def binary_to_labels(states_binary: np.ndarray) -> np.ndarray:
    """Encode each binary state as a unique integer label."""

    n_regions = states_binary.shape[1]
    powers_of_two = 2 ** np.arange(n_regions)
    return states_binary @ powers_of_two


def enumerate_states(sigma: np.ndarray) -> StateProbabilityResult:
    """Enumerate observed states and compute empirical probabilities.

    Returns a result object containing the observed states, counts,
    and probabilities. The labels are unique integer encodings of each state.
    """

    states_binary = sigma_to_binary(sigma)
    labels = binary_to_labels(states_binary)

    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    # Reconstruct observed states from labels for downstream Ising steps.
    n_regions = sigma.shape[1]
    states_binary_observed = (
        ((unique_labels[:, None] & (1 << np.arange(n_regions))) > 0).astype(int)
    )
    states_spin_observed = states_binary_observed * 2 - 1

    n_possible_states = 2 ** n_regions

    # Sanity check: probabilities should sum to 1 (within floating error).
    if not np.isclose(probabilities.sum(), 1.0):
        raise RuntimeError("Empirical probabilities do not sum to 1.")

    return StateProbabilityResult(
        states_binary=states_binary_observed,
        states_spin=states_spin_observed,
        labels=unique_labels,
        counts=counts,
        probabilities=probabilities,
        n_possible_states=n_possible_states,
    )


def plot_state_probabilities(probabilities: np.ndarray) -> None:
    """Plot a rank-ordered distribution of state probabilities."""

    sorted_probs = np.sort(probabilities)[::-1]
    plt.figure(figsize=(7, 4))
    plt.plot(sorted_probs, marker="o", linestyle="none")
    plt.title("Rank-ordered empirical state probabilities")
    plt.xlabel("State rank")
    plt.ylabel("P_data(state)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Example usage with synthetic binarized data."""

    rng = np.random.default_rng(1)
    sigma = rng.choice([-1, 1], size=(500, 6))

    result = enumerate_states(sigma)

    print(f"Observed states: {len(result.labels)} / {result.n_possible_states}")
    print("First five states (spin representation):")
    print(result.states_spin[:5])
    print("First five probabilities:")
    print(result.probabilities[:5])

    # Sparse sampling note: as N grows, 2**N grows rapidly, so many
    # theoretically possible states may never be observed in finite data.

    plot_state_probabilities(result.probabilities)


if __name__ == "__main__":
    main()
