"""Step 4: Construct and analyze the Ising energy landscape.

This module enumerates all possible brain states, computes their energies
under a fitted Ising model, identifies local minima, and optionally maps
each state to its basin of attraction via steepest descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class EnergyLandscapeResult:
    """Container for energy landscape analysis results."""

    states: np.ndarray  # Shape: (2**N, N), entries in {-1, +1}
    energies: np.ndarray  # Shape: (2**N,)
    local_minima_indices: np.ndarray  # Indices of local minima states
    basin_map: np.ndarray | None  # Shape: (2**N,), index of basin minimum


def enumerate_states(n_regions: int) -> np.ndarray:
    """Enumerate all possible spin states in {-1, +1}."""

    n_states = 2 ** n_regions
    states = np.zeros((n_states, n_regions), dtype=int)
    for idx in range(n_states):
        bits = (idx >> np.arange(n_regions)) & 1
        states[idx] = bits * 2 - 1
    return states


def compute_energy(states: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Ising energy for each state."""

    fields = -states @ h
    couplings = -0.5 * np.sum(states @ J * states, axis=1)
    return fields + couplings


def neighbor_indices(state_index: int, n_regions: int) -> np.ndarray:
    """Return indices of states at Hamming distance 1 from a state index."""

    return np.array([state_index ^ (1 << bit) for bit in range(n_regions)], dtype=int)


def find_local_minima(energies: np.ndarray, n_regions: int) -> np.ndarray:
    """Identify local minima: energy lower than all single-flip neighbors."""

    local_minima = []
    for idx, energy in enumerate(energies):
        neighbors = neighbor_indices(idx, n_regions)
        if np.all(energy < energies[neighbors]):
            local_minima.append(idx)
    return np.array(local_minima, dtype=int)


def compute_basins(energies: np.ndarray, n_regions: int) -> np.ndarray:
    """Assign each state to a local minimum via steepest descent."""

    basin_map = np.full(energies.shape[0], -1, dtype=int)
    for idx in range(energies.shape[0]):
        current = idx
        while True:
            neighbors = neighbor_indices(current, n_regions)
            next_state = neighbors[np.argmin(energies[neighbors])]
            if energies[next_state] < energies[current]:
                current = next_state
            else:
                basin_map[idx] = current
                break
    return basin_map


def analyze_energy_landscape(
    h: np.ndarray,
    J: np.ndarray,
    compute_basin_map: bool = True,
) -> EnergyLandscapeResult:
    """Construct the energy landscape from fitted Ising parameters."""

    n_regions = h.shape[0]
    states = enumerate_states(n_regions)
    energies = compute_energy(states, h, J)
    local_minima_indices = find_local_minima(energies, n_regions)
    basin_map = compute_basins(energies, n_regions) if compute_basin_map else None

    return EnergyLandscapeResult(
        states=states,
        energies=energies,
        local_minima_indices=local_minima_indices,
        basin_map=basin_map,
    )


def plot_energy_rank(energies: np.ndarray) -> None:
    """Plot energies in rank order."""

    sorted_energies = np.sort(energies)
    plt.figure(figsize=(7, 4))
    plt.plot(sorted_energies, marker="o", linestyle="none")
    plt.title("Energy landscape (rank-ordered states)")
    plt.xlabel("State rank")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Example usage with synthetic Ising parameters."""

    rng = np.random.default_rng(3)
    n_regions = 6
    h = rng.normal(scale=0.2, size=n_regions)
    J = rng.normal(scale=0.1, size=(n_regions, n_regions))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    result = analyze_energy_landscape(h, J, compute_basin_map=True)

    print(f"Local minima count: {len(result.local_minima_indices)}")
    minima_energies = result.energies[result.local_minima_indices]
    ranked = np.argsort(minima_energies)
    print("Local minima (lowest energy first):")
    for rank_idx in ranked:
        state_idx = result.local_minima_indices[rank_idx]
        print(f"State {state_idx}: energy={minima_energies[rank_idx]:.3f}")

    # Local minima correspond to stable brain states in the energy landscape.
    plot_energy_rank(result.energies)


if __name__ == "__main__":
    main()
