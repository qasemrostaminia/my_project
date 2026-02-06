"""2D schematic visualization of an Ising energy landscape (small N).

This script projects all binary spin states to 2D and visualizes their
energies and neighbor transitions, highlighting local minima (stable states).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SchematicResult:
    """Container for schematic data."""

    states: np.ndarray  # Shape: (2**N, N), entries in {-1, +1}
    energies: np.ndarray  # Shape: (2**N,)
    coords_2d: np.ndarray  # Shape: (2**N, 2)
    local_minima_indices: np.ndarray


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


def find_local_minima(energies: np.ndarray, n_regions: int) -> np.ndarray:
    """Identify local minima using single-spin-flip neighbors."""

    local_minima = []
    for idx, energy in enumerate(energies):
        neighbors = [idx ^ (1 << bit) for bit in range(n_regions)]
        if np.all(energy < energies[neighbors]):
            local_minima.append(idx)
    return np.array(local_minima, dtype=int)


def project_states_pca(states: np.ndarray) -> np.ndarray:
    """Project binary states to 2D using PCA (via SVD)."""

    centered = states - states.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def build_neighbor_edges(n_regions: int) -> list[tuple[int, int]]:
    """List edges between states that differ by one spin flip."""

    edges = []
    n_states = 2 ** n_regions
    for idx in range(n_states):
        for bit in range(n_regions):
            neighbor = idx ^ (1 << bit)
            if neighbor > idx:
                edges.append((idx, neighbor))
    return edges


def plot_energy_schematic(
    coords_2d: np.ndarray,
    energies: np.ndarray,
    edges: list[tuple[int, int]],
    local_minima: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> None:
    """Plot the 2D energy landscape schematic with neighbor transitions."""

    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, neighbor in edges:
        ax.plot(
            [coords_2d[idx, 0], coords_2d[neighbor, 0]],
            [coords_2d[idx, 1], coords_2d[neighbor, 1]],
            color="lightgray",
            linewidth=0.8,
            zorder=1,
        )

    size = 80 if probabilities is None else 80 + 400 * probabilities
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=energies,
        s=size,
        cmap="viridis_r",
        edgecolor="black",
        zorder=2,
    )

    ax.scatter(
        coords_2d[local_minima, 0],
        coords_2d[local_minima, 1],
        s=140,
        facecolor="none",
        edgecolor="red",
        linewidth=2,
        label="Local minima",
        zorder=3,
    )

    ax.set_title("2D schematic energy landscape")
    ax.set_xlabel("PC1 of spin states")
    ax.set_ylabel("PC2 of spin states")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, label="Energy (lower = deeper)")
    plt.tight_layout()
    plt.show()


def build_schematic(
    h: np.ndarray,
    J: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> SchematicResult:
    """Construct the 2D schematic for a small Ising system."""

    n_regions = h.shape[0]
    if n_regions > 4:
        raise ValueError("Use N <= 4 for a readable schematic.")

    states = enumerate_states(n_regions)
    energies = compute_energy(states, h, J)
    coords_2d = project_states_pca(states)
    local_minima = find_local_minima(energies, n_regions)

    return SchematicResult(
        states=states,
        energies=energies,
        coords_2d=coords_2d,
        local_minima_indices=local_minima,
    )


def main() -> None:
    """Example schematic for a small fitted Ising model."""

    rng = np.random.default_rng(4)
    n_regions = 4
    h = rng.normal(scale=0.3, size=n_regions)
    J = rng.normal(scale=0.15, size=(n_regions, n_regions))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    result = build_schematic(h, J)
    edges = build_neighbor_edges(n_regions)

    # Local minima correspond to stable network states (energy basins).
    plot_energy_schematic(
        result.coords_2d,
        result.energies,
        edges,
        result.local_minima_indices,
    )


if __name__ == "__main__":
    main()
