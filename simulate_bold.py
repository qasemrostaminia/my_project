"""Simulate synthetic fMRI BOLD signals for multiple brain regions.

This script generates latent neural activity with inter-regional coupling,
transforms it into BOLD-like signals via HRF convolution, adds realistic
noise, and normalizes the results for later binarization and energy landscape
analysis steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gamma


@dataclass
class SimulationConfig:
    """Configuration for synthetic fMRI BOLD simulation."""

    n_regions: int = 8
    tr: float = 0.72  # Repetition time (seconds)
    duration: float = 600.0  # Total scan duration (seconds)
    neural_time_constant: float = 1.5  # AR(1) timescale (seconds)
    coupling_strength: float = 0.2  # Scale for inter-regional coupling
    process_noise: float = 0.8  # Latent neural noise
    bold_noise: float = 0.5  # Observation noise (BOLD)
    seed: int = 7


def build_coupling_matrix(n_regions: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Create a symmetric coupling matrix with modular structure.

    The coupling matrix encodes inter-regional influences that induce
    correlated latent neural dynamics, a prerequisite for realistic
    resting-state BOLD correlations.
    """

    base = rng.normal(loc=0.0, scale=1.0, size=(n_regions, n_regions))
    base = (base + base.T) / 2.0
    np.fill_diagonal(base, 0.0)

    # Impose a simple modular structure to create correlation clusters.
    n_modules = max(2, n_regions // 3)
    module_size = n_regions // n_modules
    for module_idx in range(n_modules):
        start = module_idx * module_size
        end = n_regions if module_idx == n_modules - 1 else (module_idx + 1) * module_size
        base[start:end, start:end] += 2.0

    # Normalize to keep the dynamics stable.
    base /= np.max(np.abs(base))
    return base


def simulate_latent_neural(config: SimulationConfig) -> NDArray[np.float64]:
    """Simulate latent neural activity using a coupled AR(1) model."""

    rng = np.random.default_rng(config.seed)
    n_timepoints = int(config.duration / config.tr)

    coupling = build_coupling_matrix(config.n_regions, rng) * config.coupling_strength

    # AR(1) coefficient derived from the desired timescale.
    alpha = np.exp(-config.tr / config.neural_time_constant)
    latent = np.zeros((n_timepoints, config.n_regions), dtype=float)

    for t in range(1, n_timepoints):
        noise = rng.normal(scale=config.process_noise, size=config.n_regions)
        latent[t] = (
            alpha * latent[t - 1]
            + (1.0 - alpha) * (latent[t - 1] @ coupling)
            + noise
        )

    return latent


def canonical_hrf(tr: float, duration: float = 32.0) -> NDArray[np.float64]:
    """Generate a canonical hemodynamic response function (HRF).

    This uses a double-gamma model commonly used in fMRI analyses to
    approximate the vascular response to neural activity.
    """

    dt = tr
    time_axis = np.arange(0, duration, dt)

    # Parameters adapted from SPM canonical HRF.
    peak1 = gamma.pdf(time_axis, 6)
    peak2 = gamma.pdf(time_axis, 12)
    hrf = peak1 - 0.35 * peak2
    hrf /= np.max(np.abs(hrf))
    return hrf


def convolve_hrf(neural: NDArray[np.float64], tr: float) -> NDArray[np.float64]:
    """Convolve each neural time series with the canonical HRF."""

    hrf = canonical_hrf(tr)
    bold = np.vstack([
        np.convolve(neural[:, idx], hrf, mode="full")[: neural.shape[0]]
        for idx in range(neural.shape[1])
    ]).T
    return bold


def add_bold_noise(bold: NDArray[np.float64], noise_scale: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Add Gaussian observation noise to mimic scanner/physiological noise."""

    noise = rng.normal(scale=noise_scale, size=bold.shape)
    return bold + noise


def zscore_signals(bold: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize each region's signal for later binarization steps."""

    mean = bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    return (bold - mean) / std


def simulate_bold_signals(config: SimulationConfig) -> NDArray[np.float64]:
    """Run the full simulation pipeline for synthetic BOLD signals."""

    rng = np.random.default_rng(config.seed)
    latent = simulate_latent_neural(config)
    bold = convolve_hrf(latent, config.tr)
    bold_noisy = add_bold_noise(bold, config.bold_noise, rng)
    bold_z = zscore_signals(bold_noisy)
    return bold_z


def main() -> None:
    """Simulate BOLD signals and plot a quick summary."""

    import matplotlib.pyplot as plt

    config = SimulationConfig()
    bold_z = simulate_bold_signals(config)

    time_axis = np.arange(bold_z.shape[0]) * config.tr

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(time_axis, bold_z[:, :4])
    axes[0].set_title("Simulated BOLD signals (first 4 regions)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Z-scored BOLD")

    corr = np.corrcoef(bold_z.T)
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Inter-regional correlation matrix")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()


if __name__ == "__main__":
    main()
