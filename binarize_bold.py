"""Step 1: Binarize z-scored fMRI BOLD signals for energy landscape analysis.

This script converts continuous BOLD signals into binary spin states
(-1/+1), which serve as inputs to the Ising / pairwise maximum entropy model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BinarizationConfig:
    """Configuration for binarizing BOLD signals."""

    threshold: float = 0.0  # For z-scored signals, mean is ~0.
    visualize: bool = True


def binarize_bold(bold_z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Convert z-scored BOLD signals into binary spin states.

    Each time series is thresholded relative to its temporal mean (0 for z-scored
    data) to yield +1 for above-threshold activity and -1 otherwise. This step
    maps continuous activity into discrete spin variables for the Ising model.
    """

    if bold_z.ndim != 2:
        raise ValueError("bold_z must have shape (T, N).")

    sigma = np.where(bold_z > threshold, 1, -1).astype(int)

    # Ensure no zeros remain after binarization.
    if np.any(sigma == 0):
        raise RuntimeError("Zero-valued states found after binarization.")

    return sigma


def summarize_states(sigma: np.ndarray) -> dict[str, np.ndarray]:
    """Compute basic sanity checks on the binarized states."""

    if sigma.ndim != 2:
        raise ValueError("sigma must have shape (T, N).")

    proportion_plus = (sigma == 1).mean(axis=0)
    proportion_minus = (sigma == -1).mean(axis=0)
    return {
        "proportion_plus": proportion_plus,
        "proportion_minus": proportion_minus,
    }


def plot_binarized_activity(sigma: np.ndarray) -> None:
    """Visualize the binarized activity matrix (regions x time)."""

    plt.figure(figsize=(10, 4))
    plt.imshow(sigma.T, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Binarized activity (regions x time)")
    plt.xlabel("Time")
    plt.ylabel("Region")
    plt.colorbar(label="Spin state")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Example usage with synthetic z-scored data."""

    rng = np.random.default_rng(0)
    bold_z = rng.normal(size=(500, 6))  # Placeholder z-scored signals

    config = BinarizationConfig()
    sigma = binarize_bold(bold_z, threshold=config.threshold)
    summary = summarize_states(sigma)

    print("Proportion +1 per region:", summary["proportion_plus"])
    print("Proportion -1 per region:", summary["proportion_minus"])

    if config.visualize:
        plot_binarized_activity(sigma)


if __name__ == "__main__":
    main()
