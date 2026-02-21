"""
Differential Privacy (DP) processor.
Applies Gaussian noise to model updates (gradients/adapter deltas)
to ensure individual contributions cannot be re-identified.
"""
import numpy as np


def clip_gradients(gradients: list[float], clip_norm: float = 1.0) -> np.ndarray:
    """
    Clip gradient vector to a maximum L2 norm (gradient clipping).
    This bounds the sensitivity of each individual update.
    """
    grad_array = np.array(gradients, dtype=np.float64)
    norm = np.linalg.norm(grad_array)
    if norm > clip_norm:
        grad_array = grad_array * (clip_norm / norm)
    return grad_array


def add_gaussian_noise(
    clipped_gradients: np.ndarray,
    noise_multiplier: float = 1.1,
    clip_norm: float = 1.0,
) -> np.ndarray:
    """
    Add calibrated Gaussian noise for (ε, δ)-differential privacy.
    Noise std = noise_multiplier * clip_norm.
    """
    noise_std = noise_multiplier * clip_norm
    noise = np.random.normal(0, noise_std, size=clipped_gradients.shape)
    return clipped_gradients + noise


def apply_dp(
    gradients: list[float],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
) -> list[float]:
    """
    Full DP pipeline: clip → add noise.
    Returns the privatized gradient as a Python list.
    """
    clipped = clip_gradients(gradients, clip_norm)
    noised = add_gaussian_noise(clipped, noise_multiplier, clip_norm)
    return noised.tolist()


def validate_update(update: list[float], expected_dim: int) -> bool:
    """Validate that an incoming update has the correct dimensionality."""
    return isinstance(update, list) and len(update) == expected_dim
