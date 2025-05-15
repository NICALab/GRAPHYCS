import torch
import numpy as np


def add_poisson_dominant_noise(image: torch.Tensor,
                               target_snr: float,
                               gaussian_scale: float = 0.01) -> torch.Tensor:
    """
    Add predominantly Poisson noise, with a very small Gaussian component.

    Args:
        image:          clean image tensor (non-negative)
        target_snr:     desired SNR in dB (used for Poisson scaling)
        gaussian_scale: relative weight of the Gaussian noise (<<1 for Poisson dominance)

    Returns:
        noisy image tensor
    """
    # 1) Poisson noise (scaled so that SNR ≈ target_snr)
    snr_linear = 10 ** (target_snr / 10)
    mean_signal = image.mean()
    a = snr_linear**2 / mean_signal
    scaled = image * a
    noisy_poisson = torch.poisson(scaled) / a

    # 2) Very low‐level Gaussian noise
    amp = 10 ** (-target_snr / 10) * (image.max() - image.min())
    amp *= gaussian_scale
    noise_gauss = torch.randn_like(image) * amp

    return noisy_poisson + noise_gauss

