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


def add_noise_to_image(image, target_snr):
    noise_power_sqrt_gaussian = 10 ** (-target_snr / 10) * (image.max() - image.min())
    noise_gaussian = torch.randn_like(image) * noise_power_sqrt_gaussian
    # noise_gaussian = noise_gaussian.cuda()
    signal_power = image.mean()
    snr_linear = 10 ** (target_snr / 10)  # Convert SNR from dB to linear scale
    a = snr_linear**2 / signal_power
    # noise_power_poisson = signal_power / snr_linear
    # scale_factor = signal_power / noise_power_poisson

    ## scale factor is essentially equal to snr_linear
    scaled_image = image * a
    noisy_image_poisson = torch.poisson(scaled_image)
    noisy_image_poisson = noisy_image_poisson / a

    return noisy_image_poisson + noise_gaussian

## taken from CoCoA
def apply_poisson_noise(img, max_flux):
    img = torch.poisson(img * max_flux)
    img = img / max_flux
    
    return img
