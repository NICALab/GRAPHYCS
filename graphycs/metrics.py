import numpy as np
from scipy.fftpack import dct
import torch
import torch.nn.functional as F
from math import exp

def fourier_metric(image, wavelength=0.513, NA=0.3, camera_pixel_size = 0.5434):
    n = 1.33
    k = 2 * n * np.pi / wavelength  # wavenumber
    maxSpatialFreq = ((NA) / wavelength) #/ (1 / (2 * cameraPixelSize))
    maxSpatialFreqRatio = maxSpatialFreq / (1 / (2 * camera_pixel_size))
    lower_pass_band = 0.2

    kx, ky = np.fft.fftfreq(image.shape[0], d = (camera_pixel_size)), np.fft.fftfreq(image.shape[1], d = (camera_pixel_size))
    kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)

    kxx, kyy = np.meshgrid(kx, ky)

    frequency_crop = (kxx**2 + kyy**2) <= maxSpatialFreq**2

    fourier_spectrum = np.fft.fftshift(np.fft.fft2(image))
    fourier_spectrum = np.abs(fourier_spectrum)

    fourier_spectrum = fourier_spectrum * frequency_crop

    high_frequency_mask = ((kxx**2 + kyy**2) >= (maxSpatialFreq*lower_pass_band)**2) * frequency_crop
    low_frequency_mask = (kxx**2 + kyy**2) <= (maxSpatialFreq*lower_pass_band)**2

    high_frequency_spectrum = fourier_spectrum * high_frequency_mask
    high_frequency_spectrum_sum = high_frequency_spectrum.sum()
    
    low_frequency_spectrum = fourier_spectrum * low_frequency_mask
    low_frequency_spectrum_sum = low_frequency_spectrum.sum()
    

    return high_frequency_spectrum_sum / low_frequency_spectrum_sum



def dct_metric_pd(image, wavelength=0.513, NA=0.3, camera_pixel_size = 0.5417):
    maxSpatialFreq = ((NA) / wavelength) #/ (1 / (2 * cameraPixelSize))


    r0 = 2 * NA / wavelength
    

    img_dct = dct(dct(image.T, norm='ortho').T, norm='ortho')
    img_norm = np.linalg.norm(img_dct)
    img_abs = np.abs(img_dct / img_norm)

    kx, ky = np.fft.fftfreq(image.shape[0], d = (camera_pixel_size)), np.fft.fftfreq(image.shape[1], d = (camera_pixel_size))
    kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
    kxx, kyy = np.meshgrid(kx, ky)
    is_within_radius  = (kxx**2 + kyy**2) <= maxSpatialFreq**2
    is_nonzero_img_dct = img_dct != 0
    is_index_for_sum = np.logical_and(is_within_radius, is_nonzero_img_dct)

    img_abs_selected = img_abs[is_index_for_sum]
    sum_value = np.sum(img_abs_selected * np.log2(img_abs_selected))

    # Calculate final image quality metric
    image_quality_metrics = -2 / r0**2 * sum_value
    

    return image_quality_metrics

