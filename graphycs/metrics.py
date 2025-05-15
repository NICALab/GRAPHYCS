import numpy as np
from scipy.fftpack import dct
import torch
import torch.nn.functional as F
from math import exp

def fourier_metric_uao(image, wavelength=0.513, NA=0.3, camera_pixel_size = 0.5417):
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



def fourier_metric(image, sigma, wavelength=0.4, na=1.4):
    """
    Calculate the Fourier Metric (FM) for an image.
    
    Parameters:
    - image (np.ndarray): 2D numpy array representing the image.
    - sigma (float): Standard deviation of the Gaussian function (in 1/µm).
    - wavelength (float): Wavelength of the light (in µm).
    - na (float): Numerical aperture of the system.
    
    Returns:
    - fm (float): The calculated Fourier Metric.
    """
    # Compute the Fourier Transform of the image
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    fft_magnitude = np.abs(fft_image)
    
    # Define the spatial frequency coordinates
    ny, nx = image.shape
    y = np.fft.fftshift(np.fft.fftfreq(ny))
    x = np.fft.fftshift(np.fft.fftfreq(nx))
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    
    # Define the cutoff frequency (NA / wavelength)
    cutoff_frequency = na / wavelength
    
    # Create the circular mask (circ)
    circ_mask = rho <= cutoff_frequency
    
    # Apply the Gaussian function
    gaussian_filter = np.exp(-rho**2 / (2 * sigma**2))
    
    # Calculate the numerator (high-frequency content within circ mask)
    numerator = np.sum((fft_magnitude * gaussian_filter * circ_mask)**2)
    
    # Calculate the denominator (total intensity)
    denominator = np.sum(fft_magnitude**2)
    
    # Compute the Fourier Metric
    fm = numerator / denominator if denominator != 0 else 0
    
    return fm


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# def create_3d_window(window_size, channel = 1):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(-1)
#     _3D_window = (_2D_window * _1D_window.t().unsqueeze(0)).unsqueeze(0).unsqueeze(0)
#     window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
#     return window

# def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
#     # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
#     if val_range is None:
#         if torch.max(img1) > 128:
#             max_val = 255
#         else:
#             max_val = 1

#         if torch.min(img1) < -0.5:
#             min_val = -1
#         else:
#             min_val = 0
#         L = max_val - min_val
#     else:
#         L = val_range

#     padd = 0
#     (_, channel, depth, height, width) = img1.size()
#     if window is None:
#         real_size = min(window_size, depth, height, width)
#         window = create_3d_window(real_size, channel=channel).to(img1.device)

#     mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2

#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = v1 / v2  # contrast sensitivity

#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

#     if size_average:
#         cs = cs.mean()
#         ret = ssim_map.mean()
#     else:
#         cs = cs.mean(1).mean(1).mean(1)
#         ret = ssim_map.mean(1).mean(1).mean(1)

#     if full:
#         return ret, cs
#     return ret


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Set value range L from the provided val_range or infer it.
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    # For 2D images, the size is (B, C, H, W)
    (_, channel, height, width) = img1.size()
    
    # Create a 2D window if not provided.
    if window is None:
        real_size = min(window_size, height, width)
        window = create_2d_window(torch.tensor(real_size), channel=channel).to(img1.device)
    
    # Compute local means using 2D convolution
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Compute intermediate terms for contrast sensitivity
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1)

    if full:
        return ret, cs
    return ret

# Example 2D window creation function (Gaussian window)
def create_2d_window(window_size, channel):
    # Create a 1D Gaussian kernel
    def gauss(window_size, sigma):
        gauss_vals = torch.Tensor([torch.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss_vals / gauss_vals.sum()
    
    sigma = 1.5  # You can adjust this value
    _1D_window = gauss(window_size, sigma).unsqueeze(1)
    # Create a 2D kernel via outer product
    _2D_window = _1D_window.mm(_1D_window.t())
    # Reshape into [1, 1, H, W]
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    # Expand window to have the same number of channels as the image
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window