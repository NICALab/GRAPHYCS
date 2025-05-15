
"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     Heintzmann Lab, Friedrich-Schiller-University Jena, Germany

@author: Rainer Heintzmann, Sheng Liu, Jonas Hellgoth
"""

import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import scipy as sp
from math import factorial 
import numbers
from scipy import ndimage
import scipy.fft as fft
defaultTFDataType="float32"
defaultTFCpxDataType="complex64"

def extract_center_crop(img):
    object = img.squeeze().clone()
    object = object.unsqueeze(0).unsqueeze(0)

    # if img.dim() == 3:  # [N, H, W]
    #     _, H, W = img.shape
    # else:
    H, W = img.shape

    patch_h = H // 4
    patch_w = W // 4

    # Remove padding and directly unfold into patches
    object = object.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # object now has shape [1, 1, num_patches_H, num_patches_W, patch_h, patch_w]
    # Assuming a 4x4 grid, num_patches_H and num_patches_W are both 4

    # Extract the center 4 regions (using 0-indexing, take rows 1:3 and columns 1:3)
    center_object = object[:, :, 1:3, 1:3, :, :]

    # Reshape to combine the center patches into a single dimension (4 patches total)
    center_object = center_object.reshape(1, 4, patch_h, patch_w)
    center_object = center_object.squeeze(0)
    top_row = torch.cat((center_object[0], center_object[1]), dim=1)  # shape: [patch_h, 2*patch_w]
    bottom_row = torch.cat((center_object[2], center_object[3]), dim=1)  # shape: [patch_h, 2*patch_w]

    # Combine the two rows along the height to form the final image.
    combined_image = torch.cat((top_row, bottom_row), dim=0) 
    return combined_image


def extract_center_crop_stack(img_stack):
    # Assuming img_stack has shape [N, H, W] (or [N, 1, H, W])
    # If there is no channel dimension, add one.
    if img_stack.dim() == 3:  # [N, H, W]
        img_stack = img_stack.unsqueeze(1)  # now [N, 1, H, W]

    _, _, H, W = img_stack.shape

    patch_h = H // 4
    patch_w = W // 4

    # The following operations will work on each image in the batch
    # Unfold spatial dimensions into patches for each image in the batch
    # Here we assume that patch_h and patch_w evenly divide H and W to get a 4x4 grid.
    patches = img_stack.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # patches shape: [N, 1, num_patches_H, num_patches_W, patch_h, patch_w]
    
    # Extract the center 2x2 patches from the 4x4 grid
    center_patches = patches[:, :, 1:3, 1:3, :, :]
    # center_patches shape: [N, 1, 2, 2, patch_h, patch_w]

    # Rearrange so that the 2x2 grid is reassembled back into a single image for each sample
    # First, remove the channel dimension if not needed.
    center_patches = center_patches.squeeze(1)  # now [N, 2, 2, patch_h, patch_w]
    
    # For each image, combine patches along width then height.
    # We'll use a list comprehension over the batch dimension:
    combined_images = []
    for i in range(center_patches.size(0)):
        # Concatenate patches along width for the top and bottom rows.
        top_row = torch.cat((center_patches[i, 0, 0], center_patches[i, 0, 1]), dim=1)    # [patch_h, 2*patch_w]
        bottom_row = torch.cat((center_patches[i, 1, 0], center_patches[i, 1, 1]), dim=1) # [patch_h, 2*patch_w]
        # Concatenate the two rows along the height.
        combined_image = torch.cat((top_row, bottom_row), dim=0)  # [2*patch_h, 2*patch_w]
        combined_images.append(combined_image)
    
    # Stack the images back into a single tensor
    combined_images = torch.stack(combined_images)  # shape: [N, 2*patch_h, 2*patch_w]
    return combined_images


def michelson_contrast(image):
    max_intensity = image.max()
    min_intensity = image.min()
    contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity + 1e-9)  
    return contrast


def nuclear_norm(x):
    """
    Compute the nuclear norm of a matrix (sum of singular values).
    """
    u, s, v = torch.svd(x)
    return torch.sum(s)


# richardson_lucyregularization function
def RL_loss(recon_observed_img, observed_img, psf):
    """
    Compute the loss for the Richardson-Lucy deconvolution process using FFT.

    Parameters:
        recon_observed_img (torch.Tensor): Reconstructed observed image (after deconvolution).
        observed_img (torch.Tensor): Original observed image (blurred and noisy).
        psf (torch.Tensor): Point spread function.

    Returns:
        torch.Tensor: Regularization loss.
    """
    # Flip the PSF for backward projection
    psf_flip = torch.flip(psf, dims=(-2, -1))

    # Avoid division by zero in the ratio computation
    epsilon = 1e-10
    recon_observed_img = torch.clamp(recon_observed_img, min=epsilon)

    # Compute the ratio of observed to reconstructed
    ratio = observed_img / recon_observed_img

    # Convert ratio and PSF to FFT space
    img_fft = torch.fft.fft2(ratio)
    psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_flip), s=ratio.shape[-2:])

    # Perform convolution in FFT space
    imgBlurFFT = img_fft * psf_fft

    # Inverse FFT to return to spatial domain
    ratio_corrected = torch.real(torch.fft.ifft2(imgBlurFFT))

    # Compute the loss as the deviation of the correction from 1
    correction = ratio_corrected
    loss = torch.abs(1.0 - correction).mean()

    return loss


# Poisson noise regularization function
def poisson_loss(recon_img, observed_img, epsilon=1e-8):
    """
    Poisson loss for observed image and reconstructed image
    Args:
        recon_img: Reconstructed image (torch.Tensor)
        observed_img: Observed image (torch.Tensor)
        epsilon: Small value to prevent log(0)
    Returns:
        Loss value (torch.Tensor)
    """
    return torch.sum(recon_img - observed_img * torch.log(recon_img + epsilon))

# Define a function for TV regularization
def total_variation_loss(img):
    """
    Computes the total variation loss for a tensor `img`.
    Args:
        img (torch.Tensor): Input tensor of shape (H, W), (C, H, W), or (B, C, H, W).
    Returns:
        torch.Tensor: TV loss.
    """
    if len(img.shape) == 4:  # Batch of images: (B, C, H, W)
        h_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        w_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
    elif len(img.shape) == 3:  # Single image: (C, H, W)
        h_diff = img[:, 1:, :] - img[:, :-1, :]
        w_diff = img[:, :, 1:] - img[:, :, :-1]
        loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
    elif len(img.shape) == 2:  # 2D tensor: (H, W)
        h_diff = img[1:, :] - img[:-1, :]
        w_diff = img[:, 1:] - img[:, :-1]
        loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
    else:
        raise ValueError(f"Unsupported tensor shape for TV loss: {img.shape}")
    
    return loss

## added on 10/17/2024: this is based on COCOA's 3d psf generation code
## returns a 3D PSF of size psf_shape and psf_units can be in any units, but should be consistent with lmbda
def psf_3D_generator(psf_shape, psf_units, lmbda, n, NA, dtype):

    M_x, M_y, M_z = psf_shape
    dx, dy, dz = psf_units
    z = dz * (torch.arange(M_z) - M_z // 2)

    ## create a defocus term for propagation:

    ## fftfreq: returns sampling frequencies for the DFT which are fftshifted (centered at the maximum and minimum frequencies)
    ## frequencies are between (-1/(2 * dx)) to (1/(2 * dx))
    k_x, k_y = fft.fftfreq(M_x, dx), fft.fftfreq(M_y, dy)
    z, k_x, k_y = z.type(dtype).cuda(0), k_x.type(dtype).cuda(0), k_y.type(dtype).cuda(0)

    K_Z, K_X, K_Y = torch.meshgrid(z, k_x, k_y, indexing="ij")
    K_R = torch.sqrt(K_X ** 2 + K_Y ** 2)

    max_freq = NA / (lmbda / n)
    k_mask = (K_R <= max_freq).type(dtype).cuda(0)
    defocus_phase = torch.sqrt(1. * (n/lmbda) ** 2 - K_R ** 2).type(dtype).cuda(0)
    # defocus_phase = torch.sqrt(1. * n ** 2 - K_R ** 2 * wavelength ** 2).type(dtype).cuda(0)

    out_ind = torch.isnan(defocus_phase)
    k_prop = torch.exp(-2.j * np.pi * (K_Z) * defocus_phase)
    k_prop[out_ind] = 0.
    k_base = k_mask * k_prop

    psf3D = fft.fftshift(fft.fftshift(fft.ifftn(k_base, dim=(1,2)), dim=(0,)))
    psf3D = torch.abs(psf3D) ** 2
    psf3D = psf3D / psf3D.sum()

    return psf3D



### ssim code taken from COCOA:


def ssim_loss(img1, img2):
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_mu2 = mu1 * mu2
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    
    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    loss = 1.0 - (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / torch.clamp((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min = 1e-8, max = 1e8))
        
    return loss



##### The following code for zernike polynomial generation / noll and ANSI indexing is copied from the uiPSF code
##### https://github.com/ries-lab/uiPSF


def nl2noll(n,l):
    mm = abs(l)
    j = n * (n + 1) / 2 + 1 + max(0, mm - 1)
    if ((l > 0) & (np.mod(n, 4) >= 2)) | ((l < 0) & (np.mod(n, 4) <= 1)):
       j = j + 1
    
    return np.int32(j)

def noll2nl(j):
    n = np.ceil((-3 + np.sqrt(1 + 8*j)) / 2)
    l = j - n * (n + 1) / 2 - 1
    if np.mod(n, 2) != np.mod(l, 2):
       l = l + 1
    
    if np.mod(j, 2) == 1:
       l= -l
    
    return np.int32(n),np.int32(l)

def nl2ansi(n,l):
    j = (n*(n+2)+l)/2
    return j

def noll2ansi(i):
    n, l = noll2nl(i)
    return int(nl2ansi(n, l))

def radialpoly(n,m,rho):
    if m==0:
        g = np.sqrt(n+1)
    else:
        g = np.sqrt(2*n+2)
    r = np.zeros(rho.shape)
    for k in range(0,(n-m)//2+1):
        coeff = g*((-1)**k)*factorial(n-k)/factorial(k)/factorial((n+m)//2-k)/factorial((n-m)//2-k)
        p = rho**(n-2*k)
        r += coeff*p

    return r

def genZern1(n_max,xsz):
    Nk = (n_max+1)*(n_max+2)//2
    Z = np.ones((Nk,xsz,xsz))
    pkx = 2/xsz
    xrange = np.linspace(-xsz/2+0.5,xsz/2-0.5,xsz)
    [xx,yy] = np.meshgrid(xrange,xrange)
    rho = np.lib.scimath.sqrt((xx*pkx)**2+(yy*pkx)**2)
    phi = np.arctan2(yy,xx)

    for j in range(0,Nk):
        [n,l] = noll2nl(j+1)
        m = np.abs(l)
        r = radialpoly(n,m,rho)
        if l<0:
            Z[j] = r*np.sin(phi*m)
        else:
            Z[j] = r*np.cos(phi*m)
    return Z


## return all of the ansi indices for a given n_max
# def genZernAnsi(order_max, n_max_ansi,xsz):
#     zernikesNoll = genZern1(order_max,xsz)
#     zernikesAnsi = np.zeros((n_max_ansi + 1,xsz,xsz))
#     n_max_noll = n_max_ansi + 1 ## since noll indexing starts at one, whereas ansi indexing starts at 0
#     noll_indices = np.arange(1,n_max_noll+1,1).tolist()
#     ansi_indices = [noll2ansi(i) for i in noll_indices]
#     for i in range(len(zernikesNoll)):
#         zernikesAnsi[ansi_indices[i]] = zernikesNoll[i]

#     return zernikesAnsi

def genZernAnsi(order_max, n_max_ansi,xsz):
    zernikesNoll = genZern1(order_max,xsz)
    zernikesAnsi = np.zeros((len(zernikesNoll),xsz,xsz))
    # n_max_noll = n_max_ansi + 1 ## since noll indexing starts at one, whereas ansi indexing starts at 0
    n_max_noll = 21 ## assuming a maximum order of 5
    if order_max == 4:
        n_max_noll = 15
    elif order_max == 5:
        n_max_noll = 21
    elif order_max == 6:
        n_max_noll = 28
    noll_indices = np.arange(1,n_max_noll+1,1).tolist()
    ansi_indices = [noll2ansi(i) for i in noll_indices]

    print(ansi_indices)
    for i in range(len(zernikesNoll)):
        zernikesAnsi[ansi_indices[i]] = zernikesNoll[i]
    
    zernikesAnsi = zernikesAnsi[:n_max_ansi+1]

    return zernikesAnsi


def zernike_default_generation(order_max, n_max_ansi,xsz):

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, xsz), torch.linspace(-1, 1, xsz))
    dm_crop = torch.zeros((xsz, xsz))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, xsz)
    zernike = torch.zeros(((n_max_ansi + 1), xsz, xsz))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    return dm_crop, zernike


def zernike_pd_generation(order_max, n_max_ansi, M, pixelSize, wavelength, NA):
    ratio = pupilRadius(M, pixelSize, wavelength, NA)
    R = int( (1/ratio) * M)

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR


def zernike_pd_generation_revised(order_max, n_max_ansi, M, pixelSize, wavelength, NA, n_imm):
    maxSpatialFrequency = (NA) / wavelength
    R = int( (maxSpatialFrequency * 2 * pixelSize) * M)

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR


def zernike_generation_R(order_max, n_max_ansi, M, R):

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR


def zernike_pd_generation_simple(order_max, n_max_ansi, M):
    R = int(0.8 * M)

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR, R



def zernike_pd_generation_redo(order_max, n_max_ansi, M, pixelSize, wavelength, NA, n_imm):
    maxSpatialFrequency = (NA * n_imm) / wavelength
    R = int( (maxSpatialFrequency * 2 * pixelSize) * M) + int( (maxSpatialFrequency * 2 * pixelSize) * M  * ((0.75 - 0.04) / 5.04))
    print("R: ", R)

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR


def zernike_pd_generation_revised_dm(order_max, n_max_ansi, M, pixelSize, wavelength, NA, n_imm):
    maxSpatialFrequency = (NA * n_imm) / wavelength
    R = int( (maxSpatialFrequency * 2 * pixelSize) * M) + int( (maxSpatialFrequency * 2 * pixelSize) * M  * ((0.75 - 0.04) / 5.04))
    print("R: ", R)

    zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)
    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR

## for polynomial order greater than 14, use the rest of the Zernike polynomials for fitting (i.e. greater than 15)
def zernike_pd_generation_higher_order(order_max, n_max_ansi, M, pixelSize, wavelength, NA):
    ratio = pupilRadius(M, pixelSize, wavelength, NA)
    R = int( (1/ratio) * M)

    if n_max_ansi > 14:
        zernike_non_norm = genZernAnsi(6, 14, R)
        zernikes_higher_order = genZern1(order_max, R)
        zernikes_higher_order = zernikes_higher_order[15:n_max_ansi+1]
        zernike_non_norm = np.concatenate((zernike_non_norm, zernikes_higher_order), axis=0)
    else:
        print("n_max_ansi:", n_max_ansi)
        print("order_max:", order_max)
        zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)
    
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    ##normalize and then exclude the first piston zernike (to be compatible with PD's matlab code)

    # print("shape: ", len(zernike_non_norm))

    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR

def prechirpz1(kpixelsize, pixelsize_x, pixelsize_y, N, M, device):
    krange = torch.linspace(-N/2+0.5, N/2-0.5, N, dtype=torch.float32).to(device)
    xxK, yyK = torch.meshgrid(krange, krange, indexing='ij')
    xxK, yyK = xxK.to(device), yyK.to(device)
    
    xrange = torch.linspace(-M/2+0.5, M/2-0.5, M, dtype=torch.float32).to(device)
    xxR, yyR = torch.meshgrid(xrange, xrange, indexing='ij')
    xxR, yyR = xxR.to(device), yyR.to(device)
    
    a = 1j * torch.pi * kpixelsize
    A = torch.exp(a * (pixelsize_x * xxK**2 + pixelsize_y * yyK**2))
    C = torch.exp(a * (pixelsize_x * xxR**2 + pixelsize_y * yyR**2))
    
    brange = torch.linspace(-(N+M)/2+1, (N+M)/2-1, N+M-1, dtype=torch.float32).to(device)
    xxB, yyB = torch.meshgrid(brange, brange, indexing='ij')
    xxB, yyB = xxB.to(device), yyB.to(device)
    B = torch.exp(-a * (pixelsize_x * xxB**2 + pixelsize_y * yyB**2))
    
    Bh = torch.fft.fft2(B)
    
    return A, Bh, C


def cztfunc1(datain, param, device):
    A, Bh, C = param
    N = A.shape[0]
    L = Bh.shape[0]
    M = C.shape[0]

    # print(A.device, Bh.device, C.device)
    # print(datain.device)
    
    Apad = torch.cat((A * datain / N, torch.zeros(datain.shape[:-1] + (L-N,), dtype=torch.complex64).to(device)), dim=-1)
    Apad = torch.cat((Apad, torch.zeros(Apad.shape[:-2] + (L-N, Apad.shape[-1]), dtype=torch.complex64).to(device)), dim=-2)
    
    Ah = torch.fft.fft2(Apad)
    cztout = torch.fft.ifft2(Ah * Bh / L)
    
    dataout = C * cztout[..., -M:, -M:]
    
    return dataout


def pupilRadius(M, pixelSize, wavelength, NA):
    k_max = NA / wavelength
    sampling = 1 / (M * pixelSize)
    return ( sampling / k_max) * (M / 2)

def convPSF(object_img, zernike, amplitude_mask_dm, k):
    
    psf = amplitude_mask_dm.squeeze().float() * torch.exp(1j * k * zernike)
    psf = torch.fft.fftshift(torch.fft.fft2(psf))
    psf = torch.abs(psf)**2
    w, h = psf.shape
    psf = psf[w//2 - 75:w//2 + 75, h//2 - 75:h//2 + 75]
    psf = psf / psf.sum()
    psf = psf.float()

    if object_img.ndim == 2:
        object_img = torch.unsqueeze(torch.unsqueeze(object_img, 0), 0)

    simulated_img = F.conv2d(object_img, psf.unsqueeze(0).unsqueeze(0), padding='same')
    simulated_img = simulated_img.squeeze().squeeze()
    return simulated_img




def czt(samplingsize, M, datain):
    krange = torch.linspace(-M/2+0.5, M/2-0.5, M, dtype=torch.float32)
    xxK, yyK = torch.meshgrid(krange, krange)

    xrange = torch.linspace(-M/2+0.5, M/2-0.5, M, dtype=torch.float32)
    xxS, yyS = torch.meshgrid(xrange, xrange, indexing='ij')

    a = 1j * torch.pi * samplingsize
    A = torch.exp(a * (xxK**2 + yyK**2))
    C = torch.exp(a * (xxS**2 + yyS**2))

    brange = torch.linspace(-(M+M)/2+1, (M+M)/2-1, M+M-1, dtype=torch.float32)
    xxB, yyB = torch.meshgrid(brange, brange, indexing='ij')
    # xxB, yyB = xxB.cuda(), yyB.cuda()
    B = torch.exp(-a * (xxB**2 + yyB**2))
    
    Bh = torch.fft.fft2(B)

    Apad = torch.cat((A * datain / M, torch.zeros(datain.shape[:-1] + (M - 1,), dtype=torch.complex64)), dim=-1)
    Apad = torch.cat((Apad, torch.zeros(Apad.shape[:-2] + (M - 1, Apad.shape[-1]), dtype=torch.complex64)), dim=-2)
    
    Ah = torch.fft.fft2(Apad)
    cztout = torch.fft.ifft2(Ah * Bh / (M+M-1))
    
    dataout = C * cztout[..., -M:, -M:]

    return dataout


# def zernike_pd_generation(M, pixelSize, wavelength, NA):
#     ratio = pupilRadius(M, pixelSize, wavelength, NA)
#     R = int( (1/ratio) * M)

#     zernikeR = genZern1(6, R)
#     zernike_crop_R = torch.zeros((R, R))
#     zernike_coords_R = torch.linspace(-1, 1, zernike_crop_R.shape[-1])
#     x_zernike_R, y_zernike_R = torch.meshgrid(zernike_coords_R, zernike_coords_R, indexing='ij')
#     zernike_crop_R[torch.where((x_zernike_R**2 + y_zernike_R**2) < 1)] = 1
#     zernikeR = zernike_crop_R * torch.from_numpy(zernikeR)

#     padding = (M - R) // 2
#     extra_padding = (M - R) % 2
#     zernikeR = F.pad(zernikeR, (padding, padding + extra_padding, padding, padding + extra_padding))
#     return zernikeR

def pd_amplitude_generation(M, pixelSize, wavelength, NA):
    ratio = pupilRadius(M, pixelSize, wavelength, NA)
    dm_crop_R = torch.zeros((M, M))
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, M), torch.linspace(-1, 1, M))
    dm_crop_R[ xx**2 + yy**2 < (1 / ratio)**2] = 1
    return dm_crop_R

##allows for the generation of a larger gaussian for the pupil plane --> for applying the STN
def gaussian_amplitude_enlarged(L, pixel_size, sigma):
    gaussian_coords = torch.arange(-5 * (L/2), 5 * (L/2) - pixel_size, pixel_size)
    FX1_gaussian, FY1_gaussian = torch.meshgrid(gaussian_coords, gaussian_coords, indexing='ij')
    gaussian = torch.exp(-1*(FX1_gaussian**2+FY1_gaussian**2)/(2 * (sigma)**2)).float()
    return gaussian

## L: physical width / height of the pupil plane in m
## M: number of pixels used to sample the pupil plane
## sigma: radius of the gaussian in m
def gaussian_amplitude(M, L, sigma):
    x, y = torch.meshgrid(torch.linspace(-L/2, L/2, M), torch.linspace(-L/2, L/2, M))
    dm_crop = torch.zeros((M, M))
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, M), torch.linspace(-1, 1, M))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    gaussian = torch.exp(-1*(x**2+y**2)/(2 * (sigma)**2)).float()
    return gaussian * dm_crop

def chirped_z_params(M, cameraPixelSize, wavelength, NA, n_imm, device):
    kpixelsize = 2.0*NA * n_imm/wavelength/M #### (maximum x,y frequency component in the pupil plane) / (number of pixels)
    params = prechirpz1(kpixelsize, cameraPixelSize, cameraPixelSize, M, M, device)
    return params


def chirped_z_pd_conv(object_img, phase_aberration,amplitude_mask, k, chirped_z_params, device):
    complex_amplitude = amplitude_mask.float() * torch.exp(1j * k * (phase_aberration))
    psf = cztfunc1(complex_amplitude, chirped_z_params, device)
    psf = torch.abs(psf)**2
    psf = psf / psf.sum()

    img_fft = torch.fft.fft2(object_img)
    otf = torch.fft.fft2(torch.fft.ifftshift(psf))
    imgBlurFFT = img_fft * otf
    img = torch.real(torch.fft.ifft2(imgBlurFFT))
    return img

def pd_psf_generation(amplitude_mask, phase_mask, k):
    # Generate the pupil function
    w, h = amplitude_mask.shape
    psf = amplitude_mask * torch.exp(1j * k * phase_mask)
    psf = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(psf)))
    psf = torch.abs(psf)**2
    # psf = psf[(w//2-100):(w//2+100), (h//2-100):(h//2+100)]
    psf = psf / psf.sum()
    return psf



## note to self: must for simulation code:
def chirped_z_convolution(object_img, zernike, amplitude_mask_dm, k, chirped_z_params, device):
    psf = amplitude_mask_dm.squeeze().float() * torch.exp(1j * k * zernike)
    psf = psf.to(device)
    psf = cztfunc1(psf, chirped_z_params, device)
    psf = torch.abs(psf)**2
    w, h = psf.shape
    psf = psf[w//2 - 100:w//2 + 100, h//2 - 100:h//2 + 100]
    psf = psf / psf.sum()
    psf = psf.float()
    
    if object_img.ndim == 2:
        object_img = torch.unsqueeze(torch.unsqueeze(object_img, 0), 0)

    simulated_img = F.conv2d(object_img, psf.unsqueeze(0).unsqueeze(0), padding='same')
    simulated_img = simulated_img.squeeze().squeeze()
    return simulated_img
