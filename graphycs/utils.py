
import os

import matplotlib.pyplot as plt
import skimage.io as skio

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


##### The following code for zernike polynomial generation / noll and ANSI indexing is taken from the repository below:
##### https://github.com/ries-lab/uiPSF
"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     Heintzmann Lab, Friedrich-Schiller-University Jena, Germany

@author: Rainer Heintzmann, Sheng Liu, Jonas Hellgoth
"""

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



def genZernAnsi(order_max, n_max_ansi,xsz):
    zernikesNoll = genZern1(order_max,xsz)
    zernikesAnsi = np.zeros((len(zernikesNoll),xsz,xsz))
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
    

    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR


def pupilRadius(M, pixelSize, wavelength, NA):
    k_max = NA / wavelength
    sampling = 1 / (M * pixelSize)
    return ( sampling / k_max) * (M / 2)


def _as_numpy_for_visualization(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def wrap_phase_minus_pi_pi_for_visualization(phase_meters, wavelength, n_imm, aperture_mask=None):
    phase_np = _as_numpy_for_visualization(phase_meters)
    phase_radians = (2 * np.pi * n_imm / wavelength) * phase_np
    wrapped_phase = np.angle(np.exp(1j * phase_radians)).astype(np.float32)

    if aperture_mask is not None:
        mask_np = _as_numpy_for_visualization(aperture_mask) > 0
        if mask_np.shape == wrapped_phase.shape:
            wrapped_phase = wrapped_phase.copy()
            wrapped_phase[~mask_np] = np.nan

    return wrapped_phase


def save_zernike_phase_visualization(output_path, wrapped_phase, estimated_coeffs, gt_coeffs=None,
                                     phase_title="Estimated phase (-pi to pi)"):
    wrapped_phase = np.asarray(wrapped_phase, dtype=np.float32)
    estimated_coeffs = np.asarray(estimated_coeffs, dtype=np.float32).ravel()
    gt_coeffs = None if gt_coeffs is None else np.asarray(gt_coeffs, dtype=np.float32).ravel()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))

    phase_cmap = plt.get_cmap("jet").copy()
    phase_cmap.set_bad(color=(1, 1, 1, 0))
    im = axs[0].imshow(wrapped_phase, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axs[0].axis("off")
    axs[0].set_title(phase_title, fontdict={"fontsize": 10})
    cbar = fig.colorbar(im, ax=axs[0], shrink=0.78)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

    x = np.arange(1, len(estimated_coeffs) + 1)
    if gt_coeffs is not None and gt_coeffs.size == estimated_coeffs.size:
        axs[1].bar(x - 0.2, estimated_coeffs, width=0.4, color="r", label="Estimated")
        axs[1].bar(x + 0.2, gt_coeffs, width=0.4, color="k", label="Ground truth")
        max_abs = np.nanmax(np.abs(np.concatenate([estimated_coeffs, gt_coeffs])))
        axs[1].legend(fontsize=8)
    else:
        axs[1].bar(x, estimated_coeffs, width=0.55, color="r", label="Estimated")
        max_abs = np.nanmax(np.abs(estimated_coeffs))

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    axs[1].set_ylim(-1.1 * max_abs, 1.1 * max_abs)
    axs[1].set_title("Zernike coefficients", fontdict={"fontsize": 10})
    axs[1].set_xlabel("ANSI Zernike index")
    axs[1].set_ylabel("Coefficient (um)")
    axs[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_phase_map_visualization(output_path, wrapped_phase, title="Estimated phase (-pi to pi)"):
    phase_map = np.asarray(wrapped_phase, dtype=np.float32)
    phase_cmap = plt.get_cmap("jet").copy()
    phase_cmap.set_bad(color=(1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(phase_map, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    ax.axis("off")
    ax.set_title(title, fontdict={"fontsize": 10})
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def append_tiff_slice(output_path, image, reset=False):
    img = np.asarray(_as_numpy_for_visualization(image), dtype=np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and not reset:
        existing = skio.imread(output_path).astype(np.float32)
        if existing.ndim == img.ndim:
            existing_stack = existing[..., np.newaxis]
        elif existing.ndim == img.ndim + 1:
            existing_stack = existing
        else:
            raise ValueError(
                "Existing TIFF stack shape {} is incompatible with new slice shape {}.".format(
                    existing.shape, img.shape))
        stack = np.concatenate([existing_stack, img[..., np.newaxis]], axis=-1)
    else:
        stack = img[..., np.newaxis]
    skio.imsave(output_path, np.asarray(stack, dtype=np.float32))


def save_tiff_stack(output_path, stack):
    stack_np = np.asarray(_as_numpy_for_visualization(stack), dtype=np.float32)
    if stack_np.ndim == 3:
        stack_np = np.moveaxis(stack_np, 0, -1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    skio.imsave(output_path, stack_np)


def save_loss_curve_visualization(output_path, loss_series):
    if hasattr(loss_series, "items"):
        loss_iter = loss_series.items()
    else:
        loss_iter = loss_series

    visible = [(label, np.asarray(vals, dtype=float)) for label, vals in loss_iter if len(vals) > 0]
    if not visible:
        return

    n_plots = len(visible)
    n_cols = min(2, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.8 * n_rows), squeeze=False)

    for ax in axs.ravel():
        ax.axis("off")

    for ax, (label, vals) in zip(axs.ravel(), visible):
        ax.axis("on")
        ax.plot(vals)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label, fontdict={"fontsize": 12})
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_spatial_phase_grid_visualization(output_path, spatial_phase_grid_wrapped, title):
    phase_grid = np.asarray(spatial_phase_grid_wrapped, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 12))
    phase_cmap = plt.get_cmap("jet").copy()
    phase_cmap.set_bad(color=(1, 1, 1, 0))
    im = ax.imshow(phase_grid, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    ax.axis("off")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
