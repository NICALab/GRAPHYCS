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
from fft_conv_pytorch import fft_conv, FFTConv2d
from torch.fft import fft2, ifft2, fftshift, ifftshift



def spatially_varying_forward_overlap_save_2photon(amplitude, phases, object, psf_size, stride_w, stride_h, k):

    img_w, img_h = object.squeeze().shape


    psfs = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * phases.squeeze())), dim=(-2, -1)), dim=(-2, -1))
    psfs = torch.abs(psfs)**4
    w_, h_ = amplitude.squeeze().shape
    psfs = psfs[:, w_//2 - psf_size//2:w_//2 + psf_size//2 + 1, h_//2 - psf_size//2:h_//2 + psf_size//2 +1]
    psfs = psfs / psfs.sum((1,2), keepdim=True)
    psfs = psfs.unsqueeze(1)
    psfs = torch.flip(psfs, [2, 3])


    patch_h = stride_h + psf_size - 1
    patch_w = stride_w + psf_size - 1
    pad_w, pad_h = psf_size - 1, psf_size - 1

    object = F.pad(object, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=0)
    object = object.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    object_patches = object.flatten(2,3).squeeze(1)
    

    output_patches = fft_conv(object_patches, psfs, None, 0, groups=psfs.shape[0])


    aberrated_imgs = output_patches.flatten(2,3)
    aberrated_imgs = aberrated_imgs.permute(0,2,1) ## result: (1, patch_h*patch_w, num_patches_h*num_patches_w)
    aberrated_imgs_combined = F.fold(aberrated_imgs, (img_w, img_h), (stride_h, stride_w), stride=(stride_h, stride_w))

    return psfs, aberrated_imgs_combined, object_patches, output_patches

def spatially_varying_forward_overlap_save(amplitude, phases, object, psf_size, stride_w, stride_h, k):

    img_w, img_h = object.squeeze().shape


    psfs = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * phases.squeeze())), dim=(-2, -1)), dim=(-2, -1))
    psfs = torch.abs(psfs)**2
    w_, h_ = amplitude.squeeze().shape
    psfs = psfs[:, w_//2 - psf_size//2:w_//2 + psf_size//2 + 1, h_//2 - psf_size//2:h_//2 + psf_size//2 +1]
    psfs = psfs / psfs.sum((1,2), keepdim=True)
    psfs = psfs.unsqueeze(1)
    psfs = torch.flip(psfs, [2, 3])


    patch_h = stride_h + psf_size - 1
    patch_w = stride_w + psf_size - 1
    pad_w, pad_h = psf_size - 1, psf_size - 1

    object = F.pad(object, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=0)
    object = object.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    object_patches = object.flatten(2,3).squeeze(1)
    

    output_patches = fft_conv(object_patches, psfs, None, 0, groups=psfs.shape[0])


    aberrated_imgs = output_patches.flatten(2,3)
    aberrated_imgs = aberrated_imgs.permute(0,2,1) ## result: (1, patch_h*patch_w, num_patches_h*num_patches_w)
    aberrated_imgs_combined = F.fold(aberrated_imgs, (img_w, img_h), (stride_h, stride_w), stride=(stride_h, stride_w))

    return psfs, aberrated_imgs_combined, object_patches, output_patches

