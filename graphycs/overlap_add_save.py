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


## implement with overlap save --> adjust pad and stride accordingly
def spatially_varying_forward_overlap_save_2photon(amplitude, phases, object, psf_size, stride_w, stride_h, k):

    # print(object.shape)
    img_w, img_h = object.squeeze().shape

    # print("DFDFJDHFJDHF")
    # print(phases.shape)
    psfs = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * phases.squeeze())), dim=(-2, -1)), dim=(-2, -1))
    psfs = torch.abs(psfs)**4
    w_, h_ = amplitude.squeeze().shape
    psfs = psfs[:, w_//2 - psf_size//2:w_//2 + psf_size//2 + 1, h_//2 - psf_size//2:h_//2 + psf_size//2 +1]
    psfs = psfs / psfs.sum((1,2), keepdim=True)
    # psfs = torch.flip(psfs, [1, 2])
    psfs = psfs.unsqueeze(1)
    psfs = torch.flip(psfs, [2, 3])


    patch_h = stride_h + psf_size - 1
    patch_w = stride_w + psf_size - 1
    pad_w, pad_h = psf_size - 1, psf_size - 1

    object = F.pad(object, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=0)
    # print("found_object.shape after padding", found_object.shape)
    object = object.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    # print("found_object.shape after unfolding", found_object.shape)
    object_patches = object.flatten(2,3).squeeze(1)
    

    output_patches = fft_conv(object_patches, psfs, None, 0, groups=psfs.shape[0])
    # output_patches = F.conv2d(object_patches, psfs, padding= 0, groups=psfs.shape[0])


    aberrated_imgs = output_patches.flatten(2,3)
    aberrated_imgs = aberrated_imgs.permute(0,2,1) ## result: (1, patch_h*patch_w, num_patches_h*num_patches_w)
    aberrated_imgs_combined = F.fold(aberrated_imgs, (img_w, img_h), (stride_h, stride_w), stride=(stride_h, stride_w))

    return psfs, aberrated_imgs_combined, object_patches, output_patches

## implement with overlap save --> adjust pad and stride accordingly
def spatially_varying_forward_overlap_save(amplitude, phases, object, psf_size, stride_w, stride_h, k):

    # print(object.shape)
    img_w, img_h = object.squeeze().shape

    # print("DFDFJDHFJDHF")
    # print(phases.shape)
    psfs = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * phases.squeeze())), dim=(-2, -1)), dim=(-2, -1))
    psfs = torch.abs(psfs)**2
    w_, h_ = amplitude.squeeze().shape
    psfs = psfs[:, w_//2 - psf_size//2:w_//2 + psf_size//2 + 1, h_//2 - psf_size//2:h_//2 + psf_size//2 +1]
    psfs = psfs / psfs.sum((1,2), keepdim=True)
    # psfs = torch.flip(psfs, [1, 2])
    psfs = psfs.unsqueeze(1)
    psfs = torch.flip(psfs, [2, 3])


    patch_h = stride_h + psf_size - 1
    patch_w = stride_w + psf_size - 1
    pad_w, pad_h = psf_size - 1, psf_size - 1

    object = F.pad(object, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=0)
    # print("found_object.shape after padding", found_object.shape)
    object = object.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    # print("found_object.shape after unfolding", found_object.shape)
    object_patches = object.flatten(2,3).squeeze(1)
    

    output_patches = fft_conv(object_patches, psfs, None, 0, groups=psfs.shape[0])
    # output_patches = F.conv2d(object_patches, psfs, padding= 0, groups=psfs.shape[0])


    aberrated_imgs = output_patches.flatten(2,3)
    aberrated_imgs = aberrated_imgs.permute(0,2,1) ## result: (1, patch_h*patch_w, num_patches_h*num_patches_w)
    aberrated_imgs_combined = F.fold(aberrated_imgs, (img_w, img_h), (stride_h, stride_w), stride=(stride_h, stride_w))

    return psfs, aberrated_imgs_combined, object_patches, output_patches


def spatially_varying_forward_overlap_add(amplitude, phases, object, psf_size, stride_w, stride_h, k):
    psfs = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * phases.squeeze())), dim=(-2, -1)), dim=(-2, -1))
    psfs = torch.abs(psfs)**2
    img_w, img_h = object.squeeze().shape
    w_, h_ = amplitude.squeeze().shape

    psfs = psfs[:, w_//2 - psf_size//2:w_//2 + psf_size//2 + 1, h_//2 - psf_size//2:h_//2 + psf_size//2 +1]
    psfs = psfs / psfs.sum((1,2), keepdim=True)
    # psfs = torch.flip(psfs, [1, 2])
    psfs = psfs.unsqueeze(1)
    psfs = torch.flip(psfs, [2, 3])


    # object = F.pad(object, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2), mode='constant', value=0)
    # print("found_object.shape after padding", found_object.shape)
    object = object.unfold(2, stride_h, stride_h).unfold(3, stride_w, stride_w)
    # print("found_object.shape after unfolding", found_object.shape)
    object = object.flatten(2,3).squeeze(1)
    object_patches = F.pad(object, (psf_size//2, psf_size//2, psf_size//2, psf_size//2), mode='constant', value=0)

    output_patches = fft_conv(object_patches, psfs, None, psf_size//2, groups=psfs.shape[0])
    # output_patches = F.conv2d(object_patches, psfs, padding=psf_size//2, groups=psfs.shape[0])


    aberrated_imgs = output_patches.flatten(2,3)

    aberrated_imgs = aberrated_imgs.permute(0,2,1)
    

    aberrated_imgs_combined = F.fold(aberrated_imgs, output_size = img_h + psf_size - 1, kernel_size = stride_h + psf_size - 1, stride = stride_h)
    w, h = aberrated_imgs_combined.shape[-2], aberrated_imgs_combined.shape[-1]
    aberrated_imgs_combined = aberrated_imgs_combined[..., psf_size//2:w - psf_size//2, psf_size//2:h - psf_size//2]

    return psfs, aberrated_imgs_combined, object_patches, output_patches


#### pad the kernel and the img to be the same length
#### kernel dimension: (num_patches_h * num_patches_w, kernel_size, kernel_size)
#### img dimension: (num_patches_h * num_patches_w, patch_h, patch_w)
def linear_conv2d_stack_same(img, kernel):
    num_patches, patch_h, patch_w = img.shape
    num_patches, psf_h, psf_w = kernel.shape

    # full conv 크기
    out_h = patch_h + psf_h - 1
    out_w = patch_w + psf_w - 1

    padded_patches = F.pad(
        img,  # => [num_patches,patch_h,patch_w]
        (0, out_w - patch_w, 0, out_h - patch_h),
        mode='constant', value=0
    )
    padded_psfs = F.pad(
        kernel,    # => [num_patches,psf_h,psf_w]
        (0, out_w - psf_w, 0, out_h - psf_h),
        mode='constant', value=0
    )

    padded_patches_fft = torch.fft.rfft2(padded_patches, dim=(-2, -1))
    padded_psfs_fft = torch.fft.rfft2(padded_psfs, dim=(-2, -1))
    padded_conv_fft = padded_patches_fft * padded_psfs_fft
    padded_conv = torch.fft.irfft2(padded_conv_fft, dim=(-2, -1))
    # padded_conv *= (out_h * out_w)
    conv_output = padded_conv[:, kernel.shape[-2]//2:kernel.shape[-2]//2 + img.shape[-2], kernel.shape[-1]//2:kernel.shape[-1]//2 + img.shape[-1]]


    return conv_output, padded_patches  # => (num_patches, patch_h, patch_w)


def linear_conv2d_same(img, kernel):
    img = img[(None,)*2]
    kernel = kernel[(None,)*2]

    print(img.shape, kernel.shape)


    kernel = torch.flip(kernel, dims=[2,3])
    img_padded = F.pad(img, (kernel.shape[-2], kernel.shape[-2], kernel.shape[-1], kernel.shape[-1]))
    conv_output = F.conv2d(img_padded, kernel, padding = 0)
    conv_output = conv_output[0, 0, kernel.shape[-2]//2:kernel.shape[-2]//2 + img.shape[-2], kernel.shape[-1]//2:kernel.shape[-1]//2 + img.shape[-1]]
    return conv_output.squeeze()



def overlap_save_conv2d_fft(patch_2d: torch.Tensor, psf_2d: torch.Tensor) -> torch.Tensor:
    """
    Overlap-Save 방식으로 2D FFT 합성곱을 하는 함수 예시.
    patch_2d.shape = (patch_h, patch_w)
    psf_2d.shape   = (psf_h, psf_w)
    반환 shape = (patch_h, patch_w)
    """
    patch_h, patch_w = patch_2d.shape
    psf_h, psf_w = psf_2d.shape

    # full conv 크기
    out_h = patch_h + psf_h - 1
    out_w = patch_w + psf_w - 1

    # 동일한 크기로 zero-padding
    padded_patch = F.pad(
        patch_2d.unsqueeze(0).unsqueeze(0),  # => [1,1,patch_h,patch_w]
        (0, out_w - patch_w, 0, out_h - patch_h),
        mode='constant', value=0
    )
    padded_psf = F.pad(
        psf_2d.unsqueeze(0).unsqueeze(0),    # => [1,1,psf_h,psf_w]
        (0, out_w - psf_w, 0, out_h - psf_h),
        mode='constant', value=0
    )

    # FFT -> 곱 -> IFFT
    freq_patch = torch.fft.fft2(padded_patch, dim=(-2, -1))
    freq_psf   = torch.fft.fft2(padded_psf,   dim=(-2, -1))
    freq_conv  = freq_patch * freq_psf
    conv_full  = torch.fft.ifft2(freq_conv, dim=(-2, -1))
    conv_full  = torch.real(conv_full)

    # 유효영역 (patch_h, patch_w) 슬라이스
    result = conv_full[:, :, 0:patch_h, 0:patch_w]
    return result.squeeze(0).squeeze(0)  # => (patch_h, patch_w)

def overlap_add_conv2d_fft(patch_2d: torch.Tensor, psf_2d: torch.Tensor) -> torch.Tensor:
    """
    Overlap-Add 방식의 2D FFT 합성곱(단일 블록).
    patch_2d.shape = (patch_h, patch_w)
    psf_2d.shape   = (psf_h, psf_w)

    반환:
      full_conv.shape = (patch_h + psf_h - 1, patch_w + psf_w - 1)
    """
    patch_h, patch_w = patch_2d.shape
    psf_h, psf_w = psf_2d.shape

    # full conv 출력 크기
    out_h = patch_h + psf_h - 1
    out_w = patch_w + psf_w - 1

    # patch와 psf를 동일한 (out_h, out_w)로 zero-padding
    padded_patch = F.pad(
        patch_2d.unsqueeze(0).unsqueeze(0),  # => [1,1,patch_h,patch_w]
        (0, out_w - patch_w, 0, out_h - patch_h),
        mode='constant', value=0
    )
    padded_psf = F.pad(
        psf_2d.unsqueeze(0).unsqueeze(0),    # => [1,1,psf_h,psf_w]
        (0, out_w - psf_w, 0, out_h - psf_h),
        mode='constant', value=0
    )

    # 1) FFT
    freq_patch = torch.fft.fft2(padded_patch, dim=(-2, -1))
    freq_psf   = torch.fft.fft2(padded_psf,   dim=(-2, -1))

    # 2) 주파수영역에서 곱
    freq_conv = freq_patch * freq_psf

    # 3) IFFT + 실수부 취하기
    conv_full = torch.fft.ifft2(freq_conv, dim=(-2, -1))
    conv_full = torch.real(conv_full)

    # Overlap-Add는 “full convolution” 결과 전체를 사용 (No slicing).
    # 따라서 shape=(patch_h+psf_h-1, patch_w+psf_w-1)
    return conv_full.squeeze(0).squeeze(0)