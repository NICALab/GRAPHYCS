import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils import *
from self_calibration import *
from torch.fft import fftshift, ifftshift, ifft2, fft2
from fft_conv_pytorch import fft_conv, FFTConv2d

dtype = torch.float32

class forward_model_lsm_variant(nn.Module):
    def __init__(self,
                 init_object,    # [H,W] tensor
                 phase_added_images,  # [N, H, W] tensor
                 num_patches_h=3,  # number of patches to divide the image into
                 num_patches_w=8,  # number of patches to divide the image into
                 n_max=14,          # max Zernike idx
                 psf_size=101,
                 M=512,              # pupil grid size
                 lmbda=0.513e-6, pixel_size=0.5343e-6,
                 n=1.33, NA=0.3, diversity_imgs_num=22, device='cuda',
                 use_affine_transform=False, use_scales=False, learn_motion=False, learn_activity=False):
        super().__init__()
        # --- store constants & precompute Zernikes, masks, Gaussian beam, etc. ---
        self.M = M
        self.psf_size = psf_size
        self.k = 2*np.pi*n/lmbda

        self.device = device

        # Zernike modes:
        _, zerns = zernike_pd_generation_higher_order(6, n_max, M, pixel_size, lmbda, NA)
        self.zernikes = zerns.to(device)  # [n_modes, M, M]

        # aperture mask & Gaussian amplitude pupil
        kx = torch.fft.fftshift(torch.fft.fftfreq(M, d=pixel_size))
        ky = torch.fft.fftshift(torch.fft.fftfreq(M, d=pixel_size))
        kxx, kyy = torch.meshgrid(kx, ky, indexing="xy")
        mask = ((kxx**2+kyy**2) <= (NA/lmbda)**2).float()
        self.mask = mask.to(device)  # [M,M]
        self.dm_aperture_mask = ((kxx**2+kyy**2) <= ((5.75 / 5.04) * NA/lmbda)**2).float()
        self.dm_aperture_mask = self.dm_aperture_mask.to(device)

        self.register_buffer('aperture', mask)
        Dxp = 10e-3
        x = torch.linspace(-Dxp/2, Dxp/2, M)
        y = x.clone()
        X,Y = torch.meshgrid(x,y,indexing="xy")
        sigma =  10e-3
        gauss = torch.exp(-(X**2+Y**2)/(2*sigma**2))
        self.gaussian = gauss.to(device)*mask.to(device)  # [M,M]

        # learnable parameters
        self.estimated_obj = nn.Parameter(init_object.clone().to(device)).float()  # [H,W]
        self.zernike_coef  = nn.Parameter(torch.zeros(n_max+1).to(device)).float()  # [n_max+1]
        self.varying_zernike_coef = nn.Parameter(torch.zeros((num_patches_h*num_patches_w, n_max + 1)).to(device)).float()  # [diversity_imgs_num, n_max+1]

        self.motion_factors = torch.zeros((diversity_imgs_num + 1, 2)).to(device).float()  # [diversity_imgs_num, 2]
        self.activity_factors = phase_added_images.clone().to(device) - init_object.clone().to(device).float()  # [N, H, W]


        H, W = init_object.shape
        self.obj_h, self.obj_w = H, W  # store original object size

        ## parameters for spatially varying convolution
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.patch_h = H // num_patches_h + psf_size - 1
        self.patch_w = W // num_patches_w + psf_size - 1
        self.stride_h = H // num_patches_h
        self.stride_w = W // num_patches_w
        self.pad_h = psf_size - 1
        self.pad_w = psf_size - 1


        if use_scales:
            astig_scale_factor = torch.tensor([0.0]).to(device)
            defocus_scale_factor = torch.tensor([0.0]).to(device)
            trefoil_scale_factor = torch.tensor([0.0]).to(device)
            coma_scale_factor = torch.tensor([0.0]).to(device)
            tetrafoil_scale_factor = torch.tensor([0.0]).to(device)
            secondary_astig_scale_factor = torch.tensor([0.0]).to(device)
            spherical_aberr_scale_factor = torch.tensor([0.0]).to(device)

            self.scales = nn.ParameterList([
                nn.Parameter(astig_scale_factor),
                nn.Parameter(defocus_scale_factor),
                nn.Parameter(trefoil_scale_factor),
                nn.Parameter(coma_scale_factor),
                nn.Parameter(tetrafoil_scale_factor),
                nn.Parameter(secondary_astig_scale_factor),
                nn.Parameter(spherical_aberr_scale_factor)
            ]).to(device)

        if use_affine_transform:
            # e.g. amplitude_scale_x, offset_x, rotation_angle, grid_adjust...
            self.amp_x = nn.Parameter(torch.ones(1).to(device))
            self.amp_y = nn.Parameter(torch.ones(1).to(device))
            self.off_x = nn.Parameter(torch.zeros(1).to(device))
            self.off_y = nn.Parameter(torch.zeros(1).to(device))
            self.rot   = nn.Parameter(torch.zeros(1).to(device))
            self.grid  = nn.Parameter(torch.ones(1).to(device))
        if learn_motion:
            self.motion_factors = nn.Parameter(self.motion_factors)
        if learn_activity:
            self.activity_factors = nn.Parameter(self.activity_factors)


        self.use_affine_transform = use_affine_transform
        self.use_scales = use_scales
        self.learn_motion = learn_motion
        self.learn_activity = learn_activity

    def forward(self, added_coeffs, indices):
        est_phase = torch.einsum("i,ijk->jk", self.zernike_coef, self.zernikes)
        spatially_varying_phase = torch.einsum("ni,ijk->njk", self.varying_zernike_coef, self.zernikes)
        est_phase = est_phase.unsqueeze(0) + spatially_varying_phase 

        if self.use_scales:
            base = torch.ones(added_coeffs.shape[-1]).to(self.device)  # [n_max+1]
            filler_zero = torch.tensor([0.0]).to(self.device)
            scale_factors = torch.cat([filler_zero, filler_zero, filler_zero, self.scales[0], self.scales[1], self.scales[0], self.scales[2], self.scales[3], self.scales[3], self.scales[2], self.scales[4], self.scales[5], self.scales[6], self.scales[5], self.scales[4]]).to(self.device)
            scale_factors = base + scale_factors
            if added_coeffs.ndim == 1:
                added_coeffs = added_coeffs.unsqueeze(0)
            added_coeffs = added_coeffs * scale_factors.repeat(added_coeffs.shape[0], 1)  # [N, n_max
   

        add_phase = torch.einsum('ni,ijk->njk', added_coeffs, self.zernikes)
        B, _, _ = add_phase.shape
        N, _, _ = est_phase.shape

        # apply STN to the added phase if needed
        if self.use_affine_transform:
            add_phase = rotation_transform(add_phase, self.rot, device=self.device)
            if add_phase.ndim == 2:
                add_phase = add_phase.unsqueeze(0)

        ## important: add across the batch dimension
        total_phase = est_phase.unsqueeze(0) + add_phase.unsqueeze(1)  # [B, N, M, M]

        if self.use_affine_transform:
            total_phase = scale_transform_batched(total_phase, self.grid, self.grid, device=self.device)
        amp = self.gaussian
        if self.use_affine_transform:
            amp = scale_transform(amp, self.grid, self.grid, device=self.device)
            amp = scale_transform(amp, self.amp_x, self.amp_y, device=self.device)
            amp = shift_transform(amp, self.off_x, self.off_y, device=self.device)
        amp = self.dm_aperture_mask * amp
        amp = amp.unsqueeze(0).unsqueeze(0).expand(B, -1, -1,-1)  # [B,1, M, M]

        # pupil plane field
        pupil_field = amp * torch.exp(1j * self.k * 1e-6 * total_phase)
        if pupil_field.ndim == 2:
            pupil_field = pupil_field.unsqueeze(0)

        psf_full = fftshift(fft2(pupil_field, dim=(-2, -1)), dim=(-2, -1))

        psf_int = torch.abs(psf_full)**2
        c = self.M//2; r=self.psf_size//2
        psf = psf_int[..., c-r:c+r+1, c-r:c+r+1]
        psf_k = psf / psf.sum(dim=(-2, -1), keepdim=True)  # normalize PSF

        obj = self.estimated_obj  # [1,1,H,W]

        ### apply motion and activity factors if needed
        if self.learn_activity:
            if obj.ndim == 2:
                obj = obj.unsqueeze(0)
            obj = obj + self.activity_factors[indices]

        if obj.ndim == 2:
            obj = obj.unsqueeze(0).unsqueeze(0)
        elif obj.ndim == 3:
            obj = obj.unsqueeze(0)

        obj_pad = F.pad(obj, (self.psf_size//2, self.psf_size//2, self.psf_size//2, self.psf_size//2), mode='constant', value=0)
        obj_pad = obj_pad.unfold(2, self.patch_h, self.stride_h).unfold(3, self.patch_w, self.stride_w)
        obj_pad = obj_pad.squeeze(0).flatten(1,2)


        B, N, H, W = obj_pad.shape
        obj_patches_batched = obj_pad.reshape(B * N, 1, H, W)  # (1, B*N, H, W)
        obj_patches_batched = obj_patches_batched.transpose(0, 1)  # (B*N, 1, H, W)

        psf_kernels = psf_k.reshape(B * N, 1, self.psf_size, self.psf_size)  # (B*N, 1, psf_size, psf_size)

        output = fft_conv(obj_patches_batched, psf_kernels, None, 0, groups=(B * N))  # (B*N, 1, H+psf_size-1, W+psf_size-1)
        output = output.reshape(B, N, self.stride_h, self.stride_w)  # (B, N, H, W)
        aberrated_imgs = output.flatten(2,3)
        aberrated_imgs = aberrated_imgs.permute(0,2,1) ## result
        aberrated_imgs_combined = F.fold(aberrated_imgs, (self.obj_h, self.obj_w), (self.stride_h, self.stride_w), stride=(self.stride_h, self.stride_w))    

        if self.learn_motion:
            aberrated_imgs_combined = shift_transform_batch(aberrated_imgs_combined, self.motion_factors[indices, 0], self.motion_factors[indices, 1], device=self.device)
            _, _, w, h = aberrated_imgs_combined.shape

            aberrated_imgs_combined = aberrated_imgs_combined.squeeze()
            aberrated_imgs_combined = aberrated_imgs_combined[..., 10:w-10, 10:h-10]

        return aberrated_imgs_combined, psf_k, amp[0,...].squeeze()

