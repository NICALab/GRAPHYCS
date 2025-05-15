import os
import torch
import numpy as np
import skimage.io as skio
import torch.nn.functional as F
import argparse


# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import time
import csv
from torch.fft import fftshift, ifftshift, ifft2, fft2
from self_calibration import *
import time
from overlap_add_save import *
from utils import *


if __name__=="__main__":
    # torch.autograd.set_detect_anomaly(True)
    ### INIT

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='single_patch_03112025', type=str)
    parser.add_argument("--self_calibration_lr", default=1e-5, type=float) # 1e-3
    parser.add_argument("--lr", default=1e-3, type=float) # 1e-3
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--data_path", default="D:\\CAO_Data_2025\\20250310_overlap_save_fft\\Data\\CALIB_sys_defocus_250226_Lymph_SV_S1R1_patch2\\CALIB_sys_defocus_250226_Lymph_SV_S1R1_Z5_0.tif", type=str) # CALIB_250107_SomaGCaMP7f2dpf_S1R1_crop CALIB_250107_Lymph_S1R1_crop
    parser.add_argument("--zernike_coeff_path", default="D:\\CAO_Data_2025\\20250304_spatially_varying_test\\Data\\CALIB_sys_defocus_250226_Lymph_SV_S1R1\\appliedCoeff.txt", type=str) #2umU_seed33_30imgs_wScale
    parser.add_argument("--gt_coeff_path", default="None", type=str) # Image_phasediversity_241224_gt Image_phasediversity_241219_gt2
    parser.add_argument("--base_dir", default="D:\\CAO_Data_2025\\20250310_overlap_save_fft", type=str)

    
    parser.add_argument("--gt_wavefront_path", default="None", type=str) # Image_phasediversity_241224_gt Image_phasediversity_241219_gt2
    parser.add_argument("--psf_path", default="None", type=str) # Image_phasediversity_241224_gt Image_phasediversity_241219_gt2


    ### set options for using Self-Calibration and learnable scales
    parser.add_argument("--use_self_calibration", default=1, type=int) # 1, 0
    parser.add_argument("--use_learnable_scales", default=0, type=int) # 1,0
    parser.add_argument("--use_scheduler", default=0, type=int)


    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--nuclear_reg_weight", default=1e-8, type=float)  # 4.61e-7  2.44e-7 1e-9
    parser.add_argument("--fourier_loss_weight", default=1e-8, type=float)  # 4.61e-7  2.44e-7 1e-9

    parser.add_argument('--n_max', type=int, default=14, help='maximum order of the zernike polynomials in the ANSI indexing scheme')
    parser.add_argument('--n_max_estimated', type=int, default=21, help='maximum order of the zernike polynomials in the ANSI indexing scheme estimated by the algorithm')



    parser.add_argument("--NA", default=0.8, type=float)
    parser.add_argument("--camera_pixel_size", default=0.27e-6, type=float) #0.65 0.534323613 0.534323613
    parser.add_argument("--wavelength", default=0.513e-6, type=float) # 0.5 513e 
    parser.add_argument("--n_imm", default=1.33, type=float)
    parser.add_argument("--gaussian_sigma", default=20.0e-3, type=float)


    parser.add_argument("--avg_background", default=0.0, type=float)

    parser.add_argument("--gt_amplitude_scale_x", default=1.0, type=float)
    parser.add_argument("--gt_amplitude_scale_y", default=1.0, type=float)
    parser.add_argument("--gt_offset_x", default=-0.0, type=float)
    parser.add_argument("--gt_offset_y", default=0, type=float)
    parser.add_argument("--gt_tilt", default=0, type=float)

    parser.add_argument("--psf_size", default=101, type=int)


    parser.add_argument("--use_stopping_condition", default=0, type=int)
    parser.add_argument("--use_L1_loss", default=1, type=int)
    parser.add_argument("--img_num_factor", default=1, type=int)

    parser.add_argument("--use_2photon", default=0, type=int)


    
    
    opt = parser.parse_args()
    exp_name = opt.exp_name
    learning_rate = opt.lr
    self_calib_param_lr = opt.self_calibration_lr
    torch.manual_seed(opt.seed)

    print("Self-Calibration Usage: ", opt.use_self_calibration)


    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(opt.seed)


    n_epochs = opt.epochs


    base_dir = opt.base_dir
    os.makedirs(base_dir, exist_ok=True)

    os.makedirs(base_dir + "/{}".format(exp_name), mode=0o755,exist_ok=True)
    os.makedirs(base_dir + "/{}/phase_estimations".format(exp_name), mode=0o755,exist_ok=True)
    os.makedirs(base_dir + "/{}/found_objects".format(exp_name), mode=0o755,exist_ok=True)
    os.makedirs(base_dir + "/{}/amplitudes".format(exp_name), mode=0o755,exist_ok=True)
    os.makedirs(base_dir + "/{}/zernikes_saved".format(exp_name), mode=0o755,exist_ok=True)
    os.makedirs(base_dir + "/{}/final_results".format(exp_name), mode=0o755,exist_ok=True)
    if opt.use_self_calibration:
        os.makedirs(base_dir + "/{}/affine_transforms".format(exp_name), mode=0o755,exist_ok=True)
    if opt.use_learnable_scales:
        os.makedirs(base_dir + "/{}/learnable_scales".format(exp_name), mode=0o755,exist_ok=True)
    


    ### logging data: recon_loss, fourier_loss, offset, scale



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ## parameter initialization:
    # n = 1.33  # refractive index of immersion medium
    wavelength = opt.wavelength # wavelength in [m]
    k = 2 * np.pi * opt.n_imm / wavelength  # wavenumber
    num_apert = opt.NA
    Dxp = 20e-3  # exit pupil size in [m] 
    camera_pixel_size = opt.camera_pixel_size  # camera pixel size in [m]

    M = 255 #511

    ## generate zernike polynomials in ansi indices:
    max_zernike_idx_ansi = opt.n_max
    _, zernikes = zernike_pd_generation_higher_order(6, max_zernike_idx_ansi, M, camera_pixel_size, wavelength, num_apert)
    zernikes = zernikes.to(device)

    maxSpatialFreq = ((num_apert) / wavelength) #/ (1 / (2 * cameraPixelSize))
    maxSpatialFreqRatio = maxSpatialFreq / (1 / (2 * camera_pixel_size))

    ### generation of spatial frequency in the coordinate space
    kx, ky = torch.fft.fftfreq(M, d = (camera_pixel_size)), torch.fft.fftfreq(M, d = (camera_pixel_size))
    kx, ky = torch.fft.fftshift(kx), torch.fft.fftshift(ky)

    kxx, kyy = torch.meshgrid(kx, ky)
    kxx, kyy = kxx.to(device), kyy.to(device)

    beam_crop = (kxx**2 + kyy**2) <= maxSpatialFreq**2
    dm_crop = (kxx**2 + kyy**2) <= (maxSpatialFreq * (5.75 / 5.04))**2

    beam_crop = beam_crop.float()
    dm_crop = dm_crop.float()


    ####### Settings for the Chirped Z transform: --> for visualizing the psf with higher frequency

    ### initialize chirped-z params such that a much smaller pixel size gets used:
    kpixelsize = 2.0*num_apert /wavelength/M #### (maximum x,y frequency component in the pupil plane) / (number of pixels)
    params = prechirpz1(kpixelsize, 0.21e-6, 0.21e-6, M, M, device)



    ## generate a gaussian amplitude function:
    x, y = torch.meshgrid(torch.linspace(-Dxp/2, Dxp/2, M), torch.linspace(-Dxp/2, Dxp/2, M))
    sigma = opt.gaussian_sigma
    # gaussian = beam_crop * torch.exp(-1*(x**2+y**2)/(2 * (sigma)**2)).float().to(device)
    gaussian = beam_crop * torch.exp(-1*(x**2+y**2)/(2 * (sigma)**2)).float().to(device)

    padding_w_ = M // 2
    padding_h_ = M // 2

    ## padding goes backwards: in the form of (left, right, top, bottom, front, back)
    dm_crop_padded = torch.nn.functional.pad(dm_crop, (padding_h_, padding_h_, padding_w_, padding_w_), 'constant', 0)
    beam_crop_padded = torch.nn.functional.pad(beam_crop, (padding_h_, padding_h_, padding_w_, padding_w_), 'constant', 0)
    gaussian_padded = torch.nn.functional.pad(gaussian, (padding_h_, padding_h_, padding_w_, padding_w_), 'constant', 0)



    ## turn gt tilt to scale factors:
    gt_scale_x = torch.tensor([1.0]).to(device)
    angle = torch.deg2rad(torch.tensor([opt.gt_tilt]))
    gt_scale_y = 1 / torch.cos(angle)
    gt_scale_y = gt_scale_y.to(device)


    ## Todo: read all of the coefficients as is, not removing the first two of the ansi coefficients

    coeffs = np.loadtxt(opt.zernike_coeff_path, delimiter=' ')
    print(coeffs.shape)
    print(opt.n_max_estimated)
    coeffs = torch.tensor(coeffs, dtype=torch.float32).to(device)
    
    if opt.gt_coeff_path != 'None':
        gt_coefficients = np.loadtxt(opt.gt_coeff_path, delimiter=' ')##torch.tensor(coeffs[0], dtype=torch.float32).to(device)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        print(gt_coefficients.shape)
        ground_truth_underlying_phase = torch.einsum("i,ijk->jk", 1e-6 * gt_coefficients, zernikes)
    else: ### if no gt coefficients 
        gt_coefficients = np.zeros(coeffs[0].shape)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        ground_truth_underlying_phase = torch.einsum("i,ijk->jk", 1e-6 * gt_coefficients, zernikes)



    if opt.gt_wavefront_path != 'None': ## for the case that the wavefront is created using smoothened random noise
        ground_truth_underlying_phase = skio.imread(opt.gt_wavefront_path).astype(np.float32)
        ground_truth_underlying_phase = ground_truth_underlying_phase[0, :, :]
        ground_truth_underlying_phase = 1e-6 * torch.from_numpy(ground_truth_underlying_phase).to(device)
    


    ### important: phase-added images and added phases need to be in the same order
    phase_added_images = skio.imread(opt.data_path).astype(np.float32)
    avg_background = np.min(phase_added_images[phase_added_images > 0])
    phase_added_images = np.maximum(phase_added_images - avg_background, np.zeros_like(phase_added_images))
    phase_added_images = phase_added_images / np.max(phase_added_images)
    phase_added_images = np.squeeze(phase_added_images)

    phase_added_images = phase_added_images[::opt.img_num_factor]
    coeffs = coeffs[::opt.img_num_factor]
    
    ##first image has the ground truth image with the underlying aberrations
    original_img = phase_added_images[0]
    print("original img", original_img.shape)
    img_w, img_h = original_img.shape

     ## reshape the image for spatially varying convolution:
    ks = opt.psf_size
    pad_size = ks -1
    pad_w, pad_h = pad_size, pad_size
    ## divide using overlap_save method
   
    ### phase added images INCLUDES the original, unaberrated image
    original_img, phase_added_images = torch.from_numpy(original_img).to(device), torch.from_numpy(phase_added_images).to(device)

 
    ### Important: the units for the coefficients are in micrometers, not meters
    found_zernike_coef = torch.zeros(opt.n_max + 1, dtype=torch.float32).to(device)
    
    found_object = original_img.clone().float()
    img_energy = torch.sum(original_img.float())
    found_object = found_object.requires_grad_().to(device)

    amplitude_scale_x = torch.tensor([1.0]).to(device)
    amplitude_scale_y = torch.tensor([1.0]).to(device)
    offset_x = torch.tensor([0.0]).to(device)
    offset_y = torch.tensor([0.0]).to(device)

    grid_adjustment = torch.tensor([1.0]).to(device)

    rotation_angle = torch.tensor([0.0]).to(device)
    self_calib_params = [amplitude_scale_x, amplitude_scale_y, offset_x, offset_y, rotation_angle, grid_adjustment]

    astig_scale_factor = torch.tensor([0.0]).to(device)
    defocus_scale_factor = torch.tensor([0.0]).to(device)
    trefoil_scale_factor = torch.tensor([0.0]).to(device)
    coma_scale_factor = torch.tensor([0.0]).to(device)
    tetrafoil_scale_factor = torch.tensor([0.0]).to(device)
    secondary_astig_scale_factor = torch.tensor([0.0]).to(device)
    spherical_aberr_scale_factor = torch.tensor([0.0]).to(device)
    scale_factors_list = [astig_scale_factor, defocus_scale_factor, trefoil_scale_factor, coma_scale_factor, tetrafoil_scale_factor, secondary_astig_scale_factor, spherical_aberr_scale_factor]


    

    if (opt.use_self_calibration != 1) and (opt.use_learnable_scales != 1):
        print("Using Neither Self-Calibration on Pupil/Wavefront nor Learnable Scales")
        optimizer = torch.optim.Adam([found_zernike_coef, found_object], lr=learning_rate)
    elif (opt.use_self_calibration == 1) and (opt.use_learnable_scales != 1):
        print("Using Self-Calibration on Pupil/Wavefront but NOT using Learnable Scales")
        for param in self_calib_params:
            param.requires_grad_()
        optimizer = torch.optim.Adam([{'params': found_zernike_coef, 'lr': 1*learning_rate}, 
                                  {'params': found_object, 'lr':  1*learning_rate},
                                  {'params': amplitude_scale_x, 'lr': self_calib_param_lr},
                                  {'params': amplitude_scale_y, 'lr': self_calib_param_lr},
                                  {'params': offset_x, 'lr': self_calib_param_lr}, # 0.1
                                  {'params': offset_y, 'lr': self_calib_param_lr}, # 0.1
                                  {'params': rotation_angle, 'lr': self_calib_param_lr},
                                  {'params': grid_adjustment, 'lr': self_calib_param_lr}], lr = learning_rate) # 0.5
    elif (opt.use_self_calibration != 1)  and (opt.use_learnable_scales == 1):
        print("NOT using Self-Calibration on Pupil/Wavefront but using Learnable Scales")
        for scale_factor in scale_factors_list:
            scale_factor.requires_grad_()
        optimizer = torch.optim.Adam([{'params': found_zernike_coef, 'lr': learning_rate}, 
                                  {'params': found_object, 'lr': learning_rate},
                                  {'params': astig_scale_factor, 'lr': 0.1 * learning_rate},
                                  {'params': defocus_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': trefoil_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': coma_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': tetrafoil_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': secondary_astig_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': spherical_aberr_scale_factor, 'lr': 0.1 * learning_rate}], lr = learning_rate)
    elif (opt.use_self_calibration == 1)  and (opt.use_learnable_scales == 1):
        print("Using Self-Calibration on Pupil/Wavefront and Learnable Scales")
        for scale_factor in scale_factors_list:
            scale_factor.requires_grad_()
        for param in self_calib_params:
            param.requires_grad_()
        optimizer = torch.optim.Adam([{'params': found_zernike_coef, 'lr': learning_rate}, 
                                  {'params': found_object, 'lr': learning_rate},
                                  {'params': amplitude_scale_x, 'lr': self_calib_param_lr},
                                  {'params': amplitude_scale_y, 'lr': self_calib_param_lr},
                                  {'params': offset_x, 'lr': self_calib_param_lr}, # 0.1
                                  {'params': offset_y, 'lr': self_calib_param_lr}, # 0.1
                                  {'params': rotation_angle, 'lr': self_calib_param_lr},
                                  {'params': grid_adjustment, 'lr': self_calib_param_lr},
                                  {'params': astig_scale_factor, 'lr': 0.1 * learning_rate},
                                  {'params': defocus_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': trefoil_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': coma_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': tetrafoil_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': secondary_astig_scale_factor, 'lr': 0.1 * learning_rate},
                                    {'params': spherical_aberr_scale_factor, 'lr': 0.1 * learning_rate}], lr = learning_rate)

    mse_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    
    if opt.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=0)


    ## losses and correlation errors:
    train_loss_ep = []
    coefficient_error_ep = []
    phase_error_ep = []

    scale_factor_x_list = []
    scale_factor_y_list = []
    offset_x_list = []
    offset_y_list = []
    pixel_size_factor_list = []

    ## wavefront rms error:
    wavefront_rms_error = []
    coeff_rms_error = []

    phase_rms_error = []
    fourier_loss_ep = []
    zernikes_varying_L1_ep = []


    found_object_stack = []
    iPSF_stack = []

    amplitudes_stack = []

    ## don't use the first two coefficients (tip/tilt)
    mask = torch.ones_like(found_zernike_coef).to(device)

    mask[:3] = 0
    if opt.n_max_estimated < len(mask):
        mask[opt.n_max_estimated+1:] = 0

    
    max_intensity = torch.sum(original_img)
    print("max intensity: ", max_intensity)
    torch.autograd.set_detect_anomaly(True)

    times_rms_below_0_1 = []
    times_rms_below_0_05 = []
    rms_below_0_1_flag = False
    rms_below_0_05_flag = False
    start_time = time.time()  # Record the start time
    stop = 0

    for epoch_idx in range(n_epochs):
        epoch_loss = 0
        epoch_L1 = 0
        epoch_fourier_loss = 0
        epoch_zernikes_varying_L1 = 0

        ## check the last input used for training:

        generated_imgs = []


        for pair_idx in tqdm(range(len(coeffs)), desc="epoch {} optimization".format(epoch_idx)):
            optimizer.zero_grad()
            found_object.requires_grad_()
            found_zernike_coef.requires_grad_()


            ## create and apply learnable scales IF the option is enabled
            if opt.use_learnable_scales:
                for scale_factor in scale_factors_list:
                    scale_factor.requires_grad_()
                filler_zero = torch.tensor([0.0]).to(device)
                scale_factors = torch.cat([filler_zero, filler_zero, filler_zero, astig_scale_factor, defocus_scale_factor, astig_scale_factor, trefoil_scale_factor, coma_scale_factor, coma_scale_factor, trefoil_scale_factor, tetrafoil_scale_factor, secondary_astig_scale_factor, spherical_aberr_scale_factor, secondary_astig_scale_factor, tetrafoil_scale_factor]).to(device)
                scale_factors = torch.tensor([1.0]).to(device) + scale_factors
                if scale_factors.numel() < found_zernike_coef.numel():
                    pad_size = found_zernike_coef.numel() - scale_factors.numel()
                    scale_factors = torch.cat([
                        scale_factors,
                        torch.ones(pad_size, device=device)
                    ])
            

            random_added_coeff = coeffs[pair_idx]
            random_added_coeff = 1*random_added_coeff

            assert len(random_added_coeff) == len(found_zernike_coef)


            ## ### Beam Phase is defined on the Fourier Plane
            estimated_phase = torch.einsum('i, ikl -> kl', 1e-6 * found_zernike_coef, zernikes)

            ## define the beam phase on the DM plane along with the applied phase from the DM
            if opt.use_learnable_scales:
                applied_phase = torch.einsum("i,ijk->jk", 1e-6 * scale_factors * random_added_coeff, zernikes)
            else:
                applied_phase = torch.einsum("i,ijk->jk", 1e-6 * random_added_coeff, zernikes)


            ####### Apply Self Calibration: ######## ########
            rotated_applied_phase = rotation_transform(applied_phase, rotation_angle, device)
            total_phases = estimated_phase + rotated_applied_phase

            beam_amplitude_scaled = scale_transform(gaussian, grid_adjustment, grid_adjustment, device)
            beam_amplitude_scaled = scale_transform(beam_amplitude_scaled, amplitude_scale_x, amplitude_scale_y, device)
            beam_amplitude_scaled_shifted = shift_transform(beam_amplitude_scaled, offset_x, offset_y, device)
            beam_amplitude_scaled_shifted_cropped = dm_crop * beam_amplitude_scaled_shifted

            beam_phases_scaled = scale_transform(total_phases, grid_adjustment, grid_adjustment, device)
          
            amplitude = beam_amplitude_scaled_shifted


            psf = fftshift(ifft2(ifftshift(amplitude * torch.exp(1j * k * beam_phases_scaled))))

            if opt.use_2photon == 1:
                psf = torch.abs(psf)**4
            else:
                psf = torch.abs(psf)**2
            psf = psf[M//2 - opt.psf_size//2:M//2 + opt.psf_size//2 + 1, M//2 - opt.psf_size//2:M//2 + opt.psf_size//2 +1]
            psf = psf / psf.sum()

            ## save all of the PSFs for the different objects
            if (pair_idx == 0) and (epoch_idx % 30 == 0):

                iPSF_stack.append(psf.detach().cpu())

           
            object = F.pad(found_object, (pad_w//2, pad_w//2, pad_h//2, pad_h//2), mode='constant', value=0)
            aberrated_img = fft_conv(object.unsqueeze(0).unsqueeze(0), torch.flip(psf.unsqueeze(0).unsqueeze(0), [2,3]), None, 0)
            


            generated_imgs.append(aberrated_img.squeeze().detach().cpu().numpy())
            
            observed_img = phase_added_images[pair_idx]
            
            recon_observed_img_FT = torch.real(torch.fft.fft2(aberrated_img))
            observed_img_FT = torch.real(torch.fft.fft2(observed_img)) #### ifft2

            if opt.use_L1_loss == 1:
                Fourier_loss = L1_loss(recon_observed_img_FT.squeeze(), observed_img_FT)
                recon_loss = L1_loss(aberrated_img.squeeze(), observed_img) #+ 0.1*Fourier_loss # L1_loss mse_loss
            else:
                Fourier_loss = mse_loss(recon_observed_img_FT.squeeze(), observed_img_FT)
                recon_loss = mse_loss(aberrated_img.squeeze(), observed_img) #+ 0.1*Fourier_loss # L1_loss mse_loss


            loss = recon_loss + opt.fourier_loss_weight * Fourier_loss

            loss.backward()

     
            found_zernike_coef.grad.mul_(mask)
            
            optimizer.step()

            # del object
            if opt.use_scheduler:
                scheduler.step()
            with torch.no_grad():
                found_object.data.clamp_(min=torch.tensor([0]).to(device))
                for scale_factor in scale_factors_list:
                    scale_factor.data.clamp_(min=torch.tensor([-1.0]).to(device))


            epoch_loss += recon_loss.item()
            epoch_fourier_loss += Fourier_loss.item()
            
        current_time = time.time()
       
        
        ############ Logging data: ###############

        coefficient_error = np.abs(found_zernike_coef.detach().cpu().numpy() - gt_coefficients.cpu().numpy())
        avg_coeff_error = np.mean(coefficient_error)
        avg_phase_error = np.mean(np.abs((estimated_phase.detach().cpu().numpy() - ground_truth_underlying_phase.cpu().numpy()))) / wavelength
        # avg_phase_error = np.mean(np.abs((estimated_phase.cpu().numpy() - ground_truth_underlying_phase.cpu().numpy()))) / wavelength


        # wavefront_rms = np.sqrt(np.mean((estimated_phase.detach().cpu().numpy() - ground_truth_underlying_phase.cpu().numpy())**2)) / wavelength
        wavefront_rms = np.sqrt(np.mean((estimated_phase.detach().cpu().numpy() - ground_truth_underlying_phase.cpu().numpy())**2)) / 1e-6

        coeff_rms = np.sqrt(np.mean((found_zernike_coef.detach().cpu().numpy() - gt_coefficients.cpu().numpy())**2))

        wavefront_rms_error.append(wavefront_rms)
        coeff_rms_error.append(coeff_rms)

        coefficient_error_ep.append(avg_coeff_error)
        phase_error_ep.append(avg_phase_error)

        scale_factor_x_list.append(amplitude_scale_x.item())
        scale_factor_y_list.append(amplitude_scale_y.item())
        offset_x_list.append(offset_x.item())
        offset_y_list.append(offset_y.item())
        pixel_size_factor_list.append(grid_adjustment.item())

        
        train_loss_ep.append(epoch_loss)
        fourier_loss_ep.append(epoch_fourier_loss)
        # nuclear_loss_ep.append(epoch_nuclear_loss)
        zernikes_varying_L1_ep.append(epoch_zernikes_varying_L1)
        

        # Check if the wavefront RMS falls below 0.1
        if not rms_below_0_1_flag and wavefront_rms < 0.1:
            time_below_0_1 = current_time - start_time
            times_rms_below_0_1.append(time_below_0_1)
            rms_below_0_1_flag = True

        # Check if the wavefront RMS falls below 0.05
        if not rms_below_0_05_flag and wavefront_rms < 0.05:
            time_below_0_05 = current_time - start_time
            times_rms_below_0_05.append(time_below_0_05)
            rms_below_0_05_flag = True

        # if (opt.use_stopping_condition == 1) and epoch_idx > 90:
        if epoch_idx > 90:
            rms_delta = wavefront_rms_error[-1] - wavefront_rms_error[-2]
                ## stopping condition: less than 1 percent change in the wavefront RMS
            stop = np.abs(rms_delta) < 0.01 * wavefront_rms_error[-1]
                    

            
        if epoch_idx == opt.epochs - 1 or (opt.use_stopping_condition and stop):

            if not rms_below_0_1_flag or not rms_below_0_05_flag:
                with open(f"{base_dir}/{exp_name}/wavefront_rms_times.txt", "a") as f:
                    if not rms_below_0_1_flag:
                        f.write("Wavefront RMS never fell below 0.1.\n")
                    if not rms_below_0_05_flag:
                        f.write("Wavefront RMS never fell below 0.05.\n")

            stop_time = current_time - start_time
            np.savetxt(f"{base_dir}/{exp_name}/stop_time.txt", np.array([stop_time]))
            

            ## save all of the logging data as .npy files:
            np.save(base_dir + "/{}/final_results/train_loss.npy".format(exp_name), np.array(train_loss_ep))
            np.save(base_dir + "/{}/final_results/fourier_loss_ep.npy".format(exp_name), np.array(fourier_loss_ep))
            np.save(base_dir + "/{}/final_results/wavefront_rms.npy".format(exp_name), np.array(wavefront_rms_error))

            wavefront_final = estimated_phase.detach().cpu().numpy() / 1e-6
            residual_wavefront = wavefront_final - (ground_truth_underlying_phase.cpu().numpy() / 1e-6)
            residual_rms = np.sqrt(np.mean(residual_wavefront**2))

            gt_phase_um = ground_truth_underlying_phase / 1e-6

            vmin = min(wavefront_final.min(), gt_phase_um.min())
            vmax = max(wavefront_final.max(), gt_phase_um.max())

            beam_scaled = scale_transform(beam_crop, amplitude_scale_x, amplitude_scale_y, device)
            beam_scaled_shifted = shift_transform(beam_scaled, offset_x, offset_y, device)
            phase_estimated_vis = beam_amplitude_scaled_shifted * estimated_phase

            beam_gt = scale_transform(beam_crop, amplitude_scale_x, amplitude_scale_y, device)
            beam_gt = shift_transform(beam_gt, offset_x, offset_y, device)
            phase_gt = beam_gt * ground_truth_underlying_phase

            residual_wavefront_ = (phase_estimated_vis.detach().cpu().numpy() / 1e-6) - (phase_gt.detach().cpu().numpy() / 1e-6)
            residual_rms_ = np.sqrt(np.mean(residual_wavefront_**2))

            ### save the final wavefront and the residual wavefront:
            skio.imsave(base_dir + "/{}/final_results/estimated_wavefront.tif".format(exp_name), wavefront_final)
            skio.imsave(base_dir + "/{}/final_results/ground_truth_wavefront.tif".format(exp_name), gt_phase_um.cpu().numpy())
            skio.imsave(base_dir + "/{}/final_results/residual_wavefront.tif".format(exp_name), residual_wavefront)

            ## save the amplitude scaled and shifted image:
            beam_amplitude_scaled_shifted_save = beam_amplitude_scaled_shifted.detach().cpu().numpy() * dm_crop.cpu().numpy()
            ##calculate the gt beam amplitude:
            beam_crop_padded = torch.nn.functional.pad(dm_crop, (padding_h_, padding_h_, padding_w_, padding_w_), 'constant', 0)
            w_, h_ = beam_crop_padded.shape
            beam_gt = shift_transform(beam_crop_padded, torch.tensor([opt.gt_offset_x]).float().to(device), torch.tensor([opt.gt_offset_y]).float().to(device), device)
            beam_gt = scale_transform(beam_gt, gt_scale_x, gt_scale_y, device)
            beam_gt = beam_gt[w_//2 - M//2:w_//2 + M//2+1, h_//2 - M//2:h_//2 + M//2+1]
            plt.figure()
            plt.imshow(beam_amplitude_scaled_shifted_save, cmap='gray')
            plt.imshow(beam_gt.cpu().numpy(), cmap='gray', alpha=0.5)
            plt.title("Amplitude Scaled and Shifted")
            plt.savefig(base_dir + "/{}/final_results/amplitude_scaled_shifted.png".format(exp_name))
            plt.clf()

            ## save the gt stn params:
            gt_self_calib_params = [opt.gt_amplitude_scale_x, opt.gt_amplitude_scale_y, opt.gt_offset_x, opt.gt_offset_y]
            np.savetxt(base_dir + "/{}/final_results/gt_self_calib_params.txt".format(exp_name), gt_self_calib_params)

            ## save all stn params:
            params_save = [amplitude_scale_x.item(), amplitude_scale_y.item(), offset_x.item(), offset_y.item(), rotation_angle.item(), grid_adjustment.item()]
            np.savetxt(base_dir + "/{}/final_results/self_calib_params.txt".format(exp_name), params_save)
            rms_stats = [residual_rms, residual_rms_]
            np.savetxt(base_dir + "/{}/final_results/rms_stats.txt".format(exp_name), rms_stats)


            ## save the final wavefront and the residual wavefront
            np.savetxt(base_dir + "/{}/final_results/estimated_wavefront.txt".format(exp_name), wavefront_final)
            skio.imsave(base_dir + "/{}/final_results/estimated_wavefront.tif".format(exp_name), wavefront_final)
            residual_wavefront = wavefront_final - (ground_truth_underlying_phase.cpu().numpy() / 1e-6)

            ## save the shifted amplitude:
            np.savetxt(base_dir + "/{}/final_results/amplitude_scaled_shifted.txt".format(exp_name), beam_amplitude_scaled_shifted_save)
            skio.imsave(base_dir + "/{}/final_results/amplitude_scaled_shifted.tif".format(exp_name), beam_amplitude_scaled_shifted_save)

            if opt.use_stopping_condition and stop:
                break


        if epoch_idx % 10 == 1:
            found_object_stack_save = torch.stack(found_object_stack, 0).numpy()
            skio.imsave(base_dir + "/{}/found_objects/found_object.tif".format(exp_name, epoch_idx), found_object_stack_save)

            amplitudes_stack_save = torch.stack(amplitudes_stack, 0).numpy()
            skio.imsave(base_dir + "/{}/amplitudes/amplitude.tif".format(exp_name, epoch_idx), amplitudes_stack_save)

            ## keep on saving time:
            time_at_epoch = current_time - start_time
          

        
        if epoch_idx % 10 == 0:

            np.savetxt(base_dir + "/{}/zernikes_saved/found_zernikes_{}.txt".format(exp_name, epoch_idx), np.array(found_zernike_coef.detach().cpu().numpy()))
            
            found_object_stack.append(found_object.detach().cpu())
            amplitudes_stack.append(amplitude.detach().cpu())
            
            ###### Saving the loss plots: training loss and correlation between predicted and gt phase
            plt.figure(figsize=(10,6))
            plt.subplot(1,2,1)
            plt.plot(train_loss_ep)
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title("train loss", fontdict={'fontsize': 12})
            
            plt.subplot(1,2,2)
            plt.plot(fourier_loss_ep)
            plt.title("Fourier Loss", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("loss")

            plt.tight_layout()
            plt.savefig(base_dir + "/{}/loss_curve.png".format(exp_name))
            plt.clf()
            plt.close()

            plt.figure(figsize=(10,6))
            plt.subplot(2,2,1)
            plt.plot(scale_factor_x_list)
            plt.xlabel("epochs")
            plt.ylabel("scale factor x")
            plt.title("Scale Factor X for Gaussian Amplitude", fontdict={'fontsize': 12})
            plt.subplot(2,2,2)
            plt.plot(scale_factor_y_list)
            plt.title("Scale Factor Y for Gaussian Amplitude", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("scale factor y")
            plt.subplot(2,2,3)
            plt.plot(offset_x_list)
            plt.title("Offset X for Gaussian Amplitude", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("offset x")
            plt.subplot(2,2,4)
            plt.plot(offset_y_list)
            plt.title("Offset Y for Gaussian Amplitude", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("offset y")
            plt.tight_layout()
            plt.savefig(base_dir + "/{}/affine_transforms.png".format(exp_name))
            plt.clf()
            plt.close()

            plt.figure(figsize=(10,6))
            plt.subplot(111)
            plt.plot(pixel_size_factor_list)
            plt.xlabel("epochs")
            plt.ylabel("grid adjustment factor")
            plt.title("Grid Adjustment Factor", fontdict={'fontsize': 12})
            plt.tight_layout()
            plt.savefig(base_dir + "/{}/grid_adjustment_factor.png".format(exp_name))
            plt.clf()
            plt.close()

            params_save = [amplitude_scale_x.item(), amplitude_scale_y.item(), offset_x.item(), offset_y.item(), rotation_angle.item(), grid_adjustment.item()]
            np.savetxt(base_dir + "/{}/self_calib_params_{}.txt".format(exp_name, epoch_idx), params_save)
            

            
            if opt.use_self_calibration:
                
                shift_params = [offset_x.item(), offset_y.item()]
                # shift_params = [um_conversion_factor * i for i in shift_params]
                shift_labels = ['Shift X of Amplitude', 'Shift Y of Amplitude']

                scale_rotation_params = [amplitude_scale_x.item(), amplitude_scale_y.item(), rotation_angle.item(), grid_adjustment.item()]
                scale_rotation_labels = ['Scale X', 'Scale Y', 'Rotation Angle', 'Grid Adjustment Factor']

                np.savetxt(base_dir + "/{}/affine_transforms/affine_transform_params_{}.txt".format(exp_name, epoch_idx), np.hstack((np.array([shift_params]), np.array([scale_rotation_params]))))

            if opt.use_learnable_scales:
                scale_factor_vis = torch.cat([astig_scale_factor, defocus_scale_factor, trefoil_scale_factor, coma_scale_factor, tetrafoil_scale_factor, secondary_astig_scale_factor, spherical_aberr_scale_factor]).detach().cpu().numpy()
                scale_factor_vis = np.ones_like(scale_factor_vis) + scale_factor_vis
                scale_factor_labels = ['Astigmatism', 'Defocus', 'Trefoil', 'Coma', 'Tetrafoil', 'Secondary Astigmatism', 'Spherical Aberration']

                fig3, axs3 = plt.subplots(1,1)
                cax3 = axs3.bar(np.arange(1, len(scale_factor_vis)+1, 1), scale_factor_vis, width=0.4, color='r')
                axs3.set_title('Learnable Scale Factors', fontdict={'fontsize': 10})
                axs3.set_xticks(np.arange(1, len(scale_factor_vis)+1, 1))
                axs3.set_xticklabels(scale_factor_labels, rotation=30, ha='right', fontsize=8)
                axs3.set_xlabel('Zernike Mode')
                axs3.set_ylabel('scale')
                fig3.savefig(base_dir + "/{}/phase_estimations/learnable_scales_{}.jpg".format(exp_name, epoch_idx), dpi=300) 
                plt.clf()
                plt.close()

                np.savetxt(base_dir + "/{}/learnable_scales/learnable_scales_{}.txt".format(exp_name, epoch_idx), scale_factor_vis)





             ## overlay the coefficients:

            vmin = min(estimated_phase.min(), ground_truth_underlying_phase.min())
            vmax = max(estimated_phase.max(), ground_truth_underlying_phase.max())
            fig, axs = plt.subplots(1, 2)
		
            cax1 = axs[0].imshow(estimated_phase.detach().cpu().numpy(), vmin = vmin, vmax = vmax, cmap=cm.gray)
            im_ratio = estimated_phase.shape[0] / estimated_phase.shape[1]
            fig.colorbar(cax1, ax=axs[0], fraction=im_ratio*0.047)
            axs[0].set_title('Predicted Wavefront', fontdict={'fontsize': 10})
            axs[0].axis('off')
            axs[1].bar(np.arange(1, zernikes.shape[0]+1, 1) - 0.2, found_zernike_coef.detach().cpu().numpy(), width=0.4, color='r', label='Predicted')
            predicted = found_zernike_coef.detach().cpu().numpy()
            max_val = np.max(predicted)
            plt.ylim(max_val * -1.1, max_val * 1.1)  
            axs[1].set_title('Zernike Coefficients, RMS: {:.3f} um'.format(coeff_rms), fontdict={'fontsize': 10})
            plt.tight_layout()  # Automatically adjusts subplot params for good fit
            fig.savefig(base_dir + "/{}/phase_estimations/{}.jpg".format(exp_name, epoch_idx), dpi=300) 
            plt.clf()
            plt.close()



            beam_crop_padded = torch.nn.functional.pad(dm_crop, (padding_h_, padding_h_, padding_w_, padding_w_), 'constant', 0)
            w_, h_ = beam_crop_padded.shape
            beam_shifted = shift_transform(beam_crop_padded, offset_x, offset_y, device)
            beam_amplitude_scaled_shifted = scale_transform(beam_shifted, amplitude_scale_x, amplitude_scale_y, device)
            beam_amplitude_scaled_shifted = beam_amplitude_scaled_shifted[w_//2 - M//2:w_//2 + M//2+1, h_//2 - M//2:h_//2 + M//2+1]
            phase_estimated_vis = beam_amplitude_scaled_shifted * estimated_phase

            vmin = phase_estimated_vis.min()
            vmax = phase_estimated_vis.max()

            psf_smaller_pixel_size = cztfunc1(amplitude.float() * torch.exp(1j * k * (estimated_phase)), params, device)
            psf_smaller_pixel_size = psf_smaller_pixel_size.squeeze().detach().cpu()
            psf_smaller_pixel_size = torch.abs(psf_smaller_pixel_size)**2
            psf_smaller_pixel_size = psf_smaller_pixel_size / psf_smaller_pixel_size.sum()
            psf_smaller_pixel_size = psf_smaller_pixel_size.numpy()



            plt.figure(figsize=(10,6))
            plt.subplot(121)
            plt.imshow(phase_estimated_vis.detach().cpu().numpy(), vmin = vmin, vmax = vmax, cmap='jet')
            plt.colorbar(shrink=0.4)
            plt.title('Estimated Phase', fontdict={'fontsize': 12})
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(psf_smaller_pixel_size, cmap='jet')
            plt.title('Estimated PSF', fontdict={'fontsize': 12})
            plt.axis('off')
            plt.colorbar(shrink=0.4)
            plt.savefig(base_dir + "/{}/phase_estimations/wavefront_vis_{}.jpg".format(exp_name, epoch_idx), dpi=300)
            plt.clf()
            plt.close()

