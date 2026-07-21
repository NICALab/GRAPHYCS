import os
import torch
import numpy as np
import skimage.io as skio
import torch.nn.functional as F
import argparse

# from tqdm import tqdm
import tqdm
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from utils import *
import csv
from torch.fft import fftshift, ifftshift, ifft2, fft2
from self_calibration import *
from forward_model_lsm import forward_model_lsm
from dset import BatchDatasetFromTIFFAndZernike

if __name__=="__main__":
    # torch.autograd.set_detect_anomaly(True)
    ### INIT

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='lsm_spatially_invariant_results', type=str)
    parser.add_argument("--lr", default=2e-2, type=float) 
    parser.add_argument("--object_lr", default=2e-2, type=float) 
    parser.add_argument("--affine_transform_lr", default=1e-4, type=float) 
    parser.add_argument("--learnable_scale_lr", default=1e-4, type=float) 
    parser.add_argument("--motion_param_lr", default=1e-3, type=float) 
    parser.add_argument("--activity_lr", default=1e-3, type=float) 
    parser.add_argument("--fourier_loss_weight", default=1e-3, type=float)

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--data_path", type=str) 
    parser.add_argument("--zernike_coeff_path", type=str) 
    parser.add_argument("--gt_coeff_path", default="None", type=str) 
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--gt_wavefront_path", default="None", type=str) 
    parser.add_argument("--psf_path", default="None", type=str) 

    ### set options for using self calibration and learnable scales
    parser.add_argument("--use_affine_transform", default=1, type=int)
    parser.add_argument("--use_learnable_scales", default=1, type=int)
    parser.add_argument("--use_motion_estimation", default=1, type=int)
    parser.add_argument("--use_activity_estimation", default=1, type=int)

    parser.add_argument("--use_scheduler", default=0, type=bool)


    parser.add_argument("--epochs", default=2100, type=int)
    parser.add_argument("--batch_size", default=22, type=int)
    parser.add_argument('--n_max', type=int, default=14, help='maximum order of the zernike polynomials in the ANSI indexing scheme')
    parser.add_argument('--n_max_estimated', type=int, default=14, help='maximum order of the zernike polynomials in the ANSI indexing scheme estimated by the algorithm')


    parser.add_argument("--NA", default=0.3, type=float)
    parser.add_argument("--camera_pixel_size", default=0.5343e-6, type=float)
    parser.add_argument("--wavelength", default=0.513e-6, type=float)
    parser.add_argument("--n_imm", default=1.33, type=float)
    parser.add_argument("--psf_size", default=101, type=int)

    parser.add_argument("--silence_tqdm", default=0, type=int)
    parser.add_argument("--vis_intermediates", default=1, type=int)
    parser.add_argument("--vis_frequency", default=200, type=int)
    parser.add_argument("--use_L1_loss", default=1, type=int)
    parser.add_argument("--sparsity_weight", default=8e-9, type=float)
    parser.add_argument("--use_batch_shuffling", default=0, type=int)


    opt = parser.parse_args()
    exp_name = opt.exp_name
    learning_rate = opt.lr
    torch.manual_seed(opt.seed)

    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(opt.seed)

    n_epochs = opt.epochs

    base_dir = opt.base_dir
    os.makedirs(base_dir, exist_ok=True)

    os.makedirs(base_dir + "/{}".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/phase_estimations".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/found_objects".format(exp_name), exist_ok=True)
    if opt.use_affine_transform:
        os.makedirs(base_dir + "/{}/affine_transforms".format(exp_name), exist_ok=True)
    if opt.use_learnable_scales:
        os.makedirs(base_dir + "/{}/learnable_scales".format(exp_name), exist_ok=True)
    if opt.use_motion_estimation:
        os.makedirs(base_dir + "/{}/motion_factors".format(exp_name), exist_ok=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wavelength = opt.wavelength # wavelength in [m]
    k = 2 * opt.n_imm * np.pi / wavelength  # wavenumber


    num_apert = opt.NA
    Dxp = 10e-3  # exit pupil size in [m] 
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

    ## turn gt tilt to scale factors:
    gt_scale_x = torch.tensor([1.0]).to(device)
    angle = torch.deg2rad(torch.tensor([opt.gt_tilt]))
    gt_scale_y = 1 / torch.cos(angle)
    gt_scale_y = gt_scale_y.to(device)

    if opt.gt_coeff_path != 'None':
        has_gt_coefficients = True
        gt_coefficients = np.loadtxt(opt.gt_coeff_path, delimiter=' ')##torch.tensor(coeffs[0], dtype=torch.float32).to(device)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        ground_truth_underlying_phase = torch.einsum("i,ijk->jk", 1e-6 * gt_coefficients, zernikes)
    else: ### if no gt coefficients 
        has_gt_coefficients = False
        gt_coefficients = np.zeros((max_zernike_idx_ansi + 1), dtype=np.float32)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        ground_truth_underlying_phase = torch.zeros((M, M), dtype=torch.float32).to(device)
    if opt.gt_wavefront_path != 'None': ## for the case that the wavefront is created using smoothened random noise
        ground_truth_underlying_phase = skio.imread(opt.gt_wavefront_path).astype(np.float32)
        ground_truth_underlying_phase = ground_truth_underlying_phase[0, :, :]
        ground_truth_underlying_phase = 1e-6 * torch.from_numpy(ground_truth_underlying_phase).to(device)
    
    

    learn_affine_transform = False
    learn_scales = False
    learn_motion = False
    learn_activity = False
    if opt.use_affine_transform == 1:
        learn_affine_transform = True
    if opt.use_learnable_scales == 1:
        learn_scales = True
    if opt.use_motion_estimation == 1:
        learn_motion = True
    if opt.use_activity_estimation == 1:
        learn_activity = True

    dset = BatchDatasetFromTIFFAndZernike(opt.data_path, opt.zernike_coeff_path)
    coeff_batches = dset.coeffs
    diversity_img_batches = torch.stack(dset.ys, axis=0)

    

    print("Coeff batches shape:", coeff_batches.shape
          , "Diversity image batches shape:", diversity_img_batches.shape)

    ##first image has the ground truth image with the underlying aberrations
    original_img = dset.ys[0]
    img_w, img_h = original_img.shape

    
    print("coeff batches shape after removing the first batch:", coeff_batches.shape
          , "diversity image batches shape after removing the first batch:", diversity_img_batches.shape)
    assert coeff_batches.shape[0] == diversity_img_batches.shape[0], "Coefficient batches and diversity image batches must have the same number of elements."
 
    ks = opt.psf_size
    pad_size = ks -1
    pad_w, pad_h = pad_size, pad_size

    model = forward_model_lsm(original_img, diversity_img_batches, 14, ks, M, wavelength, camera_pixel_size, opt.n_imm, opt.NA, len(coeff_batches) - 1, device, 
                                          learn_affine_transform, learn_scales, learn_motion, learn_activity)
    
    param_groups = [
        {'params': [model.estimated_obj], 'lr': opt.object_lr},
        {'params': [model.zernike_coef], 'lr': opt.lr},
        {'params': [model.motion_factors], 'lr': opt.motion_param_lr} if model.learn_motion else {},
        {'params': [model.activity_factors], 'lr': opt.activity_lr} if model.learn_activity else {},
        {'params': list(model.scales), 'lr': opt.learnable_scale_lr} if model.use_scales else {},

        {'params': [model.amp_x], 'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
        {'params': [model.amp_y], 'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
        {'params': [model.off_x], 'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
        {'params': [model.off_y], 'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
        {'params': [model.rot],   'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
        {'params': [model.grid],  'lr': opt.affine_transform_lr} if model.use_affine_transform else {},
    ]

    # Filter out empty dictionaries (which happen when condition is False)
    param_groups = [pg for pg in param_groups if pg]

    # Now construct the optimizer
    optimizer = torch.optim.Adam(param_groups, lr=1e-3)

    mse_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    if opt.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=0)
    

    ## losses and correlation errors:
    train_loss_ep = []
    fourier_loss_ep = []
    img_loss_ep = []
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
    zernikes_varying_L1_ep = []
    activity_sparsity_loss_ep = []



    found_object_stack = []
    iPSF_stack = []

    amplitudes_stack = []

    ## don't use the first two coefficients (tip/tilt)
    mask = torch.ones_like(model.zernike_coef).to(device)
    mask[:3] = 0.0

    total_it = 0
    silence_tqdm = False
    if opt.silence_tqdm == 1:
        silence_tqdm = True

    t = tqdm.trange(opt.epochs, disable=silence_tqdm)

    ############
    # Training loop
    times_rms_below_0_1 = []
    times_rms_below_0_05 = []
    rms_below_0_1_flag = False
    rms_below_0_05_flag = False
    t0 = time.time()


    _, w_d, h_d = diversity_img_batches.shape

    for epoch_idx in t:
        if opt.use_batch_shuffling == 1:
        #     # Shuffle the dataset indices at the start of each epoch
            idxs = torch.randperm(len(dset)).long().to(device)
        else:
        #     # Use a fixed order for the dataset indices
            idxs = torch.arange(len(dset)).long().to(device)

        optimizer.zero_grad()
        model.estimated_obj.requires_grad_()
        model.zernike_coef.requires_grad_()
        if model.learn_activity:
            model.activity_factors.requires_grad_()

        epoch_train_loss = 0.0
        epoch_image_loss = 0.0
        epoch_fourier_loss = 0.0
        epoch_sparsity_loss = 0.0


        for it in range(0, len(dset), opt.batch_size):
            idx = idxs[it:min(it + opt.batch_size, len(dset))]
            coeffs, y_batch = coeff_batches[idx].to(device), diversity_img_batches[idx].to(device)

            aberrated_imgs, psfs, amplitude_transformed = model(coeffs, idx)

            if model.learn_motion:
                y_batch_cropped = y_batch[..., 10:w_d-10, 10:h_d-10]  # crop the output to remove padding
            else:
                y_batch_cropped = y_batch


            recon_observed_imgs_FT = torch.real(torch.fft.fft2(aberrated_imgs, dim=(-2,-1)))
            observed_imgs_FT = torch.real(torch.fft.fft2(y_batch_cropped, dim=(-2,-1))) 

            if opt.use_L1_loss == 1:
                Fourier_loss = L1_loss(recon_observed_imgs_FT.squeeze(), observed_imgs_FT.squeeze())
                recon_loss = L1_loss(aberrated_imgs.squeeze(), y_batch_cropped.squeeze()) #+ 0.1*Fourier_loss # L1_loss mse_loss
            else:
                Fourier_loss = mse_loss(recon_observed_imgs_FT.squeeze(), observed_imgs_FT.squeeze())
                recon_loss = mse_loss(aberrated_imgs.squeeze(), y_batch_cropped.squeeze()) #+ 0.1*Fourier_loss # L1_loss mse_loss

            if model.learn_activity:
                sparsity_loss = torch.norm(model.activity_factors, p=1) / model.activity_factors.numel()
            else:
                sparsity_loss = torch.zeros((), device=device, dtype=recon_loss.dtype)

            epoch_image_loss += recon_loss.item()
            epoch_fourier_loss += Fourier_loss.item()
            loss = recon_loss + opt.fourier_loss_weight * Fourier_loss + opt.sparsity_weight * sparsity_loss
            epoch_train_loss += loss.item()
            epoch_sparsity_loss += sparsity_loss.item()


            loss.backward()

            model.zernike_coef.grad.mul_(mask)

            optimizer.step()
            if opt.use_scheduler:
                scheduler.step()
            model.eval()
            with torch.no_grad():
                model.estimated_obj.data.clamp_(min=torch.tensor([0]).to(device))
                if opt.use_learnable_scales:
                    for p in model.scales:
                        p.data.clamp_(min=-1.0)
                if model.learn_activity:
                    model.activity_factors.data.clamp_(min=0.0)
            del coeffs, y_batch, aberrated_imgs, psfs, recon_observed_imgs_FT, observed_imgs_FT


        current_time = time.time()
        

        coefficient_error = np.abs(model.zernike_coef.detach().cpu().numpy() - gt_coefficients.cpu().numpy())
        avg_coeff_error = np.mean(coefficient_error)

        estimated_phase = torch.einsum("i,ijk->jk", 1e-6 * model.zernike_coef, zernikes)
        avg_phase_error = np.mean(np.abs((estimated_phase.detach().cpu().numpy() - ground_truth_underlying_phase.cpu().numpy()))) / wavelength
        wavefront_rms = np.sqrt(np.mean((estimated_phase.detach().cpu().numpy() - ground_truth_underlying_phase.cpu().numpy())**2)) / 1e-6

        coeff_rms = np.sqrt(np.mean((model.zernike_coef.detach().cpu().numpy() - gt_coefficients.cpu().numpy())**2))

        wavefront_rms_error.append(wavefront_rms)
        coeff_rms_error.append(coeff_rms)



        coefficient_error_ep.append(avg_coeff_error)
        phase_error_ep.append(avg_phase_error)
        if opt.use_affine_transform:
            scale_factor_x_list.append(model.amp_x.item())
            scale_factor_y_list.append(model.amp_y.item())
            offset_x_list.append(model.off_x.item())
            offset_y_list.append(model.off_y.item())
            pixel_size_factor_list.append(model.grid.item())

        ## loss for a single batch
        train_loss_ep.append(recon_loss.item())
        fourier_loss_ep.append(Fourier_loss.item())
        activity_sparsity_loss_ep.append(sparsity_loss.item())


        # Check if the wavefront RMS falls below 0.1
        if not rms_below_0_1_flag and wavefront_rms < 0.1:
            time_below_0_1 = current_time - t0
            times_rms_below_0_1.append(time_below_0_1)
            rms_below_0_1_flag = True

        # Check if the wavefront RMS falls below 0.05
        if not rms_below_0_05_flag and wavefront_rms < 0.05:
            time_below_0_05 = current_time - t0
            times_rms_below_0_05.append(time_below_0_05)
            rms_below_0_05_flag = True

                    

            
        if epoch_idx == opt.epochs - 1 and model.learn_activity:
            save_tiff_stack(
                base_dir + "/{}/found_objects/final_dynamic_sample_component.tif".format(exp_name),
                model.activity_factors.detach().cpu().numpy())

        t.set_description(f"Epoch {epoch_idx+1}/{opt.epochs}, Loss: {epoch_train_loss:.4f}, Coeff Error: {avg_coeff_error:.4f}, Phase Error: {avg_phase_error:.4f}, Wavefront RMS: {wavefront_rms:.4f}")
        if epoch_idx % 100 == 0:
            append_tiff_slice(
                base_dir + "/{}/found_objects/estimated_object_stack_every_100_epochs.tif".format(exp_name),
                model.estimated_obj.detach().cpu().numpy(),
                reset=(epoch_idx == 0))

        if epoch_idx % opt.vis_frequency == 0:

            save_loss_curve_visualization(
                base_dir + "/{}/loss_curve.png".format(exp_name),
                [
                    ("Train loss", train_loss_ep),
                    ("Fourier loss", fourier_loss_ep),
                    ("Activity L1 norm", activity_sparsity_loss_ep),
                ])

            if opt.use_affine_transform:
                shift_params = [model.off_x.detach().item(), model.off_y.detach().item()]
                scale_rotation_params = [model.amp_x.item(), model.amp_y.item(), model.rot.item(), model.grid.item()]
                np.savetxt(base_dir + "/{}/affine_transforms/affine_transform_params_{}.txt".format(exp_name, epoch_idx),
                           np.hstack((np.array([shift_params]), np.array([scale_rotation_params]))))

            if opt.use_motion_estimation and model.learn_motion:
                np.savetxt(base_dir + "/{}/motion_factors/motion_factors_{}.txt".format(exp_name, epoch_idx),
                           model.motion_factors.detach().cpu().numpy())

            if opt.use_learnable_scales:
                scale_factor_vis = torch.cat([model.scales[0], model.scales[1], model.scales[2], model.scales[3], model.scales[4], model.scales[5], model.scales[6]]).detach().cpu().numpy()
                scale_factor_vis = np.ones_like(scale_factor_vis) + scale_factor_vis
                np.savetxt(base_dir + "/{}/learnable_scales/learnable_scales_{}.txt".format(exp_name, epoch_idx), scale_factor_vis)

            wrapped_phase = wrap_phase_minus_pi_pi_for_visualization(
                estimated_phase, wavelength, opt.n_imm, beam_crop)
            save_phase_map_visualization(
                base_dir + "/{}/phase_estimations/phase_epoch_{:06d}.png".format(exp_name, epoch_idx),
                wrapped_phase)
