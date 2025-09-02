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
from forward_model_wf import forward_model_wf
from dset import BatchDatasetFromTIFFAndZernike
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":
    ### INIT

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='wf_pancreas_invariant_results', type=str)
    parser.add_argument("--lr", default=1e-2, type=float) # 1e-3
    parser.add_argument("--object_lr", default=1e-2, type=float) # 1e-3
    parser.add_argument("--affine_transform_lr", default=1e-4, type=float) # 1e-3
    parser.add_argument("--learnable_scale_lr", default=1e-4, type=float) # 1e-3
    parser.add_argument("--fourier_loss_weight", default=5e-4, type=float)  # 4.61e-7  2.44e-7 1e-9


    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--data_path", type=str) 
    parser.add_argument("--zernike_coeff_path", type=str) 
    parser.add_argument("--gt_coeff_path", default="None", type=str) 
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--gt_wavefront_path", default="None", type=str) 
    parser.add_argument("--psf_path", default="None", type=str) 

    ### set options for using STN and learnable scales
    parser.add_argument("--use_affine_transform", default=1, type=int) # 1, 0
    parser.add_argument("--use_learnable_scales", default=1, type=int) # 1,0
    parser.add_argument("--use_motion_estimation", default=0, type=int) # 1,0
    parser.add_argument("--use_activity_estimation", default=0, type=int) # 1,0

    parser.add_argument("--use_scheduler", default=0, type=bool)


    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--batch_size", default=22, type=int)
    parser.add_argument('--n_max', type=int, default=14, help='maximum order of the zernike polynomials in the ANSI indexing scheme')
    parser.add_argument('--n_max_estimated', type=int, default=14, help='maximum order of the zernike polynomials in the ANSI indexing scheme estimated by the algorithm')

    parser.add_argument("--NA", default=0.3, type=float)
    parser.add_argument("--camera_pixel_size", default=0.5343e-6, type=float) #0.65 0.534323613 0.534323613
    parser.add_argument("--wavelength", default=0.513e-6, type=float) # 0.5 513e 
    parser.add_argument("--n_imm", default=1.33, type=float)

    parser.add_argument("--use_flat_field_correction", default=0, type=int)
    parser.add_argument("--illum_path", default="Data/Illumination_Profile.tif", type=str)
    parser.add_argument("--psf_size", default=101, type=int)

    parser.add_argument("--silence_tqdm", default=0, type=int)
    parser.add_argument("--vis_intermediates", default=1, type=int)
    parser.add_argument("--vis_frequency", default=200, type=int)
    parser.add_argument("--use_L1_loss", default=1, type=int) # 1, 0

    parser.add_argument("--use_batch_shuffling", default=1, type=int) # 1, 0


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
    os.makedirs(base_dir + "/{}/iPSFs".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/gradients".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/amplitudes".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/parameter_estimations".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/zernikes_saved".format(exp_name), exist_ok=True)
    os.makedirs(base_dir + "/{}/final_results".format(exp_name), exist_ok=True)
    if opt.use_affine_transform:
        os.makedirs(base_dir + "/{}/affine_transforms".format(exp_name), exist_ok=True)
    if opt.use_learnable_scales:
        os.makedirs(base_dir + "/{}/learnable_scales".format(exp_name), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #######################################

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


    if opt.gt_coeff_path != 'None':
        gt_coefficients = np.loadtxt(opt.gt_coeff_path, delimiter=' ')##torch.tensor(coeffs[0], dtype=torch.float32).to(device)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        ground_truth_underlying_phase = torch.einsum("i,ijk->jk", 1e-6 * gt_coefficients, zernikes)
    else: ### if no gt coefficients 
        gt_coefficients = np.zeros((max_zernike_idx_ansi + 1), dtype=np.float32)
        gt_coefficients = torch.tensor(gt_coefficients, dtype=torch.float32).to(device)
        ground_truth_underlying_phase = torch.zeros((M, M), dtype=torch.float32).to(device)
    if opt.gt_wavefront_path != 'None': ## for the case that the wavefront is created using smoothened random noise
        ground_truth_underlying_phase = skio.imread(opt.gt_wavefront_path).astype(np.float32)
        ground_truth_underlying_phase = ground_truth_underlying_phase[0, :, :]
        ground_truth_underlying_phase = 1e-6 * torch.from_numpy(ground_truth_underlying_phase).to(device)
    
    

    learn_affine_transform = False
    learn_scales = False

    if opt.use_affine_transform == 1:
        learn_affine_transform = True
    if opt.use_learnable_scales == 1:
        learn_scales = True

    dset = BatchDatasetFromTIFFAndZernike(opt.data_path, opt.zernike_coeff_path)
    # coeff_batches = torch.cat(dset.coeffs, axis=0).unsqueeze(1).to(device)
    coeff_batches = dset.coeffs.to(device)
    diversity_img_batches = torch.stack(dset.ys, axis=0).to(device)

    if opt.use_flat_field_correction == 1:
        illum_image = skio.imread(opt.illum_path).astype(np.float32)
        max_val_illum_image = np.max(illum_image)
        illum_image_normalized = illum_image / max_val_illum_image
        illum_image_normalized = torch.from_numpy(illum_image_normalized).to(device)



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

    model = forward_model_wf(original_img, 14, ks, M, wavelength, camera_pixel_size, opt.n_imm, opt.NA, len(coeff_batches) - 1, device, 
                                          learn_affine_transform, learn_scales)
    
    param_groups = [
        {'params': [model.estimated_obj], 'lr': opt.object_lr},
        {'params': [model.zernike_coef], 'lr': opt.lr},
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
    nuclear_loss_ep = []
    zernikes_varying_L1_ep = []


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

    generated_imgs = []


    for epoch_idx in t:
        if opt.use_batch_shuffling == 1:
            # Shuffle the dataset indices at the start of each epoch
            idxs = torch.randperm(len(dset)).long().to(device)
        else:
            # Use a fixed order for the dataset indices
            idxs = torch.arange(len(dset)).long().to(device)
        optimizer.zero_grad()
        model.estimated_obj.requires_grad_()
        model.zernike_coef.requires_grad_()

        epoch_train_loss = 0.0
        epoch_image_loss = 0.0
        epoch_fourier_loss = 0.0

        # print(idxs)

        for it in range(0, len(dset), opt.batch_size):
            # idx = idxs[it:it+opt.batch_size]
            idx = idxs[it:min(it + opt.batch_size, len(dset))]
            coeffs, y_batch = coeff_batches[idx], diversity_img_batches[idx]

            aberrated_imgs, psfs, amplitude_transformed = model(coeffs, idx)

            ### apply flat field correction:
            if opt.use_flat_field_correction == 1:
                aberrated_imgs =  aberrated_imgs * illum_image_normalized

            recon_observed_imgs_FT = torch.real(torch.fft.fft2(aberrated_imgs, dim=(-2,-1)))
            observed_imgs_FT = torch.real(torch.fft.fft2(y_batch, dim=(-2,-1))) #### ifft2

            if opt.use_L1_loss == 1:
                Fourier_loss = L1_loss(recon_observed_imgs_FT.squeeze(), observed_imgs_FT.squeeze())
                recon_loss = L1_loss(aberrated_imgs.squeeze(), y_batch.squeeze()) #+ 0.1*Fourier_loss # L1_loss mse_loss
            else:
                Fourier_loss = mse_loss(recon_observed_imgs_FT.squeeze(), observed_imgs_FT.squeeze())
                recon_loss = mse_loss(aberrated_imgs.squeeze(), y_batch.squeeze()) #+ 0.1*Fourier_loss # L1_loss mse_loss


            epoch_image_loss += recon_loss.item()
            epoch_fourier_loss += Fourier_loss.item()
            loss = recon_loss + opt.fourier_loss_weight * Fourier_loss
            epoch_train_loss += loss.item()

            if (epoch_idx % opt.vis_frequency) == 0 and it == 0:

                psfs_squeezed = psfs.squeeze().detach().cpu().numpy()
                if opt.vis_intermediates == 1:
                    model_outputs_squeezed = aberrated_imgs.squeeze().detach().cpu().numpy()
                    if model_outputs_squeezed.ndim == 3:
                        generated_imgs.append(model_outputs_squeezed[-1, :, :])
                    elif model_outputs_squeezed.ndim == 2:
                        generated_imgs.append(model_outputs_squeezed)
                found_object_stack.append(model.estimated_obj.detach().cpu().numpy())
                if psfs_squeezed.ndim == 3:
                    iPSF_stack.append(psfs_squeezed[-1, :,:])
                elif psfs_squeezed.ndim == 2:
                    iPSF_stack.append(psfs_squeezed)
                amplitudes_stack.append(amplitude_transformed.detach().cpu().numpy())
                if opt.vis_intermediates == 1:
                    skio.imsave(base_dir + "/{}/found_objects/generated_img_{}.tif".format(exp_name, epoch_idx), np.array(generated_imgs))

            loss.backward()

            model.zernike_coef.grad.mul_(mask)

            optimizer.step()
            if opt.use_scheduler:
                scheduler.step()
            with torch.no_grad():
                model.estimated_obj.data.clamp_(min=torch.tensor([0]).to(device))
                if opt.use_learnable_scales:
                    for p in model.scales:
                        p.data.clamp_(min=-1.0)
        
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
            offset_y_list.append(model.off_x.item())
            pixel_size_factor_list.append(model.grid.item())

        ## loss for a single batch
        train_loss_ep.append(recon_loss.item())
        fourier_loss_ep.append(Fourier_loss.item())

        if epoch_idx % opt.vis_frequency == 0:
            np.save(base_dir + "/{}/train_loss.npy".format(exp_name), np.array(train_loss_ep))
            np.save(base_dir + "/{}/fourier_loss.npy".format(exp_name), np.array(fourier_loss_ep))


            
        if epoch_idx == opt.epochs - 1:


            stop_time = current_time - t0
            np.savetxt(f"{base_dir}/{exp_name}/stop_time.txt", np.array([stop_time]))

            np.save(base_dir + "/{}/final_results/train_loss.npy".format(exp_name), np.array(train_loss_ep))
            np.save(base_dir + "/{}/final_results/fourier_loss_ep.npy".format(exp_name), np.array(fourier_loss_ep))
            np.save(base_dir + "/{}/final_results/nuclear_loss_ep.npy".format(exp_name), np.array(nuclear_loss_ep))
            np.save(base_dir + "/{}/final_results/zernikes_varying_L1_ep.npy".format(exp_name), np.array(zernikes_varying_L1_ep))
            np.save(base_dir + "/{}/final_results/wavefront_rms.npy".format(exp_name), np.array(wavefront_rms_error))
            with torch.no_grad():

                wavefront_final = estimated_phase.detach().cpu().numpy() / 1e-6
                residual_wavefront = wavefront_final - (ground_truth_underlying_phase.cpu().numpy() / 1e-6)
                residual_rms = np.sqrt(np.mean(residual_wavefront**2))

                gt_phase_um = ground_truth_underlying_phase / 1e-6

                vmin = min(wavefront_final.min(), gt_phase_um.min())
                vmax = max(wavefront_final.max(), gt_phase_um.max())

            ### save the final wavefront and the residual wavefront:
            skio.imsave(base_dir + "/{}/final_results/estimated_wavefront.tif".format(exp_name), wavefront_final)
            skio.imsave(base_dir + "/{}/final_results/ground_truth_wavefront.tif".format(exp_name), gt_phase_um.cpu().numpy())
            skio.imsave(base_dir + "/{}/final_results/residual_wavefront.tif".format(exp_name), residual_wavefront)

            ## save the final wavefront and the residual wavefront
            np.savetxt(base_dir + "/{}/final_results/estimated_wavefront.txt".format(exp_name), wavefront_final)
            skio.imsave(base_dir + "/{}/final_results/estimated_wavefront.tif".format(exp_name), wavefront_final)
            residual_wavefront = wavefront_final - (ground_truth_underlying_phase.cpu().numpy() / 1e-6)

     

        if epoch_idx % opt.vis_frequency == 1:
            found_object_stack_save = np.array(found_object_stack)
            skio.imsave(base_dir + "/{}/found_objects/found_object.tif".format(exp_name, epoch_idx), found_object_stack_save)
            iPSF_stack_save = np.array(iPSF_stack)
            skio.imsave(base_dir + "/{}/iPSFs/psf.tif".format(exp_name), iPSF_stack_save)
            skio.imsave(base_dir + "/{}/iPSFs/psf.tif".format(exp_name, epoch_idx), iPSF_stack_save)
            amplitudes_stack_save = np.array(amplitudes_stack)
            skio.imsave(base_dir + "/{}/amplitudes/amplitude.tif".format(exp_name, epoch_idx), amplitudes_stack_save)

            times_to_save = np.array([
                ["Time (s) when RMS < 0.1"] + times_rms_below_0_1,
                ["Time (s) when RMS < 0.05"] + times_rms_below_0_05
            ], dtype=object)

            np.savetxt(f"{base_dir}/{exp_name}/wavefront_rms_times.txt", times_to_save, fmt="%s")

            ## keep on saving time:
            time_at_epoch = current_time - t0
            np.savetxt(f"{base_dir}/{exp_name}/time_at_epoch_{epoch_idx}.txt", np.array([epoch_idx, time_at_epoch]))

        t.set_description(f"Epoch {epoch_idx+1}/{opt.epochs}, Loss: {epoch_train_loss:.4f}, Coeff Error: {avg_coeff_error:.4f}, Phase Error: {avg_phase_error:.4f}, Wavefront RMS: {wavefront_rms:.4f}")
        if epoch_idx % opt.vis_frequency == 0:

            np.savetxt(base_dir + "/{}/zernikes_saved/found_zernikes_{}.txt".format(exp_name, epoch_idx), np.array(model.zernike_coef.detach().cpu().numpy()))
            
            ###### Saving the loss plots: training loss and correlation between predicted and gt phase
            plt.figure(figsize=(10,6))
            plt.subplot(2,2,1)
            plt.plot(train_loss_ep)
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title("train loss", fontdict={'fontsize': 12})
            
            plt.subplot(2,2,2)
            plt.plot(fourier_loss_ep)
            plt.title("Fourier Loss", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("loss")

            plt.subplot(2,2,3)
            plt.plot(img_loss_ep)
            plt.title("Image Loss", fontdict={'fontsize': 12})
            plt.xlabel("epochs")
            plt.ylabel("loss")


            plt.tight_layout()
            plt.savefig(base_dir + "/{}/loss_curve.png".format(exp_name))
            plt.clf()
            plt.close()

            if opt.use_affine_transform:
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

            
            if opt.use_affine_transform:
                with torch.no_grad():
                    shift_params = [model.off_x.detach().item(), model.off_y.detach().item()]
                    # shift_params = [um_conversion_factor * i for i in shift_params]
                    shift_labels = ['Shift X of Amplitude', 'Shift Y of Amplitude']

                    scale_rotation_params = [model.amp_x.item(), model.amp_y.item(), model.rot.item(), model.grid.item()]
                    scale_rotation_labels = ['Scale X', 'Scale Y', 'Rotation Angle', 'Grid Adjustment Factor']

                    np.savetxt(base_dir + "/{}/affine_transforms/affine_transform_params_{}.txt".format(exp_name, epoch_idx), np.hstack((np.array([shift_params]), np.array([scale_rotation_params]))))




            if opt.use_learnable_scales:
                scale_factor_vis = torch.cat([model.scales[0], model.scales[1], model.scales[2], model.scales[3], model.scales[4], model.scales[5], model.scales[6]]).detach().cpu().numpy()
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
            fig, axs = plt.subplots(2, 2)
		
            cax1 = axs[0, 0].imshow(estimated_phase.detach().cpu().numpy(), vmin = vmin, vmax = vmax, cmap='jet')
            im_ratio = estimated_phase.shape[0] / estimated_phase.shape[1]
            fig.colorbar(cax1, ax=axs[0, 0], fraction=im_ratio*0.047)
            axs[0, 0].set_title('Predicted Wavefront', fontdict={'fontsize': 10})
            axs[0, 0].axis('off')
            axs[0, 1].bar(np.arange(1, zernikes.shape[0]+1, 1) - 0.2, model.zernike_coef.detach().cpu().numpy(), width=0.4, color='r', label='Predicted')
            axs[0, 1].bar(np.arange(1, zernikes.shape[0]+1, 1) + 0.2, gt_coefficients.cpu().numpy(),  width=0.4, color='#000000', label='Ground Truth')
            predicted = model.zernike_coef.detach().cpu().numpy()
            ground_truth = gt_coefficients.cpu().numpy()
            max_val = max(np.max(predicted), np.max(ground_truth))
            plt.ylim(max_val * -1.1, max_val * 1.1)  

            axs[0, 1].set_title('Zernike Coefficients, RMS: {:.3f} um'.format(coeff_rms), fontdict={'fontsize': 10})

            cax2 = axs[1, 0].imshow(np.squeeze(ground_truth_underlying_phase.cpu().numpy()), vmin = vmin, vmax = vmax, cmap='jet')
            im_ratio = ground_truth_underlying_phase.shape[0] / ground_truth_underlying_phase.shape[1]
            fig.colorbar(cax2, ax=axs[1, 0], fraction=im_ratio*0.047)
            axs[1, 0].set_title('Underlying Phase', fontdict={'fontsize': 10})
            axs[1, 0].axis('off')
            axs[1, 1].plot(wavefront_rms_error)
            axs[1, 1].set_title('Wavefront RMS Error: {:.3f} lmbda'.format(wavefront_rms), fontdict={'fontsize': 9})
            axs[1, 1].set_xlabel('epochs')
            axs[1, 1].set_ylabel('error (wavelengths)')
            
            plt.tight_layout()  # Automatically adjusts subplot params for good fit
            fig.savefig(base_dir + "/{}/phase_estimations/{}.jpg".format(exp_name, epoch_idx), dpi=300) 


            plt.clf()
            plt.close()




