# GRAPHYCS

This repository contains the open-source code for the paper:

**"Graph-based modeling of optical system enables adaptive optics with self-calibration over large field of view"**


## About GRAPHYCS

Fluorescence microscopy is fundamentally limited by aberrations that degrade resolution and image quality. While adaptive optics can compensate for these distortions, existing approaches either require complex hardware or rely on idealized models, leading to suboptimal correction in real systems. Here we introduce GRAPHYCS, a computational framework that bridges the gap between models and physical systems through differentiable graph-based modeling with automatic self-calibration. Through simulations, GRAPHYCS achieves 94.5% improvement in wavefront accuracy and substantially better aberration-corrected image quality (PSNR: 38.07 dB vs. 23.93 dB) compared to a state-of-the-art phase-diversity method under system non-idealities. In real-world microscopy experiments, GRAPHYCS delivers 50.4%, 83.0%, and 75.9% improvement in BRISQUE, NIQE, and PIQE metrics for sample-induced aberration correction, while effectively handling spatially varying aberrations across fields of view exceeding 1 mm². GRAPHYCS enables high-resolution imaging across extended regions without additional hardware complexity, providing a practical solution for wide-area aberration correction.

---
## Dataset

You can download the dataset used for GRAPHYCS from https://zenodo.org/records/15421484.

## Reproducing Experimental Results

In order to download the data from the bash folder, download the data for the figures of our paper from Zenodo and place them in a Data folder. 

To reconstruct the results for simulations under ideal conditions, run the following:

```bash

python graphycs/GRAPHYCS_spatially_invariant_wf.py \
        --base_dir               "Results" \
        --exp_name               "ideal_simulated_no_self_calib" \
        --data_path              "Data/Figure2/Diversity_Images_Ideal.tif" \
        --zernike_coeff_path     "Data/Figure2/appliedCoeff.txt" \
        --epochs                 2001 \
        --use_affine_transform   1 \
        --use_learnable_scales   1 \
        --use_flat_field_correction 0 \
        --n_imm                  1.0 \
        --batch_size             12 \
        --lr                     1e-2 \
        --object_lr              2e-3 \
        --affine_transform_lr    1e-6 \
        --learnable_scale_lr     1e-6 \
        --fourier_loss_weight    7e-4 \
        --vis_frequency          100

## Requirements

While not specific requirements, the code was tested using the folowing versions of the Python packages and dependencies:

- **Python**: 3.11.10  
- **CUDA**: 12.4 (if using GPU)
- **PyTorch**: 2.2.1
- **torchvision**: 0.17.1
- **torchaudio**: 2.2.1
- **NumPy**: 1.26.4
- **SciPy**: 1.13.1
- **fft-conv-pytorch**: 1.2.0
- **matplotlib**: 3.8.4
- **scikit-image**: 0.23.2
- **tqdm**: 4.66.4

The list of Python packages and dependencies are specified in the [`requirements.txt`](requirements.txt) file.
