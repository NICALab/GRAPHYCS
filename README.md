# GRAPHYCS

This repository contains the open-source code for the paper:

**"Graph-based modeling of optical system enables adaptive optics with self-calibration over large field of view"**


## About GRAPHYCS

Fluorescence microscopy is fundamentally limited by aberrations that degrade resolution and image quality. While adaptive optics can compensate for these distortions, existing approaches either require complex hardware or rely on idealized models, leading to suboptimal correction in real systems. Here we introduce GRAPHYCS, a computational framework that bridges the gap between models and physical systems through differentiable graph-based modeling with automatic self-calibration. Through simulations, GRAPHYCS achieves 94.5% improvement in wavefront accuracy and substantially better aberration-corrected image quality (PSNR: 38.07 dB vs. 23.93 dB) compared to a state-of-the-art phase-diversity method under system non-idealities. In real-world microscopy experiments, GRAPHYCS delivers 50.4%, 83.0%, and 75.9% improvement in BRISQUE, NIQE, and PIQE metrics for sample-induced aberration correction, while effectively handling spatially varying aberrations across fields of view exceeding 1 mm². GRAPHYCS enables high-resolution imaging across extended regions without additional hardware complexity, providing a practical solution for wide-area aberration correction.

---
## Dataset

You can download the dataset used for GRAPHYCS from https://zenodo.org/records/15421484.

## Reproducing Experimental Results

In order to reconstruct the results of our paper, download the data from Zenodo and place them in a Data folder. 

To reconstruct the results for simulated/experimental data, run the script with the appropriate forward model.
GRAPHYCS is comprised of 4 different forward models:
Spatially invariant forward model for widefield imaging (GRAPHYCS_spatially_invariant_wf.py)
Spatially variant forward model for widefield imaging (GRAPHYCS_spatially_invariant_wf.py)
Spatially invariant forward model for light-sheet imaging (GRAPHYCS_spatially_invariant_lsm.py)
Spatially variant forward model for light-sheet imaging (GRAPHYCS_spatially_variant_lsm.py)

For example, to reconstruct the experimental results for widefield data in Figure 3, run the following:

```bash
python graphycs/GRAPHYCS_spatially_invariant_wf.py \
        --base_dir               "Results" \
        --exp_name               "nonideal_simulated_results" \
        --data_path              "Data/Figure2/Diversity_Images_Nonideal.tif" \
        --zernike_coeff_path     "Data/Figure2/appliedCoeff.txt"
        
```
To apply flat-field correction for data imaged with the large 1094 μm x 1094 μm field of view in Figure 4, place the illumination profile in the Data directory as well: 

```bash
python graphycs/GRAPHYCS_spatially_variant_wf.py \
        --base_dir                       "Results" \
        --exp_name                       "pancreas_full_fov_variant_forward_model" \
        --data_path                      "Data/Figure4/Diversity_Images_SampleAberration_Pancreas_LargeFoV.tif" \
        --zernike_coeff_path             "Data/Figure4/appliedCoeff.txt" \
        --use_flat_field_correction       1 \
        --illum_path                     "Data/IlluminationProfile/Illumination_Profile.tif"
        


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
