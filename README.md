# GRAPHYCS

This repository contains the open-source code for the paper:

**"Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration"**


## About GRAPHYCS

Fluorescence microscopy is fundamentally limited by aberrations that degrade resolution and image quality. While adaptive optics can compensate for these distortions, existing approaches face significant limitations: sensor-based methods require complex hardware that increases system complexity and cost, while computational methods rely on idealized models that lead to suboptimal correction when applied to real systems with inevitable imperfections. Moreover, existing computational methods assume static samples, limiting their applicability to live imaging where specimen motion and biological activity occur during acquisition. Here we introduce GRAPHYCS, a computational framework that bridges the gap between models and physical systems through differentiable graph-based modeling with automatic self-calibration and dynamic sample compatibility. In simulations, GRAPHYCS achieves a wavefront RMS error of 0.0252 μm compared to 0.2367 μm for the analytic phase-diversity wavefront sensing method under system non-idealities, and delivers superior aberration-corrected image quality (PSNR: 27.67 dB vs. 23.68 dB). In real-world microscopy experiments, GRAPHYCS demonstrates enhanced sample-induced aberration correction with PSNR of 21.35 dB compared to 19.68 dB for analytic phase-diversity, SSIM of 0.7713 vs. 0.5966, and PCC of 0.9493 vs. 0.8948, while effectively handling spatially varying aberrations across fields of view exceeding 1 mm². Furthermore, GRAPHYCS enables aberration correction in dynamic biological samples, successfully performing simultaneous wavefront sensing and neuronal activity detection in live larval zebrafish brain imaging where conventional phase-diversity methods fail. Overall, GRAPHYCS enables high-resolution imaging across extended regions and in dynamic biological samples without additional hardware complexity, providing a practical solution for aberration correction.

---
## Dataset

You can download the dataset used for GRAPHYCS from https://zenodo.org/records/17049780.

## Requirements

While not specific requirements, the code was tested using the folowing versions of the Python packages and dependencies:

- **Python**: 3.11.10  
- **CUDA**: 12.4
- **PyTorch**: 2.2.1

## Installation

You can install GRAPHYCS as follows:

**1. Clone the directory**
```bash
git clone https://github.com/NICALab/GRAPHYCS.git
```
**2. Navigate into the directory**
```bash
cd ./GRAPHYCS
```

**3. Create and activate a new conda environment**
```bash
conda create -n graphycs python=3.11.10
conda activate graphycs
```
**4. In the conda environment, install all of the necessary packages using requirements.txt**
```bash
pip install -r requirements.txt
```

## Reproducing Experimental Results

In order to reconstruct the results of our paper, download the data from Zenodo and place them in a Data folder. 

To reconstruct the results for simulated/experimental data, run the script with the appropriate forward model.
GRAPHYCS is comprised of 4 different forward models:
- **Widefield — spatially invariant:** `graphycs/GRAPHYCS_spatially_invariant_wf.py`
- **Widefield — spatially variant:** `graphycs/GRAPHYCS_spatially_variant_wf.py`
- **Light-sheet — spatially invariant:** `graphycs/GRAPHYCS_spatially_invariant_lsm.py`
- **Light-sheet — spatially variant:** `graphycs/GRAPHYCS_spatially_variant_lsm.py`

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
```        




The list of Python packages and dependencies are specified in the [`requirements.txt`](requirements.txt) file.
