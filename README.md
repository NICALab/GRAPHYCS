# GRAPHYCS

This repository contains the open-source code for the paper:

**"Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration"**


## About GRAPHYCS

Sensorless adaptive optics offers significant advantages over hardware-based wavefront sensing but faces persistent challenges: its performance degrades when idealized models fail to capture system imperfections, it is largely restricted to spatially invariant aberrations, and it cannot accommodate dynamic biological samples due to static-object assumptions. Here we present GRAPHYCS (GRAph-modeling and PHase-diversitY-based Computational adaptive optics with Self-calibration), a differentiable graph-based modeling framework that addresses all three limitations. GRAPHYCS automatically self-calibrates to correct system-specific non-idealities, enables spatially variant wavefront sensing across extended fields of view by modeling local aberrations, and supports dynamic live-sample imaging where conventional computational methods fail. In simulations, GRAPHYCS achieves up to a ninefold improvement in wavefront sensing accuracy under system non-idealities. In real microscopy experiments, it consistently outperforms state-of-the-art methods. Furthermore, in live zebrafish brain imaging, GRAPHYCS enables simultaneous wavefront sensing and neuronal activity detection—an application beyond the reach of existing approaches without additional hardware complexity.

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

## Executable demonstration

A demonstration of our algorithm on an aberrated image can be used in GRAPHYCS_demo.ipynb.
Download the repository on your workspace and run the demo to see the results on the aberrated image in the Data folder. 

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
To apply flat-field correction for pancreas tissue data in Figure 3 and Figure 4, place the illumination profile in the Data directory as well: 

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
