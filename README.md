# GRAPHYCS

This repository contains the open-source code for the paper:

**["Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration"](https://doi.org/10.1016/j.isci.2026.116769)**

*iScience* 29, 116769 (2026).

## About GRAPHYCS

Sensor less adaptive optics offers significant advantages over hardware-based wavefront sensing but faces persistent challenges: Its performance degrades when idealized models fail to capture system imperfections, it is largely restricted to spatially invariant aberrations, and it cannot accommodate dynamic biological samples due to static-object assumptions. Here we present graph-modeling and phase-diversity-based computational adaptive optics with self-calibration (GRAPHYCS), a differentiable graph-based modeling framework that addresses all three limitations. GRAPHYCS automatically self-calibrates to correct system-specific non-idealities, enables spatially variant wavefront sensing across extended fields of view by modeling local aberrations, and supports dynamic live-sample imaging where conventional computational methods fail. In simulations, GRAPHYCS achieves up to a 9-fold improvement in wavefront sensing accuracy compared to analytic phase diversity under system non-idealities. In real microscopy experiments, it consistently outperforms phase-diversity-based methods compared in this study. Furthermore, in live zebrafish brain imaging, GRAPHYCS enables simultaneous wavefront sensing and neuronal activity detection—an application beyond the reach of existing approaches without additional hardware complexity.

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

A demonstration of our algorithm on an aberrated image can be used in **GRAPHYCS_demo.ipynb**.

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

## Model Variants

### Widefield, Spatially Invariant

Use `GRAPHYCS_spatially_invariant_wf.py` when the aberration is modeled as field-independent.

### Widefield, Spatially Variant

Use `GRAPHYCS_spatially_variant_wf.py` when the field of view is divided into local patches with spatially varying aberrations.
In order to reproduce the results of Figure 4 of the GRAPHYCS paper, put the Illumination_Profile.tif from Zenodo in a directory and provide as an input to --illum_path.

### Light-Sheet, Spatially Invariant

Use `GRAPHYCS_spatially_invariant_lsm.py` for light-sheet data with dynamic-sample support through optional motion and sparse activity estimation.

### Light-Sheet, Spatially Variant

Use `GRAPHYCS_spatially_variant_lsm.py` for light-sheet data with spatially varying wavefronts, optional motion estimation, and sparse activity estimation.

## Default Saved Results

Each run writes outputs under:

```text
<base_dir>/<exp_name>/
```


### Common outputs for all model variants

- `loss_curve.png`  
  Updated at `--vis_frequency`. Shows the tracked training losses for the active model.

- `phase_estimations/phase_epoch_XXXXXX.png`  
  Intermediate estimated spatially invariant phase maps, saved at `--vis_frequency`. 

- `found_objects/estimated_object_stack_every_100_epochs.tif`  
  Estimated object snapshots saved as a `float32` TIFF stack using `skimage.io`. One new slice is appended every 100 epochs, starting at epoch 0. 

### Self-calibration outputs

When affine transformation self-calibration is enabled:

- `affine_transforms/affine_transform_params_XXXXXX.txt`  
  Saved at `--vis_frequency`. Columns are `[off_x, off_y, amp_x, amp_y, rot, grid]`.

When learnable Zernike scales are enabled:

- `learnable_scales/learnable_scales_XXXXXX.txt`  
  Saved at `--vis_frequency`. No learnable-scale PNG is saved by default.

When motion estimation is enabled for light-sheet runs:

- `motion_factors/motion_factors_XXXXXX.txt`  
  Saved at `--vis_frequency`. Each row stores the learned motion shift parameters for one diversity image/sample index.

### Spatially variant convolution outputs

For `GRAPHYCS_spatially_variant_wf.py` and `GRAPHYCS_spatially_variant_lsm.py`:

- `phase_estimations/spatially_varying_phase_epoch_XXXXXX.png`  
  Spatially varying phase component grid, saved in the same folder and at the same frequency as the spatially invariant phase map. 

### Dynamic light-sheet outputs

For light-sheet runs with sparse activity estimation enabled:

- `found_objects/final_dynamic_sample_component.tif`  
  Final learned dynamic sample component saved once at the end of training as a `float32` TIFF stack using `skimage.io`. The saved array layout is `(height, width, component_index)`.

The default result-saving profile does not write PSF stacks, amplitude stacks, residual wavefront images, coefficient text dumps, or final-results bundles.

## License

This repository is distributed under the GNU General Public License v3.0. See `LICENSE` for details.

## Citation

If you use GRAPHYCS, please cite the accompanying manuscript:

```bibtex
@article{graphycs,
  title={Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration},
  author={Eun-Seo Cho, Joon Park, Hyungwon Jin, Yoonjae Chung, Minho Eom, Hyejin Shin, Jae-Byum Chang, Jung-Hoon Park, Young-Gyu Yoon},
  journal={iScience},
  year={2026}
}
```

