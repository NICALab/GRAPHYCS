
import torch
import numpy as np
from torch.utils.data import Dataset
import skimage.io as skio

class BatchDatasetFromTIFFAndZernike(Dataset):
    def __init__(self, tif_path, coeff_path, device='cuda'):
        self.device = device
        super().__init__()

        # === Load raw intensity images
        phase_added_images = skio.imread(tif_path).astype(np.float32)

        avg_background = np.min(phase_added_images[phase_added_images > 0])
        phase_added_images = np.maximum(phase_added_images - avg_background, np.zeros_like(phase_added_images))
        phase_added_images = phase_added_images / np.max(phase_added_images)
        phase_added_images = np.squeeze(phase_added_images)  # remove singleton dimensions
        _, w, h = phase_added_images.shape
        
        self.ys = [torch.from_numpy(img).to(self.device) for img in phase_added_images]

        # === Load Zernike coefficients
        coeffs = np.loadtxt(coeff_path)
        if coeffs.ndim == 1:
            coeffs = coeffs[np.newaxis, ...]
        self.coeffs = torch.from_numpy(coeffs).float().to(self.device)


    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.coeffs[idx], self.ys[idx], idx