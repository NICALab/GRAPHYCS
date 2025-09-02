import torch
import torch.nn.functional as F
import numpy as np

dtype = torch.cuda.FloatTensor


def apply_transform(img, theta, device):
    if img.ndim == 2:
        img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)  # (Batch, Channel, H, W)

    if theta.ndim == 2:  # (2, 3)
        theta = theta.unsqueeze(0)  # (1, 2, 3)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    img_scaled = F.grid_sample(img, grid, align_corners=False)
    return img_scaled.squeeze()  # 배치 및 채널 차원 제거


def affine_transform(gaussian_intensity_large, scale_factor_x, scale_factor_y, offset_x_learned, offset_y_learned):
    offset_x_learned = torch.max(torch.min(offset_x_learned, torch.tensor([0.2]).cuda()), torch.tensor([-0.2]).cuda())
    offset_y_learned = torch.max(torch.min(offset_y_learned, torch.tensor([0.2]).cuda()), torch.tensor([-0.2]).cuda())

    theta_tilt = torch.stack([
        torch.cat([scale_factor_x, torch.tensor([0]).cuda(), offset_x_learned]),
        torch.cat([torch.tensor([0]).cuda(), scale_factor_y, offset_y_learned])
    ]).unsqueeze(0)
    gaussian_intensity_large = torch.unsqueeze(torch.unsqueeze(gaussian_intensity_large, 0), 0)
    grid = F.affine_grid(theta_tilt, gaussian_intensity_large.size())
    gaussian_intensity_large = F.grid_sample(gaussian_intensity_large, grid)
    gaussian_intensity_large = gaussian_intensity_large.squeeze()
    return gaussian_intensity_large



def scale_transform(img, scale_factor_x, scale_factor_y, device):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif img.ndim == 3:
        img = img.unsqueeze(1)  # (N,1,H,W)

    N = img.shape[0]
    if scale_factor_x.numel() == 1:
        scale_factor_x = scale_factor_x.repeat(N)
    if scale_factor_y.numel() == 1:
        scale_factor_y = scale_factor_y.repeat(N)

    zeros = torch.zeros_like(scale_factor_x)

    theta = torch.stack([
        torch.stack([scale_factor_x, zeros, zeros], dim=1),
        torch.stack([zeros, scale_factor_y, zeros], dim=1)
    ], dim=1)  # (N, 2, 3)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    img_scaled = F.grid_sample(img, grid, align_corners=False)
    return img_scaled.squeeze()

def shift_transform(img, offset_x_learned, offset_y_learned, device):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif img.ndim == 3:
        img = img.unsqueeze(1)  # (N,1,H,W)

    N = img.shape[0]
    if offset_x_learned.numel() == 1:
        offset_x_learned = offset_x_learned.repeat(N)
    if offset_y_learned.numel() == 1:
        offset_y_learned = offset_y_learned.repeat(N)

    ones = torch.ones_like(offset_x_learned)
    zeros = torch.zeros_like(offset_x_learned)

    theta = torch.stack([
        torch.stack([ones, zeros, offset_x_learned], dim=1),
        torch.stack([zeros, ones, offset_y_learned], dim=1)
    ], dim=1)  # (N, 2, 3)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    img_scaled = F.grid_sample(img, grid, align_corners=False)
    return img_scaled.squeeze()


def shift_transform_batch(img, offset_x_learned, offset_y_learned, device):
    """
    img: (1, H, W) or (N, 1, H, W) tensor
    offset_x_learned: (N,) tensor of x shifts
    offset_y_learned: (N,) tensor of y shifts
    returns: (N, 1, H, W) tensor of shifted images
    """
    if img.ndim == 3:
        # Input is (1, H, W), repeat N times
        N = offset_x_learned.shape[0]
        img = img.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, 1, H, W)
    elif img.ndim == 4:
        N = img.shape[0]
    else:
        raise ValueError("img must be of shape (1, H, W) or (N, 1, H, W)")

    # Build affine matrices for each sample
    ones = torch.ones(N, 1, device=device)
    zeros = torch.zeros(N, 1, device=device)

    theta = torch.cat([
        torch.cat([ones, zeros, offset_x_learned.view(N, 1)], dim=1),
        torch.cat([zeros, ones, offset_y_learned.view(N, 1)], dim=1)
    ], dim=1).view(N, 2, 3)  # (N, 2, 3)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    img_shifted = F.grid_sample(img, grid, align_corners=False)

    return img_shifted  # (N, 1, H, W)


def rotation_transform(img, angle_rad, device):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif img.ndim == 3:
        img = img.unsqueeze(1)  # (N,1,H,W)

    N = img.shape[0]
    if angle_rad.numel() == 1:
        angle_rad = angle_rad.repeat(N)

    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)

    zeros = torch.zeros_like(cos)
    theta = torch.stack([
        torch.stack([cos, sin, zeros], dim=1),
        torch.stack([-sin, cos, zeros], dim=1)
    ], dim=1)  # (N, 2, 3)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    img_scaled = F.grid_sample(img, grid, align_corners=False)

    return img_scaled.squeeze()


def scale_transform_batched(img, scale_factor_x, scale_factor_y, device):
    # ── Bring to 4D tensor (B, C, H, W) ──────────────────────────────────────
    if img.ndim == 2:            # (H, W)
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:          # (C, H, W) or (B, H, W)
        img = img.unsqueeze(0)

    B, C, H, W = img.shape

    # ── Make per-image scale factors ─────────────────────────────────────────
    # allow scalars or per-image vectors
    if torch.numel(scale_factor_x) == 1:
        scale_factor_x = scale_factor_x.repeat(B)
    if torch.numel(scale_factor_y) == 1:
        scale_factor_y = scale_factor_y.repeat(B)

    zeros = torch.zeros_like(scale_factor_x, device=device)

    # ── Build the batch of 2×3 affine mats ──────────────────────────────────
    theta = torch.stack([
        torch.stack([scale_factor_x, zeros,           zeros], dim=1),
        torch.stack([zeros,           scale_factor_y, zeros], dim=1),
    ], dim=1).to(device)   # (B, 2, 3)

    # ── Warp ────────────────────────────────────────────────────────────────
    grid       = F.affine_grid(theta, img.size(), align_corners=False)
    img_scaled = F.grid_sample (img,  grid,      align_corners=False)

    return img_scaled  # shape = (B, C, H, W)


def shift_transform_batched(img, offset_x, offset_y, device):
    # ── Bring to 4D tensor (B, C, H, W) ──────────────────────────────────────
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)
    # now img.shape == (B, C, H, W)

    B, C, H, W = img.shape

    # ── Make per-image offsets ───────────────────────────────────────────────
    if torch.numel(offset_x) == 1:
        offset_x = offset_x.repeat(B)
    if torch.numel(offset_y) == 1:
        offset_y = offset_y.repeat(B)

    ones  = torch.ones_like(offset_x, device=device)
    zeros = torch.zeros_like(offset_x, device=device)

    theta = torch.stack([
        torch.stack([ones,  zeros, offset_x], dim=1),
        torch.stack([zeros, ones,  offset_y], dim=1),
    ], dim=1).to(device)   # (B, 2, 3)

    # ── Warp ────────────────────────────────────────────────────────────────
    grid      = F.affine_grid(theta, img.size(), align_corners=False)
    img_shift = F.grid_sample (img,  grid,      align_corners=False)

    return img_shift  # shape = (B, C, H, W)