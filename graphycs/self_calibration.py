import torch
import torch.nn.functional as F
import numpy as np

dtype = torch.cuda.FloatTensor


def apply_transform(img, theta, device):
    # 2D 이미지를 4D 텐서로 확장
    if img.ndim == 2:
        img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)  # (Batch, Channel, H, W)

    # theta가 배치 차원을 포함하도록 확장
    if theta.ndim == 2:  # (2, 3)
        theta = theta.unsqueeze(0)  # (1, 2, 3)

    # Affine grid 생성 및 변환 적용
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
        img = torch.unsqueeze(torch.unsqueeze(img,0), 0)
    elif img.ndim == 3:
        img = torch.unsqueeze(img, 0)

    theta = torch.stack([
        torch.cat([scale_factor_x, torch.tensor([0]).to(device), torch.tensor([0]).to(device)]),
        torch.cat([torch.tensor([0]).to(device), scale_factor_y, torch.tensor([0]).to(device)])
    ]).unsqueeze(0)

    grid = F.affine_grid(theta, img.size()).to(device)
    img_scaled = F.grid_sample(img, grid)
    return img_scaled.squeeze()

def shift_transform(img, offset_x_learned, offset_y_learned, device):
    if img.ndim == 2:
        img = torch.unsqueeze(torch.unsqueeze(img,0), 0)


    theta = torch.stack([
        torch.cat([torch.tensor([1]).to(device), torch.tensor([0]).to(device), offset_x_learned]),
        torch.cat([torch.tensor([0]).to(device), torch.tensor([1]).to(device), offset_y_learned])
    ]).unsqueeze(0)

    grid = F.affine_grid(theta, img.size())
    img_scaled = F.grid_sample(img, grid)
    return img_scaled.squeeze()


## receive an angle in radiants
def rotation_transform(img, angle_rad, device):
    if img.ndim == 2:
        img = torch.unsqueeze(torch.unsqueeze(img,0), 0)

    theta_rot = torch.stack([
            torch.cat([torch.cos(angle_rad), torch.sin(angle_rad), torch.tensor([0]).to(device)]),
            torch.cat([-1 * torch.sin(angle_rad), torch.cos(angle_rad), torch.tensor([0]).to(device)])
        ]).unsqueeze(0)


    grid = F.affine_grid(theta_rot, img.size())
    img_scaled = F.grid_sample(img, grid)
    return img_scaled.squeeze()