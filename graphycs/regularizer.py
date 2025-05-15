import torch
import numpy as np

def second_order_total_variation(img: torch.Tensor) -> torch.Tensor:
    """
    Computes the second order total variation (TV₂) of a 2D image tensor.
    The image is assumed to be a single-channel tensor of shape (H, W).

    TV₂(u) = sum_{i,j} sqrt( (u_xx)^2 + 2*(u_xy)^2 + (u_yy)^2 )
    """
    # Compute first order derivatives along y (row) and x (column)
    grad_y, grad_x = torch.gradient(img)
    
    # Compute second order derivatives:
    # For grad_y, the derivative with respect to y gives u_yy and with respect to x gives one estimate for u_xy.
    dyy, dxy1 = torch.gradient(grad_y)
    
    # For grad_x, the derivative with respect to x gives u_xx and with respect to y gives another estimate for u_xy.
    dxy2, dxx = torch.gradient(grad_x)
    
    # Average the two estimates of the mixed derivative:
    dxy = 0.5 * (dxy1 + dxy2)
    
    # Compute the pointwise second order derivative norm (isotropic TV₂)
    second_deriv_norm = torch.sqrt(dxx**2 + 2*dxy**2 + dyy**2)
    
    # Sum over all pixels to get the total variation value
    tv2 = torch.sum(second_deriv_norm)
    return tv2