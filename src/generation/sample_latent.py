"""
sample_latent.py — Utilities for latent space exploration in the VAE.

Provides:
    - Random sampling from the prior
    - Interpolation between two points
    - Grid traversal over 2D latent subspace
"""
import torch
import numpy as np
from typing import Optional


def sample_prior(n_samples: int, latent_dim: int,
                 device: str = "cpu") -> torch.Tensor:
    """Sample z ~ N(0, I)  shape: (n_samples, latent_dim)"""
    return torch.randn(n_samples, latent_dim, device=device)


def spherical_interpolate(z1: torch.Tensor, z2: torch.Tensor,
                           steps: int = 10) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two latent vectors.
    More natural than linear interpolation on the unit sphere.
    Returns (steps, latent_dim).
    """
    z1_n = z1 / z1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z2_n = z2 / z2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    omega = torch.acos((z1_n * z2_n).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7))

    alphas = torch.linspace(0, 1, steps, device=z1.device)
    results = []
    for a in alphas:
        if omega.abs() < 1e-6:
            z = (1 - a) * z1 + a * z2
        else:
            z = (torch.sin((1 - a) * omega) / torch.sin(omega)) * z1 +                 (torch.sin(a * omega) / torch.sin(omega)) * z2
        results.append(z)
    return torch.stack(results)        # (steps, latent_dim)


def linear_interpolate(z1: torch.Tensor, z2: torch.Tensor,
                        steps: int = 10) -> torch.Tensor:
    """Linear interpolation between z1 and z2. Returns (steps, latent_dim)."""
    alphas = torch.linspace(0, 1, steps, device=z1.device)
    return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])


def latent_grid(z_center: torch.Tensor, dim1: int = 0, dim2: int = 1,
                extent: float = 3.0, grid_size: int = 5) -> torch.Tensor:
    """
    Create a 2D grid of latent vectors varying along dim1 and dim2.
    Useful for visualising smooth musical transitions.
    Returns (grid_size*grid_size, latent_dim).
    """
    vals = torch.linspace(-extent, extent, grid_size, device=z_center.device)
    samples = []
    for v1 in vals:
        for v2 in vals:
            z = z_center.clone()
            z[dim1] = v1
            z[dim2] = v2
            samples.append(z)
    return torch.stack(samples)
