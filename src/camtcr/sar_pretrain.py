"""Noise2Noise VQ-VAE pretraining of the CA-MTCR SAR encoder (Sec. 3.2, Eq. 9).

Two SAR acquisitions of the same scene are two noisy views of one surface: the network
encodes y1 and is supervised with y2, so it cannot learn the speckle. After pretraining
only `encoder` is kept to initialize `CAMTCR.sar`.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import camtcr.model


class VectorQuantizer(nn.Module):
    def __init__(self, codes: int, dim: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codes, dim)
        self.codebook.weight.data.uniform_(-1 / codes, 1 / codes)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z [B,C,h,w] -> (straight-through quantized z, nearest codebook vectors e_c)."""
        flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        idx = torch.cdist(flat, self.codebook.weight).argmin(1)
        e = self.codebook(idx).view(z.shape[0], z.shape[2], z.shape[3], -1).permute(0, 3, 1, 2)
        return z + (e - z).detach(), e


class SARDenoiser(nn.Module):
    def __init__(self, dim: int, patch: int, codes: int = 512) -> None:
        super().__init__()
        self.encoder = camtcr.model.SAREncoder(dim, patch)
        self.quantizer = VectorQuantizer(codes, dim)
        layers: list[nn.Module] = [
            camtcr.model.MBConv(dim),
            camtcr.model.MBConv(dim),
        ]
        for _ in range(int(math.log2(patch))):
            layers += [nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.GELU()]
        self.decoder = nn.Sequential(*layers, nn.Conv2d(dim, camtcr.model.S1_BANDS, 3, padding=1))

    def forward(self, y1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """-> (reconstruction of y1, encoder output z_e, codebook vectors e_c)."""
        z_e = self.encoder(y1)
        z_q, e_c = self.quantizer(z_e)
        return self.decoder(z_q), z_e, e_c


def pretrain_loss(
    y1_hat: Tensor, y2: Tensor, z_e: Tensor, e_c: Tensor, alpha: float = 1.0, beta: float = 0.1
) -> Tensor:
    """Eq. (9): ||y2 - y1_hat||_2 + alpha ||sg[z_e] - e_c||^2 + beta ||z_e - sg[e_c]||^2.

    The reconstruction term is a mean squared error ("L2 reconstruction loss"), as in the
    VQ-VAE the paper cites, rather than the unsquared norm of the printed equation.
    """
    return (
        F.mse_loss(y1_hat, y2)
        + alpha * F.mse_loss(e_c, z_e.detach())
        + beta * F.mse_loss(z_e, e_c.detach())
    )
