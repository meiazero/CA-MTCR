"""CA-MTCR objective, Eq. (16): lambda1 L1 + lambda2 feature loss + lambda3 style loss.

VGG16 sees the RGB bands (B4, B3, B2) with ImageNet normalization; the paper does not say
how 13 bands enter VGG16.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RGB = (3, 2, 1)  # B4, B3, B2 in the 13-band S2 order
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG16_RELU_X_1 = (1, 6, 11, 18, 25)  # relu1_1 ... relu5_1 in torchvision's vgg16.features
STYLE_FROM = 3  # Eq. (15) sums from l = 4: relu4_1 and relu5_1


class VGG16Features(nn.Module):
    """Frozen ImageNet VGG16 returning relu1_1..relu5_1. Downloads weights on first use.

    A module, not a closure, so `CAMTCRLoss.to(device)` moves VGG16 with it.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1
        self.features = vgg16(weights=weights).features[: VGG16_RELU_X_1[-1] + 1]
        self.features.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "VGG16Features":
        return super().train(False)  # always frozen

    def forward(self, x: Tensor) -> list[Tensor]:
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in VGG16_RELU_X_1:
                out.append(x)
        return out


def gram(f: Tensor) -> Tensor:
    """Gram matrix normalized by C*H*W, the convention behind lambda3 = 250."""
    B, C, H, W = f.shape
    flat = f.reshape(B, C, H * W)
    return flat @ flat.transpose(1, 2) / (C * H * W)


class CAMTCRLoss(nn.Module):
    def __init__(
        self,
        extractor: Callable[[Tensor], list[Tensor]],  # VGG16Features() for training
        l1: float = 2.0,
        feature: float = 1.0,
        style: float = 250.0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.weights = (l1, feature, style)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        fp = self.extractor(self._rgb(pred))
        ft = self.extractor(self._rgb(target))
        l_fr = sum(F.l1_loss(a, b) for a, b in zip(fp, ft, strict=True))
        l_sr = sum(
            F.l1_loss(gram(a), gram(b))  # ||.||_1 as printed in Eq. (15)
            for a, b in zip(fp[STYLE_FROM:], ft[STYLE_FROM:], strict=True)
        )
        w1, w2, w3 = self.weights
        return w1 * F.l1_loss(pred, target) + w2 * l_fr + w3 * l_sr

    def _rgb(self, x: Tensor) -> Tensor:
        return (x[:, RGB] - self.mean) / self.std
