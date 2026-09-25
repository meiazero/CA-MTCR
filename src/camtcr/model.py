"""CA-MTCR (You et al., Sci China Inf Sci 69(3):132306, 2026), re-implemented from the paper.

No official code exists. Every value the paper does not state is a constructor argument
with a default documented in README.md next to this file; nothing is hidden in the body.
Equation and figure numbers refer to the paper.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

S2_BANDS = 13
S1_BANDS = 2


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x.mean((2, 3), keepdim=True))


class MBConv(nn.Module):
    """Inverted residual block (MobileNetV2) with squeeze-excitation, stride 1."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class TransformerBlock(nn.Module):
    """Post-norm Transformer block (Vaswani et al.) over the h*w token grid.

    Post-norm so the region mask R applied to this block's input reaches the attention: a
    pre-norm block's LayerNorm divides out the per-token scale F~ * R + F~ (measured
    difference 6e-5), which would make region selection a no-op.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, N, C/heads]
        attn = F.scaled_dot_product_attention(q, k, v)
        x = self.norm1(x + self.proj(attn.transpose(1, 2).reshape(B, N, C)))
        return self.norm2(x + self.mlp(x))


class SpatialAttention(nn.Module):
    """CBAM spatial attention: [mean_c, max_c] -> 7x7 conv -> sigmoid; gives R1 (Fig. 3)."""

    def __init__(self, kernel: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv(pooled))


class ChannelAttention(nn.Module):
    """CBAM channel attention over the two region cues [R1, R2] (the CAM of Fig. 3)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = self.mlp(x.mean((2, 3), keepdim=True))
        mx = self.mlp(x.amax((2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class RegionSelection(nn.Module):
    """Soft mask R from the block output (R1) and the input/output cosine distance (R2)."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialAttention()
        self.cam = ChannelAttention(2)
        self.conv = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, before: Tensor, after: Tensor) -> Tensor:
        r1 = self.spatial(after)
        r2 = 1 - F.cosine_similarity(before, after, dim=1).unsqueeze(1)
        return torch.sigmoid(self.conv(self.cam(torch.cat([r1, r2], 1))))


class RegionSelectiveEncoder(nn.Module):
    """Optical encoder of Fig. 3: patch embedding, then L region-selective blocks.

    Each block's output is re-weighted as F_next = F~ * R + F~ (the residual path in Fig. 3
    leaves from F~). `region_selection=False` keeps only the Transformer blocks ("w/o RegS",
    Table 2).
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch: int,
        grid: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        region_selection: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, dim, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.select = (
            nn.ModuleList(RegionSelection() for _ in range(depth)) if region_selection else None
        )

    def forward(self, x: Tensor) -> Tensor:
        f = self.embed(x)
        B, C, h, w = f.shape
        f = f + F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        for j, block in enumerate(self.blocks):
            out = block(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, h, w)
            if self.select is not None:
                out = out * self.select[j](f, out) + out
            f = out
        return f


class SAREncoder(nn.Module):
    """Three convolutions (the first log2(patch) with stride 2) and two MBConv blocks."""

    def __init__(self, dim: int, patch: int) -> None:
        super().__init__()
        downs = int(math.log2(patch))
        if 2**downs != patch or downs > 3:
            raise ValueError(f"patch must be 1, 2, 4 or 8, got {patch}")
        chans = [S1_BANDS, dim // 2, dim, dim]
        layers: list[nn.Module] = []
        for i in range(3):
            stride = 2 if i < downs else 1
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride, 1), nn.GELU()]
        self.net = nn.Sequential(*layers, MBConv(dim), MBConv(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ChangeAwareFusion(nn.Module):
    """Eqs. (10)-(12): concat [c_t, F_f^t] -> conv -> L-TAE attention over dates -> weighted sum.

    Heads are averaged into one map W^t, broadcast over channels as in Eq. (12).
    """

    def __init__(self, dim: int, heads: int, d_k: int, max_doy: int = 366) -> None:
        super().__init__()
        self.heads = heads
        self.d_k = d_k
        self.conv = nn.Conv2d(dim + 1, dim, 3, padding=1)
        self.norm = nn.GroupNorm(1, dim)
        self.key = nn.Linear(dim, heads * d_k)
        self.query = nn.Parameter(torch.randn(heads, d_k) * d_k**-0.5)
        self.register_buffer("pe", _sinusoid(max_doy + 1, dim), persistent=False)

    def forward(self, feats: Tensor, change: Tensor, doy: Tensor) -> Tensor:
        """feats [B,T,C,h,w], change [B,T,1,h,w], doy [B,T] -> [B,C,h,w]."""
        B, T, C, h, w = feats.shape
        x = self.conv(torch.cat([change, feats], 2).flatten(0, 1))
        x = self.norm(x).view(B, T, C, h, w).permute(0, 3, 4, 1, 2)  # B h w T C
        x = x + self.pe[doy.long()].view(B, 1, 1, T, C)
        keys = self.key(x).view(B, h, w, T, self.heads, self.d_k)
        logits = torch.einsum("bhwtnk,nk->bhwnt", keys, self.query) / math.sqrt(self.d_k)
        weights = logits.softmax(-1).mean(3)  # B h w T
        return torch.einsum("bhwt,btchw->bchw", weights, feats)


def _sinusoid(n: int, dim: int) -> Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(1000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Decoder(nn.Module):
    """log2(patch) x [transposed conv x2, MBConv], then 1x1 to 13 bands and a sigmoid."""

    def __init__(self, dim: int, patch: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [MBConv(dim)]
        for _ in range(int(math.log2(patch))):
            layers += [nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.GELU(), MBConv(dim)]
        self.net = nn.Sequential(*layers, nn.Conv2d(dim, S2_BANDS, 1))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class CAMTCR(nn.Module):
    def __init__(
        self,
        dim: int = 160,
        patch: int = 4,
        image_size: int = 256,
        depth: int = 9,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        fusion_heads: int = 4,
        fusion_d_k: int = 16,
        region_selection: bool = True,
        change_aware: bool = True,
    ) -> None:
        super().__init__()
        self.change_aware = change_aware
        self.optical = RegionSelectiveEncoder(
            S2_BANDS, dim, patch, image_size // patch, depth, heads, mlp_ratio, region_selection
        )
        self.sar = SAREncoder(dim, patch)
        self.mmf = nn.Sequential(nn.Conv2d(2 * dim, dim, 3, padding=1), MBConv(dim))
        self.fusion = ChangeAwareFusion(dim, fusion_heads, fusion_d_k)
        self.decoder = Decoder(dim, patch)

    def forward(
        self, optical: Tensor, sar: Tensor, doy: Tensor, sar_ok: Tensor | None = None
    ) -> Tensor:
        """optical [B,T,13,H,W], sar [B,T,2,H,W], doy [B,T], sar_ok [B,T] bool -> [B,13,H,W].

        The LAST date is the step N to reconstruct, the others are historical references
        (Eq. 2). `sar_ok` marks dates whose SAR was actually acquired (None = all).
        """
        B, T = optical.shape[:2]
        f_opt = self.optical(optical.flatten(0, 1))
        f_sar = self.sar(sar.flatten(0, 1))
        f_fused = self.mmf(torch.cat([f_opt, f_sar], 1)).unflatten(0, (B, T))
        f_sar = f_sar.unflatten(0, (B, T))
        change = self._change(f_sar, sar_ok)
        return self.decoder(self.fusion(f_fused, change, doy))

    def _change(self, f_sar: Tensor, sar_ok: Tensor | None) -> Tensor:
        """Eq. (10): per-pixel cosine similarity of each date's SAR features to step N.

        Ones at N. "w/o CA" (Table 2) feeds ones for every date, i.e. no change information;
        a date without SAR, or any date when N has none, gets that same "no information" 1.
        """
        if not self.change_aware:
            return torch.ones_like(f_sar[:, :, :1])
        sim = F.cosine_similarity(f_sar, f_sar[:, -1:], dim=2).unsqueeze(2)
        change = torch.cat([sim[:, :-1], torch.ones_like(sim[:, -1:])], 1)
        if sar_ok is None:
            return change
        known = (sar_ok & sar_ok[:, -1:])[:, :, None, None, None]
        return torch.where(known, change, torch.ones_like(change))
