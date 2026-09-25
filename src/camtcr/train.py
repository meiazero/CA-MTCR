"""Training settings of Sec. 4.1 and the SAR pretraining loop of Sec. 3.2.

The data pipeline and the epoch loop belong to whoever trains the model; these functions
hold only what the paper fixes, so no caller can drift from it.
"""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

import camtcr.model
import camtcr.sar_pretrain


def optimizer(
    model: camtcr.model.CAMTCR, lr: float = 1e-4, sar_lr: float = 1e-5
) -> torch.optim.Adam:
    """Adam at `lr`, except the pretrained SAR encoder, fine-tuned at `sar_lr`."""
    sar = list(model.sar.parameters())
    ids = {id(p) for p in sar}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": rest, "lr": lr}, {"params": sar, "lr": sar_lr}])


def scheduler(
    opt: torch.optim.Optimizer, after: int = 10, every: int = 5, factor: float = 0.5
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch decay "by 50% every 5 epochs following the initial 10 epochs".

    Read as: full rate for epochs 0-14, then halved at epochs 15, 20, 25. The paper does not
    say whether the first halving is at epoch 10 or 15; call `.step()` once per epoch.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda epoch: factor ** max(0, (epoch - after) // every)
    )


def pretrain_sar(
    net: camtcr.sar_pretrain.SARDenoiser,
    pairs: Callable[[], Iterable[tuple[Tensor, Tensor]]],
    epochs: int = 50,
    lr: float = 2e-4,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> list[float]:
    """Noise2Noise VQ-VAE pretraining; `pairs()` yields (y1, y2) SAR batches of one scene.

    Returns the mean loss per epoch. Afterwards load the encoder into the main model with
    `model.sar.load_state_dict(net.encoder.state_dict())`.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    device = next(net.parameters()).device
    history = []
    net.train()
    for _ in range(epochs):
        total, n = 0.0, 0
        for y1, y2 in pairs():
            y1_hat, z_e, e_c = net(y1.to(device))
            loss = camtcr.sar_pretrain.pretrain_loss(y1_hat, y2.to(device), z_e, e_c, alpha, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        history.append(total / max(n, 1))
    return history
