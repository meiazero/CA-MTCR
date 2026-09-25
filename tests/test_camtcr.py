"""Behaviour CA-MTCR must have by the paper's definitions, on a tiny configuration."""

import torch
import torch.nn.functional as F
from torch import nn

import camtcr
import camtcr.loss
import camtcr.model
import camtcr.sar_pretrain
import camtcr.train

TINY = {"dim": 16, "patch": 4, "image_size": 32, "depth": 2, "heads": 2, "fusion_d_k": 4}


def _inputs(B: int = 2, T: int = 3, H: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.rand(B, T, 13, H, H), torch.rand(B, T, 2, H, H), torch.tensor([[10, 40, 70]] * B)


def test_output_is_one_reflectance_image() -> None:
    opt, sar, doy = _inputs()
    out = camtcr.CAMTCR(**TINY).eval()(opt, sar, doy)
    assert out.shape == (2, 13, 32, 32)
    assert out.min() >= 0 and out.max() <= 1


def test_history_order_does_not_matter() -> None:
    model = camtcr.CAMTCR(**TINY).eval()
    opt, sar, doy = _inputs()
    swap = torch.tensor([1, 0, 2])
    with torch.no_grad():
        a = model(opt, sar, doy)
        b = model(opt[:, swap], sar[:, swap], doy[:, swap])
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_unchanged_sar_equals_no_change_information() -> None:
    """Identical SAR at every date means no change, so Eq. (10) gives c = 1 everywhere.

    That is exactly what the "w/o CA" variant feeds.
    """
    torch.manual_seed(1)
    aware = camtcr.CAMTCR(**TINY).eval()
    blind = camtcr.CAMTCR(**TINY, change_aware=False).eval()
    blind.load_state_dict(aware.state_dict())
    opt, sar, doy = _inputs()
    sar = sar[:, :1].expand_as(sar)
    with torch.no_grad():
        torch.testing.assert_close(aware(opt, sar, doy), blind(opt, sar, doy))


def test_changed_sar_changes_the_output() -> None:
    torch.manual_seed(2)
    model = camtcr.CAMTCR(**TINY).eval()
    opt, sar, doy = _inputs()
    changed = sar.clone()
    changed[:, 0] = torch.rand_like(changed[:, 0])
    with torch.no_grad():
        assert not torch.allclose(model(opt, sar, doy), model(opt, changed, doy))


def test_loss_is_zero_only_at_the_target() -> None:
    loss = camtcr.loss.CAMTCRLoss(
        extractor=lambda x: [F.avg_pool2d(x, 2 ** (i + 1)) for i in range(5)]
    )
    target = torch.rand(2, 13, 32, 32)
    assert loss(target, target) == 0
    assert loss(target.flip(-1), target) > 0


def test_sar_pretraining_trains_the_encoder_through_the_quantizer() -> None:
    net = camtcr.sar_pretrain.SARDenoiser(dim=8, patch=4, codes=16)
    y1, y2 = torch.rand(2, 2, 2, 32, 32).unbind(0)
    y1_hat, z_e, e_c = net(y1)
    assert y1_hat.shape == y2.shape
    camtcr.sar_pretrain.pretrain_loss(y1_hat, y2, z_e, e_c).backward()
    assert net.encoder.net[0].weight.grad.abs().sum() > 0
    assert net.quantizer.codebook.weight.grad.abs().sum() > 0


def test_region_mask_reaches_the_next_blocks_attention(monkeypatch) -> None:
    """A pre-norm block would divide the mask out; the next block's attention must see it."""
    seen = []
    sdpa = camtcr.model.F.scaled_dot_product_attention

    def record(q, k, v):
        seen.append((q @ k.transpose(-2, -1)).softmax(-1))
        return sdpa(q, k, v)

    monkeypatch.setattr(camtcr.model.F, "scaled_dot_product_attention", record)
    torch.manual_seed(3)
    enc = camtcr.model.RegionSelectiveEncoder(13, 16, 4, 8, 2, 2, 4.0, True).eval()
    x = torch.rand(1, 13, 32, 32)
    maps = []
    for value in (0.0, 1.0):
        enc.select[0].forward = lambda before, after, v=value: torch.full_like(after[:, :1], v)
        seen.clear()
        with torch.no_grad():
            enc(x)
        maps.append(seen[1])
    assert not torch.allclose(maps[0], maps[1], atol=1e-4)


def test_missing_sar_gives_no_change_information() -> None:
    torch.manual_seed(4)
    aware = camtcr.CAMTCR(**TINY).eval()
    blind = camtcr.CAMTCR(**TINY, change_aware=False).eval()
    blind.load_state_dict(aware.state_dict())
    opt, sar, doy = _inputs()
    none = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(aware(opt, sar, doy, none), blind(opt, sar, doy))
        partial = torch.tensor([[False, True, True]] * 2)
        assert not torch.allclose(aware(opt, sar, doy, partial), blind(opt, sar, doy))


def test_extractor_module_moves_with_the_loss() -> None:
    extractor = nn.Sequential(nn.Conv2d(3, 4, 1))
    loss = camtcr.loss.CAMTCRLoss(extractor=lambda x: [extractor(x)] * 5)
    module_loss = camtcr.loss.CAMTCRLoss(extractor=extractor)
    assert len(list(module_loss.parameters())) == 2 and not list(loss.parameters())


def test_optimizer_and_schedule_follow_the_paper() -> None:
    model = camtcr.CAMTCR(**TINY)
    opt = camtcr.train.optimizer(model)
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 1e-5]
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    sched = camtcr.train.scheduler(opt)
    factors = []
    for _ in range(30):
        factors.append(opt.param_groups[0]["lr"] / 1e-4)
        opt.step()
        sched.step()
    assert factors[14] == 1 and factors[15] == 0.5 and factors[20] == 0.25 and factors[29] == 0.125


def test_sar_pretraining_lowers_the_loss() -> None:
    torch.manual_seed(5)
    net = camtcr.sar_pretrain.SARDenoiser(dim=8, patch=4, codes=16)
    clean = torch.rand(4, 2, 32, 32)
    pairs = [(clean + 0.05 * torch.randn_like(clean), clean + 0.05 * torch.randn_like(clean))]
    history = camtcr.train.pretrain_sar(net, lambda: pairs, epochs=20, lr=1e-3)
    assert history[-1] < history[0]
    model = camtcr.CAMTCR(dim=8, patch=4, image_size=32, depth=1, heads=2, fusion_d_k=4)
    model.sar.load_state_dict(net.encoder.state_dict())
