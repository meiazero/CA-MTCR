# CA-MTCR — re-implementation from the paper

You, Xu, Pan, Dong, Yang, Xia. *Change-aware multi-temporal cloud removal.* Sci China
Inf Sci 69(3):132306, 2026. doi:10.1007/s11432-025-4669-x. **No official code.**
Source: the paper's Figs. 2–4, Eqs. (2)–(16) and Sec. 4.1. Code in `src/camtcr/`:
`model.py` (network), `loss.py` (Eqs. 13–16), `sar_pretrain.py` (Eq. 9), `train.py`
(optimizer, schedule and SAR pretraining loop of Sec. 4.1). The data pipeline and the
epoch loop are the caller's.

```bash
uv sync && uv run pytest -q && uvx ruff check .
```

Audited twice against the paper (paper → code and code → paper, 2026-09-24); every
finding is either fixed below or listed as a deviation.

## Stated in the paper, implemented as stated

| Item | Where |
|---|---|
| Target = last step N, others = history (Eq. 2) | `CAMTCR.forward` |
| Optical and SAR encoders applied to each date (Eqs. 3–4) | `CAMTCR.forward` |
| Region-selective encoder: patch embedding, L = 9 Transformer blocks; R1 = spatial attention on the block output, R2 = cosine distance between block input and output; concat → CAM → conv → sigmoid → R; F̃ ⊙ R + F̃ passed to the next block, in every block (Fig. 3) | `RegionSelectiveEncoder`, `RegionSelection` |
| SAR encoder = 3 conv + 2 MBConv; Noise2Noise VQ-VAE pretraining, Eq. (9), α = 1, β = 0.1, 50 epochs, lr 2e-4, then fine-tuned inside the model | `SAREncoder`, `sar_pretrain.py`, `train.pretrain_sar` |
| Multi-modal fusion = concat → conv → MBConv per date (Eq. 5) | `CAMTCR.mmf` |
| Change map = per-pixel cosine similarity of SAR features to step N, ones at N (Eq. 10) | `CAMTCR._change` |
| Fusion: concat [c_t, F_f^t] → conv → L-TAE weights → Σ F_f^t ⊙ W^t, broadcast over channels (Eqs. 11–12); decoder on F_c (Eq. 8) | `ChangeAwareFusion`, `Decoder` |
| Loss 2·L1 + 1·VGG16 feature (relu1_1–relu5_1, L1) + 250·style (relu4_1, relu5_1, L1 on Gram matrices) (Eqs. 13–16), VGG16 frozen | `loss.py` |
| Adam, lr 1e-4, SAR encoder 1e-5, ×0.5 every 5 epochs after the first 10 | `train.optimizer`, `train.scheduler` |
| Batch 6, 30 epochs | caller (values in the caller's config) |
| Ablations "Baseline", "w/o RegS", "w/o CA" (Table 2) | `region_selection`, `change_aware` flags |

## Not stated — our choice (every one is a constructor argument)

| Gap | Choice | Reason |
|---|---|---|
| Patch size, embedding dim, heads | patch 4, dim 160, 4 heads | 7.25 M params and 365 GMACs (T = 3, 256², attention included) vs reported 7.45 M / 456.57 G. The paper's "FLOPs" are MACs: UnCRtainTS, counted the same way, gives 36.2 G vs its reported 37.85 G. Patch 2, dim 144 matched the params but costs 4,368 GMACs (9.6×), 2,087 of them in global attention over 128² tokens |
| Transformer block type | post-norm (Vaswani et al.), global attention, MLP ratio 4, learned 2-D position embedding | with pre-norm the next block's LayerNorm divides out F̃ ⊙ R + F̃, and the mask never reaches the attention (tested) |
| Mask form | F̃ ⊙ R + F̃, as drawn in Fig. 3 | the text says the mask is "suppressing features from the cloudy regions", which F̃ ⊙ R alone would do; the figure's add path scales tokens by 1 + R ∈ [1, 2], a relative suppression |
| R2 | 1 − cosine similarity | text says "cosine distance"; the figure legend says "cosine similarity" |
| Spatial attention, CAM | CBAM modules | the paper names them without detail |
| MBConv | inverted residual, expansion 4, squeeze-excitation, GroupNorm, GELU | the paper cites MobileNetV2 (BatchNorm, ReLU6, no squeeze-excitation) |
| SAR encoder convolutions | 3×3; widths 2 → dim/2 → dim → dim; the first log2(patch) with stride 2 to reach the optical grid; GELU, no norm | the paper says "three convolution layers" |
| Fusion convolutions | 3×3 in the multi-modal fusion and in Eq. (11); GroupNorm after the Eq. (11) conv | not stated |
| L-TAE | 4 heads, d_k 16, averaged into one map; sinusoidal day-of-year encoding (period 1000) added to the keys; no dropout | Eq. (12) broadcasts one map over channels; the paper only cites L-TAE |
| Decoder | log2(patch) × [transposed conv ×2, MBConv], 1×1 conv, sigmoid | the paper only names H_d |
| VGG16 input for 13 bands | RGB (B4, B3, B2), ImageNet normalization | not stated |
| Style distance | L1 on Gram matrices normalized by C·H·W | Eq. (15) prints ‖·‖₁, the text says squared Frobenius; L1 with this normalization is the convention λ3 = 250 comes from. The squared form made the term ~40× smaller |
| Eq. (9) reconstruction term | mean squared error | the paper calls it an L2 reconstruction loss; the equation prints the unsquared norm |
| VQ-VAE | codebook 512 × dim, uniform init ±1/512; decoder 2 MBConv + transposed convs + 3×3 conv | the paper cites VQ-VAE only |
| Learning-rate schedule | full rate for epochs 0–14, halved at 15, 20, 25 | "decay by 50% every 5 epochs following the initial 10 epochs" does not say whether the first halving is at 10 or 15 |
| "w/o CA" | change map of ones at every date (input channel kept) | "exclude change information" |
| Missing SAR | a date without SAR, or every date when step N has none, gets change 1 ("no information", as "w/o CA"); pass `sar_ok` | a zero-filled SAR image still yields a nonzero feature, so its cosine would be arbitrary |

Not implemented: the "w/o SAR" variant (Table 3) and the optical-only input (Fig. 7b).

## Use on another dataset

The model restores the last date of its input. With a benchmark that gives cloudy inputs
and a clear target at another date (AllClear), use the input date nearest to the target
as step N and the others as history; "history" may then lie after N in time. SAR
pretraining pairs are the SAR of an input date and the SAR of the target date ("SAR pairs
that correspond to selected cloudy and cloud-free optical pairs"), both actually
acquired.

## Tests

`tests/test_camtcr.py`: output is one reflectance image; history order is irrelevant;
identical SAR at all dates equals "w/o CA"; changed SAR changes the output; missing SAR
gives no change information; the region mask reaches the next block's attention; loss is
zero only at the target; a module extractor moves with the loss; optimizer groups and
schedule follow Sec. 4.1; the VQ straight-through passes gradients to the encoder and the
codebook; SAR pretraining lowers its loss and its encoder loads into the model.

## Checks before push

Run `git config core.hooksPath .githooks` once per clone. `.githooks/pre-push` then
refuses a push that fails `uvx ruff format --check .` or `uvx ruff check .`.
