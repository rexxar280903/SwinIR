# Method

This document defines everything that is computed: the data, the network, the training
objective and the metrics. The experimental protocol (which runs, which statistics, which
decisions) is in [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Task and degradation

Single-image super-resolution at scale ×2. For every source image *I*:

1. RGBA images are composited on white; non-square images are centre-cropped to a square.
2. **HR** = bicubic resize of *I* to 128 × 128 (PIL, `Image.BICUBIC`).
3. **LR** = bicubic resize of HR to 64 × 64 (same kernel).

The network sees LR and must predict HR. Using a single, known degradation (bicubic
downsampling) is the classical "bicubic SR" setting of SwinIR's Table 2 and of most SR
benchmarks; it isolates the effect of the loss from the effect of unknown real-world blur,
noise and compression. The readiness gate verifies that every LR file equals `bicubic(HR)`
exactly (check R1.3).

## 2. Network: SwinIR, unchanged

SwinIR (Liang et al., 2021) has three parts:

```
LR ─► conv 3×3 (shallow features F0)
        │
        ├─► K × RSTB ─► conv 3×3 ─► (+ F0)          deep feature extraction, long skip
        │     RSTB = L × STL ─► conv 3×3, with a residual connection
        │     STL  = LN ─► W-MSA / SW-MSA ─► + ─► LN ─► MLP ─► +
        ▼
   upsampler (pixel shuffle) ─► SR
```

* **W-MSA**: self-attention inside non-overlapping 8 × 8 windows, with a learned relative
  position bias. **SW-MSA**: the same on windows shifted by 4 pixels, alternating layer by
  layer, so information crosses window borders.
* The input is normalised by subtracting a fixed RGB mean; the output adds it back.

The file [`edgesr/models/swinir.py`](../edgesr/models/swinir.py) is the official implementation.
The only edits are three helpers formerly imported from `timm` and `meshgrid(indexing="ij")`,
which gives the same tensor. Gate check R3.2 downloads the official file and compares the
code line by line. All differences between experimental conditions are therefore in the loss.

| Preset | embed dim | RSTB × STL | heads | upsampler | params (×2) | role |
|---|---|---|---|---|---|---|
| `light` | 60 | 4 × 6 | 6 | pixel shuffle (direct) | 0.91 M¹ | **default for the ablation** (SwinIR-light) |
| `medium` | 180 | 4 × 6 | 6 | conv + pixel shuffle | 8.02 M | the original notebook's configuration |
| `classical` | 180 | 6 × 6 | 6 | conv + pixel shuffle | 11.75 M | SwinIR paper, classical SR |

¹ The SwinIR paper lists 878 K for SwinIR-light ×2; the difference is the 24 relative-position
bias tables (24 × 1,350 values), which the paper does not count.

The ablation uses `light` because 20+ training runs (sweep, three seeds) must fit a free Kaggle
GPU quota. Gate check R6.1 measures the throughput of `light` and `medium` on the actual GPU, so
the choice can be justified with numbers. Whether the effect of the loss carries over to the
larger model is a separate question (see the threats to validity).

## 3. Training objective

Let $\hat{y}$ be the SR output and $y$ the HR target, both in $[0,1]^{3\times H\times W}$.

### 3.1 Sobel gradient magnitude

$$
K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\qquad K_y=K_x^{\top},
\qquad S(x)=\sqrt{(K_x * x)^2+(K_y * x)^2+\varepsilon},\ \varepsilon=10^{-6}.
$$

The convolution is applied to each RGB channel separately (depthwise) with **replicate
padding**, so a constant image has zero gradient everywhere, including the border. ε keeps
the derivative of the square root finite where both gradients are zero. For a step of height
*h* the response next to the step is 4*h*.

### 3.2 Loss

$$
\mathcal{L}=\underbrace{\lVert \hat{y}-y\rVert_1}_{\text{pixel (mean)}}\;+\;\lambda\,\mathcal{L}_{\text{edge}}
$$

| `edge_mode` | $\mathcal{L}_{\text{edge}}$ |
|---|---|
| `none` (L1 baseline) | 0 |
| `static` | $\operatorname{mean}\,\lvert S(\hat{y})-S(y)\rvert$ |
| `adaptive` | $\operatorname{mean}\; w \odot \lvert S(\hat{y})-S(y)\rvert,\quad w=\dfrac{S(y)-\min S(y)}{\max S(y)-\min S(y)+10^{-8}}$ |

For `adaptive`, min and max are taken **per image** over its channels and pixels, so
$w\in[0,1]$ is 0 on the flattest pixel of the image and 1 on its strongest edge. $w$ is computed
from the ground truth only and carries no gradient: it says *where* gradient errors count, it
does not move. The loss is evaluated in float32 even when the network runs in float16.

**Why the weighting might help.** Anime line art is mostly flat colour separated by thin,
high-contrast outlines. A static gradient loss spends most of its weight on the flat interior,
where gradients are near zero and errors are small. The adaptive weight concentrates the edge
term on the outlines, which is where blur is visible. **Why it might not.** Weak edges (hair
strands, shading) get small weights, and the network may trade them for strong edges.
This is exactly what the experiment measures.

**Effective strength.** Because $w\le1$, adaptive and static are not equally strong at the same
λ. The study therefore sweeps λ for both and compares each at its own validation-selected
λ*. The analysis also reports the *edge share* $\lambda\mathcal{L}_{\text{edge}}/\mathcal{L}$ at the end of
training, i.e. how much of the objective the edge term really is.

**Relation to earlier work.** Penalising the difference of image gradients is an established
idea in SR, e.g. the gradient loss of SPSR (Ma et al., 2020) and the gradient-variance loss of
Abrahamyan et al. (2022). The contribution here is not a new loss family; it is (i) a spatial
weighting by ground-truth edge strength, tested against the unweighted version at matched
tuning, (ii) on line-art-dominated anime images, (iii) with a pre-registered, multi-seed
protocol. Claims must stay within that scope.

## 4. Metrics

All metrics are computed per image and then averaged over the test set. The SR output is
clamped to [0, 1] and **rounded to 8 bit** first, as if saved to PNG; then `crop_border = 2`
pixels (the scale) are removed on every side. This follows the SwinIR/BasicSR test code.

| Metric | Definition | Better |
|---|---|---|
| PSNR-Y | $10\log_{10}(255^2/\mathrm{MSE})$ on the Y channel of BT.601 YCbCr: $Y=16+(65.481R+128.553G+24.966B)/255$, with R, G, B in [0, 255]. **Main fidelity metric** (SR convention). | ↑ |
| SSIM-Y | SSIM (Wang et al., 2004) on Y: Gaussian window 11 × 11, σ = 1.5, $K_1=0.01$, $K_2=0.03$, data range 255, valid region only. | ↑ |
| PSNR-RGB, SSIM-RGB | The same on RGB (SSIM averaged over channels). Reported for completeness. | ↑ |
| **Edge-PSNR-Y** | PSNR on Y restricted to the edge mask $M=\{p: S_{\text{Y}}(y)_p>\tau\}$, where $S_{\text{Y}}$ is the Sobel magnitude of $Y/255$ of the HR image and $\tau=0.5$. $\mathrm{MSE}_M=\sum_{p\in M}(\hat{Y}_p-Y_p)^2/\lvert M\rvert$. Images without edge pixels are excluded (and counted). **Primary endpoint.** | ↑ |
| GMSD | Gradient Magnitude Similarity Deviation (Xue et al., 2014): Prewitt gradient magnitudes $g$ of full-range luma after 2 × 2 average pooling, $\mathrm{GMS}=(2g_1g_2+T)/(g_1^2+g_2^2+T)$, $T=170$, GMSD = standard deviation of GMS. An established edge-sensitive metric that does not depend on our choices. | ↓ |
| Edge-PSNR-Y at τ = 0.25 and 1.0 | Sensitivity analysis for the threshold. | ↑ |
| LPIPS (optional) | Learned perceptual distance (Zhang et al., 2018), AlexNet features, `--lpips`. | ↓ |

**The threshold τ.** With the unnormalised Sobel kernel, a step of *h* grey levels in Y gives a
magnitude of 4*h*/255; τ = 0.5 corresponds to a step of about 32 levels, which selects line art and
strong contours but not shading or JPEG noise. Gate check R1.7 reports the share of edge pixels
for τ ∈ {0.15, 0.25, 0.5, 1.0} on the validation set; τ is locked (in `configs/base.json` and the
protocol lock) before any training result is seen.

**Why Edge-PSNR is not enough on its own.** It is a metric we defined, and a loss that targets
edges could be tuned to it. GMSD and the τ sensitivity analysis guard against that; PSNR-Y and
SSIM-Y show whether edge gains cost global fidelity.

## 5. Optimisation

| Setting | Value | Note |
|---|---|---|
| Iterations | 10,000 ≈ 23 epochs (set by the GPU budget, gate R6.1: 32 img/s on a T4, 2.8 h per run) | the same for every condition |
| Batch | 32 LR images of 64 × 64 | whole images, no patch cropping |
| Optimiser | Adam, β = (0.9, 0.999), lr 2·10⁻⁴ | as in the official SwinIR training code (KAIR) |
| Schedule | cosine from 2·10⁻⁴ to 10⁻⁷ over all iterations, per step | a function of the step, so resuming cannot change it; official: step decay over 500 k iterations |
| Gradient clipping | global norm 1.0 | kept from the notebook; official: none |
| Stochastic depth | 0.1 | SwinIR default |
| Augmentation | random flip/rotation (one of 8 dihedral transforms), same for LR and HR | |
| Precision | float16 autocast on CUDA, loss and metrics in float32/float64 | gate R4.6 compares with float32 |
| Validation | every 2,000 iterations on the full validation split | used only for curves and λ selection |
| Final model | the last iteration (no early stopping, no weight EMA) | avoids selecting on noisy validation points; official SwinIR evaluates an EMA (0.999) of the weights |

**Deviations from the official recipe** (shorter schedule, cosine decay, clipping, no EMA, batch 32
instead of 64) are forced by the compute budget and apply identically to every condition, so they
cannot favour one loss. They do mean that absolute PSNR values are below what a 500 k-iteration
SwinIR reaches; the paper should say so.

Same seed ⇒ same initial weights, same data order and same augmentations for every condition,
so runs with the same seed differ *only* in the loss. This is what makes the paired statistics
in EXPERIMENTS.md valid.

## 6. Code map

| Concept | Code |
|---|---|
| Degradation, split, deduplication, manifest | `edgesr/data.py: prepare_dataset, make_pair, dhash` |
| Dataset, augmentation, resumable sampler | `edgesr/data.py: SRPairDataset, dihedral, TrainSampler` |
| Network presets | `edgesr/models/__init__.py` |
| Sobel, adaptive weight, loss | `edgesr/losses.py` |
| Metrics | `edgesr/metrics.py` |
| Training loop, checkpoint, evaluation, bicubic | `edgesr/engine.py` |
| Plan, λ selection rule, grid extension | `edgesr/ablation.py` |
| Bootstrap, Wilcoxon, Holm | `edgesr/stats.py` |
| Readiness gate | `edgesr/readiness.py` |

## References

- Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R. *SwinIR: Image Restoration Using Swin Transformer.* ICCV Workshops, 2021. arXiv:2108.10257.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV, 2021.
- Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 13(4), 2004.
- Xue, W., Zhang, L., Mou, X., Bovik, A. C. *Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index.* IEEE TIP 23(2), 2014.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR, 2018.
- Ma, C., Rao, Y., Cheng, Y., Chen, C., Lu, J., Zhou, J. *Structure-Preserving Super Resolution with Gradient Guidance.* CVPR, 2020.
- Abrahamyan, L., Truong, A. M., Philips, W., Deligiannis, N. *Gradient Variance Loss for Structure-Enhanced Image Super-Resolution.* ICASSP, 2022.
- Zhao, H., Gallo, O., Frosio, I., Kautz, J. *Loss Functions for Image Restoration with Neural Networks.* IEEE TCI 3(1), 2017.
