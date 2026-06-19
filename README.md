# SwinIR + Adaptive Edge-Aware Loss for Anime Face Super-Resolution

An *image super-resolution* (SR) implementation based on **SwinIR** (Swin Transformer for Image Restoration), modified with an **adaptive edge-based (Sobel) loss function** to improve line/outline sharpness in anime images, evaluated on the *Anime Faces (Waifu2x)* dataset.

> Backbone architecture follows the paper:
> **Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCV Workshops, 2021.**
> [arXiv:2108.10257](https://arxiv.org/abs/2108.10257) · [Official Repo](https://github.com/JingyunLiang/SwinIR)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Model Architecture (SwinIR)](#-model-architecture-swinir)
- [Contributions / Modifications in This Project](#-contributions--modifications-in-this-project)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Loss Function](#-loss-function)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Experiment Configuration](#-experiment-configuration)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Checkpoint & Metadata](#-checkpoint--metadata)
- [Limitations & Technical Notes](#-limitations--technical-notes)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🧩 Overview

This project fully reimplements the **SwinIR** architecture (with no structural modifications) as a ×2 *super-resolution* backbone, then adds a **gradient-based (edge-aware) loss component** to encourage the model to produce sharper edges/outlines — an important characteristic for anime images, which typically feature crisp *line art*. In addition, a dedicated **Edge-PSNR** evaluation metric is introduced to specifically measure reconstruction quality at image edges, beyond the standard PSNR/SSIM metrics.

---

## 🏗️ Model Architecture (SwinIR)

The backbone used is the **original SwinIR implementation** (RSTB + STL) with no structural changes, consisting of three main modules:

```
Input (LQ) ──► Shallow Feature Extraction (Conv 3×3)
                        │
                        ▼
              Deep Feature Extraction
        ┌─────────────────────────────────┐
        │  RSTB → RSTB → ... → RSTB → Conv │   (K Residual Swin
        └─────────────────────────────────┘    Transformer Blocks)
                        │   (+ long skip connection from shallow feature)
                        ▼
              HQ Image Reconstruction
                  (Sub-pixel Conv / PixelShuffle)
                        │
                        ▼
                  Output (SR Image)
```

**Residual Swin Transformer Block (RSTB):**
- Composed of several **Swin Transformer Layers (STL)** + 1 convolutional layer + residual connection.
- STL uses **Window-based Multi-head Self-Attention (W-MSA)** with *relative position bias*, along with **shifted windows** alternating between layers to enable cross-window interaction.
- Each STL: `LayerNorm → MSA → residual → LayerNorm → MLP (GELU) → residual`.

**Model configuration used in this project** (`model_size = 'large'`):

| Parameter | Value |
|---|---|
| Upscale factor | ×2 |
| Window size | 8 |
| Embedding dim | 180 |
| Depths (STL count per RSTB) | [6, 6, 6, 6] |
| Number of attention heads | [6, 6, 6, 6] |
| MLP ratio | 2 |
| Upsampler | `pixelshuffle` |
| Residual connection in RSTB | `1conv` |
| Input size (LR) | 64×64 → Output (HR) 128×128 |

Module implementations (`Mlp`, `WindowAttention`, `SwinTransformerBlock`, `RSTB`, `SwinIR`, etc.) are adapted directly from the official SwinIR code (credit: Ze Liu, modified by Jingyun Liang), **with no changes to the core architecture**.

---

## 🚀 Contributions / Modifications in This Project

What differentiates this project from baseline SwinIR (the original paper uses only **L1 pixel loss** for classical SR) lies in the **loss function and evaluation metrics**, not the architecture:

### 1. Sobel-based Edge Extraction Module
A `SobelOperator` module that extracts *gradient magnitude* from an image using horizontal and vertical Sobel kernels, applied *depthwise* (per-channel) via `F.conv2d` with `groups=C`. This module forms the basis of every edge-based loss variant below.

### 2. Sobel Loss
An additional loss that compares the predicted *edge map* against the ground-truth *edge map*, instead of comparing raw pixels:

```
L_sobel = w · ‖Sobel(I_pred) − Sobel(I_target)‖₁
```

### 3. Combined Loss (Pixel + Sobel, static weighting)
A combination of pixel loss (selectable as L1 / Charbonnier / MSE) with a fixed-weight Sobel loss:

```
L_combined = w_pixel · L_pixel + L_sobel
```

### 4. ⭐ Adaptive Sobel Loss (main loss used)
**This is the project's main contribution.** Unlike standard Sobel loss, the per-pixel contribution to the edge loss is **dynamically adjusted based on the strength of the ground-truth edge itself** (min-max normalized), so that areas with sharp edges/lines are penalized more heavily than flat areas:

```
edge_weight = (E_target − min(E_target)) / (max(E_target) − min(E_target) + ε)
L_edge      = mean( edge_weight · |E_pred − E_target| )
L_total     = w_pixel · L_pixel(L1) + w_sobel · L_edge
```

where `E(·) = Sobel(·)`.

**Motivation:** in anime images, outline edges (sharp boundaries) are perceptually far more important than fine texture in flat areas (skin, plain backgrounds). This adaptive weighting pushes the model to prioritize edge reconstruction without sacrificing training stability in non-edge regions.

### 5. Edge-PSNR — Additional Evaluation Metric
In addition to standard PSNR and SSIM, an **Edge-PSNR** metric is introduced: PSNR computed **only on pixels identified as edges** (based on an *edge mask* derived from the ground-truth HR image, using a specific Sobel magnitude threshold). This metric provides a more specific view of how well the model reconstructs line detail, which is not adequately captured by global PSNR.

```
mask        = 1[ Sobel_magnitude(I_HR) > threshold ]
Edge-MSE    = Σ( mask · (I_SR − I_HR)² ) / Σ(mask)
Edge-PSNR   = 10 · log10( 1 / Edge-MSE )
```

| Aspect | SwinIR (Original Paper) | This Project |
|---|---|---|
| Architecture | RSTB + STL | **Same (unchanged)** |
| Loss (classical SR) | L1 pixel loss only | L1 pixel loss + **Adaptive Sobel Edge Loss** |
| Evaluation metrics | PSNR, SSIM | PSNR, SSIM, **Edge-PSNR** |
| Data domain | Natural images (DIV2K, etc.) | **Anime faces** (Waifu2x dataset) |
| Scale | ×2 / ×3 / ×4 | ×2 |

---

## 📂 Dataset & Preprocessing

- **Source:** [Anime Faces (Waifu2x)](https://www.kaggle.com/datasets) — an anime character face dataset.
- **Preprocessing:**
  - HR images are resized to **128×128** (bicubic interpolation).
  - LR images are generated from HR via downsampling to **64×64** (bicubic), giving a scale factor of ×2.
  - Data split: **90% train / 10% test** (`train_test_split`, `random_state=42`).
  - Stored in a `train/{HR,LR}` and `test/{HR,LR}` folder structure as image pairs with identical filenames.
- **Dataset classes:** `SRDataset` (paired LR–HR from separate folders) and `SingleFolderDataset` (LR generated on-the-fly from a single HR folder) — both return tensors normalized to the `[0, 1]` range.

> ⚠️ Note: in the included experiment configuration, there is a `LIMIT_DATA` parameter that caps the amount of data processed (useful for *quick testing*/pipeline debugging). Make sure this value is adjusted (or set to `None`) before full-scale training.

---

## 🎯 Loss Function

The loss is selected via the `config.loss_type` parameter, with 4 options: `'l1'`, `'sobel'`, `'combined'`, `'adaptive'`.

**Configuration used in the final experiment:**
```python
loss_type     = 'adaptive'   # AdaptiveSobelLoss
pixel_weight  = 1.0
sobel_weight  = 0.1          # recommended 0.1–0.5 for the anime domain
```

The base pixel loss uses **L1**, chosen to stay consistent with the SR training convention from the SwinIR paper, before adding the adaptive edge component.

---

## 📊 Evaluation & Metrics

| Metric | Description |
|---|---|
| **PSNR** | Computed globally per image using `skimage.metrics.peak_signal_noise_ratio`, averaged per batch. |
| **SSIM** | Computed globally using `skimage.metrics.structural_similarity`. |
| **Edge-PSNR** | PSNR restricted to edge regions only (see [novelty section](#5-edge-psnr--additional-evaluation-metric)). |

Validation runs at the end of every epoch (`validate()`), with SR output *clamped* to `[0, 1]` before metrics are computed.

---

## ⚙️ Experiment Configuration

Managed via the `AnimeConfig` class:

```python
scale          = 2
window_size    = 8
model_size     = 'large'        # embed_dim=180, depths=[6,6,6,6], heads=[6,6,6,6]
upsampler      = 'pixelshuffle'

loss_type      = 'adaptive'
pixel_weight   = 1.0
sobel_weight   = 0.1

batch_size     = 8
num_epochs     = 2
learning_rate  = 2e-4
betas          = (0.9, 0.999)
optimizer      = Adam
scheduler      = CosineAnnealingLR (eta_min=1e-7)
grad_clip_norm = 1.0
```

A `main_finetuning()` function is also available for **resuming training from a checkpoint** or **fine-tuning** with a new learning rate, including full restoration of optimizer & scheduler state.

---

## 🗂️ Project Structure

> *(matches the cell structure of the notebook — adjust accordingly if the code is split into separate `.py` files)*

```
.
├── swinir-l1.ipynb          # Main notebook (model, training, evaluation)
├── dataset_split/            # Preprocessing output (train/test, HR/LR)
├── experiments_anime/        # Experiment output & logs
└── model_epoch_*_*.pth       # Training checkpoints
```

Key modules within the notebook:
- **Model:** `SwinIR`, `RSTB`, `SwinTransformerBlock`, `WindowAttention`, `Mlp`
- **Loss:** `SobelOperator`, `SobelLoss`, `CombinedLoss`, `AdaptiveSobelLoss`
- **Data:** `SRDataset`, `SingleFolderDataset`, `build_dataloaders`
- **Utilities:** `AverageMeter`, `calculate_psnr`, `calculate_ssim`, `calculate_edge_psnr`, `get_edge_mask`
- **Training:** `build_model`, `build_loss`, `train_epoch`, `validate`, `main`, `main_finetuning`

---

## ▶️ How to Run

```bash
pip install timm
```

1. **Preprocess the dataset** — run the *Resize & Split Dataset* cell to build LR–HR pairs from the raw dataset.
2. **Train from scratch:**
   ```python
   main()
   ```
3. **Resume / fine-tune from a checkpoint:**
   ```python
   main_finetuning(
       checkpoint_path="model_epoch_2_adaptive.pth",
       load_optimizer=True,
       new_lr=None,          # None = continue with the checkpoint's LR
       additional_epochs=2,
       new_batch_size=8
   )
   ```
4. **Check checkpoint metadata** without loading the full model:
   ```python
   read_checkpoint_metadata("model_epoch_2_adaptive.pth")
   ```

---

## 💾 Checkpoint & Metadata

Each checkpoint (`.pth`) stores, in addition to `model_state_dict`/`optimizer_state_dict`/`scheduler_state_dict`:
- Full hyperparameters (scale, window size, loss type, pixel/sobel weight, etc.)
- Training & validation metrics per epoch (`train_loss`, `pixel_loss`, `edge_loss`, `val_psnr`, `val_ssim`, `val_edge_psnr`)
- Experiment metadata (timestamp, device, epoch duration, institution)

---

## ⚠️ Limitations & Technical Notes

- **Experiment scale:** the example configuration uses `num_epochs = 2` with a limited `LIMIT_DATA` — suitable for a *pipeline sanity check*, but full quantitative results require training on the complete dataset with a representative number of epochs.
- **Vertical Sobel kernel:** please double-check the last row of the `sobel_y` kernel in `SobelOperator` — the standard vertical Sobel kernel is `[[-1,-2,-1],[0,0,0],[1,2,1]]`, but the current implementation has `[1,2,3]` as the last row. This should be corrected before being reported as a final result, as it will slightly shift the detected gradient direction.
- **Architecture unchanged:** all performance gains (if any) come from the loss/training-objective side, not from structural changes to SwinIR. This should be stated explicitly in the methodology section to keep the contribution claim accurate (contribution to the *training objective* and *evaluation metric*, not the *network architecture*).

---

## 🔭 Future Work

- Systematic ablation experiments across loss variants (`l1` vs `sobel` vs `combined` vs `adaptive`) with an equal number of epochs for a fair comparison.
- Validation on other scale factors (×3, ×4).
- Sensitivity study on `sobel_weight` and `edge_threshold` for Edge-PSNR.
- Fix the vertical Sobel kernel in `SobelOperator`.

---

## 📚 References

```bibtex
@inproceedings{liang2021swinir,
  title     = {SwinIR: Image Restoration Using Swin Transformer},
  author    = {Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle = {IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2021}
}
```

- Liu, Ze, et al. *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."* arXiv:2103.14030, 2021.
- Charbonnier, P., et al. *"Two deterministic half-quadratic regularization algorithms for computed imaging."* ICIP, 1994.

---

## 📝 License & Attribution

The SwinIR architecture implementation is adapted from the [official repository](https://github.com/JingyunLiang/SwinIR) (Ze Liu, modified by Jingyun Liang). Please include appropriate attribution if this code is reused, along with a citation to the original SwinIR paper above.
