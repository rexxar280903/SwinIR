# Edge-Aware Loss for SwinIR Super-Resolution of Anime Faces

Does telling a super-resolution network *where the edges are* make it draw sharper line art?
This repository trains the unmodified **SwinIR** transformer (Liang et al., 2021) for ×2
super-resolution of anime faces with three objectives, an L1 pixel loss, L1 plus a **static**
Sobel edge loss, and L1 plus an **adaptive** Sobel edge loss that weights each pixel by the
strength of the ground-truth edge, and compares them in a pre-registered, multi-seed ablation.

> **Status (2026-10-05): code and protocol ready, results pending.** The readiness gate passes on
> CPU with synthetic data; the full ablation has not been run yet. No numbers are reported here
> until it has. An earlier notebook version was audited and found unusable for results; see
> [docs/AUDIT.md](docs/AUDIT.md).

## What is compared

| Condition | Training objective |
|---|---|
| Bicubic | no learning: PIL bicubic upsampling |
| L1 | $\lVert\hat y-y\rVert_1$ (the SwinIR default) |
| Static Sobel | $\lVert\hat y-y\rVert_1+\lambda\,\mathrm{mean}\lvert S(\hat y)-S(y)\rvert$ |
| Adaptive Sobel | $\lVert\hat y-y\rVert_1+\lambda\,\mathrm{mean}\big(w\odot\lvert S(\hat y)-S(y)\rvert\big)$, $w$ = per-image min–max of $S(y)$ |

$S$ is the Sobel gradient magnitude (replicate padding). The network, data, schedule and seeds
are identical across conditions; only the loss changes. λ is swept for both edge losses and
chosen on validation data, so static and adaptive are compared at their own best λ.

**Metrics** (per image, SR rounded to 8 bit, 2-pixel border crop, as in SwinIR's test code):
PSNR-Y and SSIM-Y (fidelity), **Edge-PSNR-Y** (PSNR on pixels where the HR Sobel magnitude
exceeds τ = 0.5; primary endpoint), GMSD (an established gradient-based quality index), and
optionally LPIPS. Definitions: [docs/METHOD.md](docs/METHOD.md).

## Design in one table

| | |
|---|---|
| Data | Kaggle *anime faces waifu2x*; HR 128², LR 64² by bicubic; exact duplicates removed, near-duplicates kept within one split; 80/10/10 train/val/test |
| Model | SwinIR-light (0.91 M parameters), official code, unchanged |
| Training | 10,000 iterations × batch 32 (≈ 23 epochs), Adam 2·10⁻⁴, cosine decay, flips/rotations, fp16 on GPU |
| Stage 1 | seed 0: L1 + 5 λ values × {static, adaptive} = 11 runs (+ a pre-registered grid extension if λ* lands on the edge of the grid) |
| Stage 2 | seeds 1–3: L1, static(λ*), adaptive(λ*) = 9 runs |
| Hypotheses | H1 adaptive > L1 and H2 adaptive > static on Edge-PSNR-Y; H3 adaptive non-inferior to L1 on PSNR-Y (0.05 dB) |
| Statistics | paired two-level bootstrap over seeds and test images, Wilcoxon + Holm, all seeds must agree |

Full protocol, decision rules and threats to validity: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Reproduce

Everything runs on a free Kaggle GPU. The notebook
[`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) does all steps below; the same
commands work anywhere with PyTorch ≥ 2.3.

```bash
# 1. data (≈ 3 min): splits, duplicate grouping, manifest
python scripts/prepare_data.py --src /kaggle/input/anime-faces-waifu2x --out data

# 2. readiness gate (≈ 10 min): data integrity, leakage, loss/metric identities,
#    exact resume, a dry run of the whole pipeline, GPU-hour budget, protocol lock
python scripts/readiness_check.py --data data --out readiness --gpu-hours 25 --require-gpu

# 3. the ablation (resumable; --shard k/n splits it over GPUs or sessions)
python scripts/run_ablation.py --data data --runs runs --stage all

# 4. tables, hypothesis tests and figures -> runs/analysis/RESULTS.md
python scripts/analyze.py --runs runs --data data
```

Single runs and evaluations:

```bash
python scripts/train.py --data data --runs runs --set edge_mode=adaptive edge_weight=0.2 seed=1 --test
python scripts/evaluate.py --data data --bicubic --out runs/bicubic
python scripts/inspect_checkpoint.py runs/light_x2_l1_s1/model_final.pt
```

Tests (CPU, ≈ 2 min): `python -m pytest tests -q`, or `python tests/test_core.py` without pytest.
A local check of the whole pipeline on synthetic images:
`python scripts/readiness_check.py --synthetic 80 --out tmp/readiness --quick`.

## Results

Pending. `scripts/analyze.py` writes `RESULTS.md` with the main table (mean ± SD over three
seeds), the λ sweep, the hypothesis table and the figures; they will be added here unchanged
once the ablation is complete, whether the hypotheses are supported or not.

## Repository layout

```
edgesr/                   the package
  models/swinir.py        official SwinIR network (unchanged)
  models/__init__.py      presets: light / medium / classical
  losses.py               Sobel operator, static & adaptive edge loss
  metrics.py              PSNR, SSIM, Edge-PSNR, GMSD (Y channel, 8-bit, border crop)
  data.py                 dataset preparation, paired dataset, resumable sampler
  engine.py               training loop, checkpoint/resume, evaluation, bicubic baseline
  ablation.py             run plan, validation-only λ selection, grid extension rule
  stats.py                paired two-level bootstrap, Wilcoxon, Holm
  readiness.py            the readiness gate (R0–R7)
scripts/                  command-line entry points (prepare, train, evaluate, ablation, analyze, gate)
configs/base.json         the locked training/metric settings
tests/test_core.py        unit tests (also runnable without pytest)
notebooks/kaggle_runner.ipynb      end-to-end Kaggle notebook
notebooks/legacy/swinir-l1.ipynb   the original notebook, kept for reference
docs/METHOD.md            definitions: data, network, loss, metrics, optimisation
docs/EXPERIMENTS.md       pre-registered protocol and readiness gate
docs/AUDIT.md             what was wrong with the original notebook and how it was fixed
docs/PANDUAN_ID.md        full walkthrough in Indonesian
```

## Limitations

* The dataset has no description and an unknown licence. Its name suggests that the images were
  upscaled with waifu2x, in which case the "ground truth" contains synthetic detail
  ([AUDIT.md §D](docs/AUDIT.md#d-open-questions-about-the-data-to-verify-not-yet-known)). The
  images are not redistributed here.
* Bicubic degradation only, faces only, 128 × 128 HR, ×2. No claim is made about real-world
  degradations, full illustrations or larger scales.
* The lightweight SwinIR is trained for 10 k iterations to fit a free GPU quota, so absolute
  PSNR is below a fully trained model; the comparison between losses is the object of study.
* Gradient losses for SR exist (e.g. Ma et al., 2020; Abrahamyan et al., 2022). The question here
  is the effect of edge-strength weighting under a controlled protocol, not a new loss family.

## Citation and acknowledgements

The network code is from the official [SwinIR repository](https://github.com/JingyunLiang/SwinIR)
(Apache License 2.0), originally by Ze Liu and modified by Jingyun Liang. If you use this
repository, please cite SwinIR:

```bibtex
@inproceedings{liang2021swinir,
  title     = {SwinIR: Image Restoration Using Swin Transformer},
  author    = {Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle = {IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages     = {1833--1844},
  year      = {2021}
}
```

Citation metadata for this repository: [CITATION.cff](CITATION.cff).
