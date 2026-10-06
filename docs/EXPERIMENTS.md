# Experimental protocol (pre-registered)

**Status:** draft, to be locked with `configs/protocol_lock.json` before the first sweep run.
Everything in this file is decided *before* any training result is seen. Changes after the lock
go into the deviation log at the end, with date and reason.

Definitions of the loss and the metrics: [METHOD.md](METHOD.md). Why the protocol looks like
this (what went wrong in the first version): [AUDIT.md](AUDIT.md).

## 1. Questions

| | Question | Answered by |
|---|---|---|
| RQ1 | Does adding a Sobel edge term to the L1 loss improve the reconstruction of edges by SwinIR on anime faces? | H1 (adaptive vs L1), secondary: static vs L1 |
| RQ2 | Does weighting the edge term by ground-truth edge strength (adaptive) beat the unweighted edge term (static) when both are tuned the same way? | H2 |
| RQ3 | What does the edge term cost in global fidelity? | H3 (PSNR-Y non-inferiority), SSIM-Y |
| RQ4 | How sensitive are the results to λ and to the Edge-PSNR threshold τ? | weight sweep, τ ∈ {0.25, 0.5, 1.0} (descriptive) |

## 2. Hypotheses and decision rules

Primary endpoint: **Edge-PSNR-Y** on the test set (τ = 0.5). Comparisons use the confirmation
runs (seeds 1, 2, 3) at the λ* chosen on validation data.

| | Hypothesis | Supported if |
|---|---|---|
| **H1** | adaptive(λ*) > L1 on Edge-PSNR-Y | 95% CI of the mean paired difference > 0 **and** Holm-adjusted p < 0.05 **and** the difference is positive for each of the 3 seeds |
| **H2** | adaptive(λ*) > static(λ*) on Edge-PSNR-Y | same rule |
| **H3** | adaptive(λ*) is non-inferior to L1 on PSNR-Y, margin 0.05 dB | lower end of the 95% CI of (adaptive − L1) > −0.05 dB |

* **CI**: two-level paired bootstrap, 10,000 replicates. Each replicate resamples the 3 seeds with
  replacement (the same seeds for both conditions, since equal seeds share initial weights and
  data order) and, independently, the test images with replacement (`edgesr/stats.py`).
* **p**: Wilcoxon signed-rank test on per-image differences averaged over seeds; Holm correction
  over the family {H1, H2}. H3 is a non-inferiority question and uses the CI only.
* Requiring all three seeds to agree is deliberately strict: with three seeds, one seed can
  carry a mean difference.
* All metrics and all comparisons (including static vs L1, SSIM-Y, GMSD and the τ sensitivity
  analysis) are reported in full whatever the outcome, labelled *secondary* and uncorrected.
* A non-supported hypothesis is a result, not a failure. It is reported with the same tables.

## 3. Data

| Item | Value |
|---|---|
| Source | Kaggle `mcparadip/anime-faces-waifu2x` (licence: unknown; see AUDIT.md §D) |
| Preparation | `scripts/prepare_data.py`: RGBA → white, centre square, HR 128 × 128, LR 64 × 64, PIL bicubic |
| Duplicates | exact duplicates (identical HR pixels) dropped; images with the same 64-bit dHash kept in one split |
| Split | 80 / 10 / 10 by dHash group, seed 42, from the sorted file list |
| Record | `manifest.csv` (key, split, source path, size, mode, SHA-1, dHash) and `dataset_info.json`; the SHA-1 of the manifest is part of the protocol lock |
| Test split access | only `evaluate_run` / `evaluate_bicubic` (`allow_test=True`); training and λ selection cannot open it |

## 4. Runs

All runs share `configs/base.json` (model `light`, 10,000 iterations × batch 32, see METHOD.md §5).
A run is identified by `{model}_x2_{condition}_s{seed}`.

| Stage | Seeds | Conditions | Runs |
|---|---|---|---|
| Bicubic | – | PIL bicubic upsampling, no training | 0 (evaluation only) |
| 1 · Sweep | 0 | L1; static λ ∈ {0.05, 0.1, 0.2, 0.5, 1.0}; adaptive λ ∈ {0.05, 0.1, 0.2, 0.5, 1.0} | 11 |
| 1b · Extension (conditional) | 0 | if a mode's λ* is the largest grid value: add λ ∈ {2, 5}; if the smallest: add λ ∈ {0.02, 0.01}. At most once per mode | 0–4 |
| 2 · Confirmation | 1, 2, 3 | L1; static(λ*_static); adaptive(λ*_adaptive) | 9 |

Total: 20–24 training runs. The sweep seed (0) is not reused for confirmation, so a λ that won
the sweep by a lucky seed gets no advantage in the test comparison.

### Weight selection (validation only)

Implemented in `edgesr/ablation.py: select_weights`; the result is saved as `runs/selection.json`.

1. For every sweep run, take the mean of the last 3 validation points (6 k, 8 k and 10 k for 10 k iterations)
   of Edge-PSNR-Y and of PSNR-Y.
2. A λ is *eligible* if its PSNR-Y is at most 0.10 dB below the L1 run of the same seed.
3. λ* = the eligible λ with the highest Edge-PSNR-Y; differences below 0.01 dB count as ties and
   go to the smaller λ.
4. If no λ is eligible, λ* = the λ with the highest PSNR-Y (reported as a note).
5. The boundary rule (stage 1b) is applied to the base grid only; after the extension, steps 1–4
   are repeated on the extended grid.

## 5. Evaluation

* The final weights of each run (iteration 10,000) are evaluated once on the test split, in
  float32, and the per-image values are written to `test_per_image.csv`.
* Metrics: PSNR-Y, SSIM-Y, Edge-PSNR-Y (τ = 0.5; 0.25 and 1.0 as sensitivity), GMSD, PSNR-RGB,
  SSIM-RGB, edge coverage; LPIPS if the package is available (secondary).
* `scripts/analyze.py` produces every table and figure for the paper from these files:
  main table (mean ± SD over seeds), sweep table with edge share, hypothesis table, all paired
  differences, validation curves, and a sample figure with the first test images in key order
  (not hand-picked).

## 6. Threats to validity

| Threat | Consequence | Mitigation / how it is reported |
|---|---|---|
| Targets may be waifu2x outputs (AUDIT.md §D) | "HR" detail is partly synthetic; models learn to imitate waifu2x | All conditions share the targets, so the comparison holds; the data section states the source and its likely processing |
| One dataset, faces only, 128 px | Results may not transfer to full anime frames, other line art, or larger images | Stated as scope; no claim beyond anime faces at ×2 |
| Bicubic degradation only | Not real-world SR | Stated as scope (classical SR setting) |
| `light` model, 10 k iterations | Absolute PSNR below a fully trained SwinIR; the loss effect might differ for larger models | Budget measured in the gate; the medium preset's cost is reported; claims limited to the trained setting |
| Three seeds | Coarse estimate of training variance | Two-level bootstrap, per-seed differences shown, all-seeds-agree rule |
| Edge-PSNR is our own metric | Risk of a metric that favours our loss | GMSD (established), τ sensitivity, PSNR/SSIM reported side by side |
| λ chosen on one seed | Selection noise | Confirmation on new seeds; the whole sweep is reported |
| Many secondary comparisons | Chance findings | Only H1–H3 carry decisions; secondary results are labelled uncorrected |

## 7. Compute budget

Gate check R6.1 measures, on the Kaggle GPU, the training throughput of the model (images/s),
the data-loader throughput and the evaluation cost, and reports the projected GPU hours of the
whole plan and the largest `total_iters` that fits the budget given with `--gpu-hours`. If the
plan does not fit, `total_iters` in `configs/base.json` is lowered **before** the lock (it is the
same for every condition).

Measured on Kaggle (2026-10-06, Tesla T4, PyTorch 2.11): `light` trains at 32 img/s with a peak of
12.3 GB; 20,000 iterations would take 5.6 h per run and about 112 GPU hours for 20 runs. `medium`
does not fit in T4 memory at batch 32. `total_iters` was therefore set to 10,000 (≈ 23 epochs,
2.8 h per run, about 57 GPU hours, i.e. about 30 notebook hours with two T4s in parallel) before
the lock. With two T4 GPUs, `run_ablation.py --shard 1/2` and `--shard 2/2` run
two runs at once (cell 6 of `notebooks/kaggle_runner.ipynb`).

## 8. Readiness gate

`scripts/readiness_check.py` must report **READY** (or **READY WITH WARNINGS**, each warning
resolved or acknowledged in writing) on the Kaggle GPU with the full dataset before stage 1 starts.

| ID | Check | Passes when |
|---|---|---|
| R0.1–R0.4 | Python/PyTorch, GPU, packages, disk | PyTorch ≥ 2.3; GPU visible (`--require-gpu`); enough free disk for all runs |
| R1.1 | Manifest vs files | every split non-empty and identical to the manifest; not a `--limit` pilot |
| R1.2 | Image sizes | HR 128², LR 64², RGB |
| R1.3 | Degradation | LR == PIL-bicubic(HR) bit-exactly on sampled pairs |
| R1.4 | Leakage | no identical or dHash-identical HR image in two splits; files re-hash to the manifest |
| R1.5 | Split ratios | within ±2 % of 80/10/10 |
| R1.6 | Sources | no source smaller than 128 px (else the "HR" would be upsampled); sizes recorded |
| R1.7 | Edge-mask coverage | mean coverage at τ between 2 % and 40 %, < 1 % of images without edges |
| R2.1–R2.5 | Sobel, adaptive weight, losses, metric identities, scikit-image cross-check | exact identities hold (e.g. Edge-PSNR = PSNR for a uniform error) |
| R3.1–R3.2 | Model | parameter count as expected; network code identical to the official SwinIR |
| R4.1 | Overfit | each loss mode reduces the loss on 8 images by > 40 % |
| R4.2 | Gradient flow | every parameter gets a finite, non-zero gradient |
| R4.3 | Resume | 4 + resume + 4 steps == 8 straight steps, bit-exact (CPU) |
| R4.4 | Seeds | a seed fixes init, data order and augmentation |
| R4.5 | Test isolation | the test split cannot be opened by training code |
| R4.6 | AMP | float16 training tracks float32 within 10 % over 30 steps |
| R5.1 | Dry run | the real CLI runs sweep → bicubic → extension → confirmation → analysis on tiny data |
| R5.2 | Bicubic on validation | all metrics finite, PSNR-Y in a plausible range |
| R6.1 | GPU budget | projected GPU hours ≤ budget and one run < 12 h |
| R7.1 | Protocol lock | `protocol_lock.json` written; once committed, later runs must match it |

## 9. Deviation log

| Date | Deviation | Reason | Effect on conclusions |
|---|---|---|---|
| – | – | – | – |
