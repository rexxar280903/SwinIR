# Audit of the original notebook (v0)

The first version of this project was a single Kaggle notebook, now kept unchanged at
[`notebooks/legacy/swinir-l1.ipynb`](../notebooks/legacy/swinir-l1.ipynb). Before running a full
ablation, the notebook was read line by line. This document lists every problem found, the
evidence, what it would have done to the results, and where the fix lives.

**Summary.** The notebook never produced a usable result: its last run used 10 images and
2 epochs, and its L1 baseline could not run on a GPU. More importantly, three defects would have
invalidated a full run even with all the data: the "edge" loss responded to brightness (kernel typo),
there was no validation split, and the edge metric was biased by a constant −4.77 dB. All issues
below are fixed in the `edgesr/` package and most are now guarded by an automated check in the
readiness gate (`scripts/readiness_check.py`, IDs in the last column).

Numbers marked † come from `python scripts/audit_legacy.py` on 64 synthetic cartoon images and
are illustrative. Run the same script with `--data` on the real dataset to get the values for a
paper.

## A. Critical: results would be wrong or missing

| ID | Problem | Evidence | Effect | Fix | Check |
|---|---|---|---|---|---|
| A1 | **Vertical Sobel kernel typo.** `sobel_y` ended in `[1, 2, 3]` instead of `[1, 2, 1]`. | The kernel sums to 2, so a flat region of intensity 0.7 has "edge magnitude" 1.40 instead of 0. The notebook's adaptive weight correlated about as much with brightness (r = 0.50†) as with true edge strength (r = 0.58†); flat pixels got mean weight 0.25† vs 0.42† on edges. | The "edge" loss was partly a brightness-weighted pixel loss, and the adaptive weighting emphasised bright flat areas (e.g. white backgrounds). Any conclusion about edge-aware training would have been about a different loss. | `sobel_y = sobel_x.T`, defined once in `edgesr/losses.py`. With the fix, flat pixels get weight 0.0003† vs 0.56† on edges. | R2.1 |
| A2 | **L1 baseline crashed on GPU.** In `train_epoch`, the non-edge branch called `criterion(sr_imgs, hr_imgs)` with `hr_imgs` still on the CPU. | `sr_imgs` lives on CUDA; PyTorch raises a device-mismatch error. The same branch also left the pixel/edge meters at 0. | The baseline that every claim is measured against could not be trained. | One code path for all losses (`EdgeAwareLoss`, `edge_mode="none"` is plain L1). | R4.1 trains all three modes |
| A3 | **10 images, 2 epochs.** `LIMIT_DATA = 10` (9 train / 1 test) and `num_epochs = 2`. | Cell 3 and `AnimeConfig`. | Every number the notebook could print was a pipeline smoke test, not a result. | Full dataset by default; `--limit` exists only for pilots and is flagged as "PILOT ONLY" by the gate. | R1.1 |
| A4 | **No validation split.** Only train/test; validation ran on the test images every epoch. | `build_dataloaders` used `test/` as `val_*_dir`. | Choosing λ, the epoch or the checkpoint on test images makes the reported test numbers optimistic. | 80/10/10 split. The test split refuses to load unless `allow_test=True`, which only the final evaluation passes; λ selection reads `val_log.csv` only. | R4.5 |

## B. Major: biased numbers or confounded comparisons

| ID | Problem | Evidence | Effect | Fix | Check |
|---|---|---|---|---|---|
| B1 | **Edge-PSNR biased by −4.77 dB.** The mask had shape `[B,1,H,W]`, the squared error `[B,3,H,W]`; the sum ran over 3 channels but was divided by the number of masked pixels. | MSE was exactly 3× too large: Edge-PSNR = true value − 10·log10(3) = −4.771 dB†. It was also pooled over a batch instead of per image, then averaged per batch. | Ranking between methods is unaffected (constant offset), but the values cannot be compared with PSNR or with any other work, and a reader would conclude that edges are reconstructed much worse than they are. | Per-image Edge-PSNR on Y; MSE averaged over masked pixels (`edgesr/metrics.py`). A uniform error now gives Edge-PSNR = PSNR. | R2.4 |
| B2 | **Zero padding in Sobel.** Both the loss and the metric mask used `padding=1` (zeros). | Every border pixel becomes an "edge": 100%† of border pixels were in the notebook's mask, which made up 19%† of all mask pixels. | The adaptive loss gave image borders the highest weight; Edge-PSNR was partly a border metric. | Replicate padding: a constant image gives exactly zero response, also at the border. | R2.1 |
| B3 | **Batch-level normalisation of the adaptive weight.** `min`/`max` were taken over the whole batch. | `(E − E.min()) / (E.max() − E.min())` on a `[B,C,H,W]` tensor. | The weight of an image depended on which other images shared its batch. | Min–max per image. | R2.2 |
| B4 | **Non-standard metrics.** PSNR/SSIM on float RGB, no 8-bit rounding, no border crop, scikit-image SSIM with its default 7×7 uniform window, averaged per batch. | `calculate_psnr`, `calculate_ssim`, `validate`. | Numbers not comparable with the SR literature (which reports Y-channel PSNR/SSIM with border crop = scale and Gaussian SSIM) and slightly mis-weighted (last smaller batch counted like a full one). | Output rounded to 8 bit; Y channel (BT.601) and RGB; crop = scale; Gaussian 11×11 SSIM; per-image values written to CSV. Cross-checked against scikit-image on Kaggle. | R2.4, R2.5 |
| B5 | **Resume re-warmed the learning rate.** `main_finetuning` restored a `CosineAnnealingLR` whose `T_max` was the old epoch count. | Restoring after 2 epochs and continuing gave LR per epoch 2e-4, 1e-4, 1e-7, **1e-4, 2e-4, 1e-4**†. | "Fine-tuning" silently restarted with a high LR; the schedule of a resumed run differed from an uninterrupted one. | Iteration-based cosine schedule computed from the step number, so resuming cannot change it. Resume is tested to give bit-identical weights. | R4.3 |
| B6 | **Split not reproducible, duplicates not handled.** `glob` order (filesystem dependent) was shuffled with seed 42; identical or near-identical faces could land in train and test. | Cell 3. | A re-run could get a different test set; duplicates across splits inflate test scores. | Sorted file list; exact duplicates dropped; near-duplicates (same 64-bit dHash) kept in one split; `manifest.csv` records everything. | R1.4 |
| B7 | **No seeds, one run per condition.** Only `random.seed(42)` for the split. | No `torch.manual_seed`; DataLoader shuffling unseeded. | Runs could not be repeated, and seed-to-seed noise, which can be as large as the loss effects under study, could not be separated from a method effect. | `seed` fixes initial weights, data order and augmentation; 3 confirmation seeds; paired statistics. | R4.4 |
| B8 | **Static vs adaptive at the same λ.** The notebook planned to compare both at `sobel_weight = 0.1`. | Adaptive weights are ≤ 1, so at equal λ the adaptive edge term is much smaller. | A difference could come from the effective loss scale, not from the weighting. | Both modes get the same λ sweep; each is tuned on validation; the comparison is best vs best. The share of the edge term in the objective is reported. | — |

## C. Minor: hygiene and clarity

| ID | Problem | Fix |
|---|---|---|
| C1 | `model_size = 'large'` was classical SwinIR with 4 instead of 6 RSTBs (8.0M params), while the README said the architecture was "unchanged". The code was unchanged, the configuration was not. | Named presets (`light`, `medium`, `classical`) with parameter counts checked by tests. |
| C2 | Dependency on `timm` through the deprecated `timm.models.layers` path. | Three small helpers re-implemented; the network code is otherwise byte-identical to the official file (R3.2 compares it on Kaggle). |
| C3 | Experiment name `"..._x2sobel : 0.1"` (spaces, colon); checkpoints written to the working directory as `model_epoch_{e}_{loss}.pth`, so two λ values overwrote each other. | One folder per run named `{model}_x{scale}_{condition}_s{seed}`. |
| C4 | Full checkpoint every epoch, non-atomic writes. | `last.pt` every 2,000 iterations via write-then-rename; `model_final.pt` (weights only) at the end; `last.pt` deleted after evaluation. |
| C5 | `AnimeConfig` defined twice with different `loss_type`; class attributes evaluated once, so editing `loss_type` later did not change `experiment_name`. | One `TrainConfig` dataclass saved as `config.json` in every run. |
| C6 | Duplicate losses (`CombinedLoss` ≈ `AdaptiveSobelLoss(adaptive=False)`); an edge-only `'sobel'` loss without a pixel term (Sobel ignores a constant colour shift, so colours would drift). | One `EdgeAwareLoss` with `edge_mode ∈ {none, static, adaptive}`; edge-only training dropped. |
| C7 | Personal metadata (name, university) written into every checkpoint. | Checkpoints hold config, step and environment only; authorship belongs in `CITATION.cff`. |
| C8 | No data augmentation. | Random flips/rotations (8 dihedral transforms), as in the official SwinIR training code. Recorded in `configs/base.json`. |

## D. Open questions about the data (to verify, not yet known)

1. **What are the source images?** The Kaggle dataset `mcparadip/anime-faces-waifu2x` (≈2.1 GB,
   ≈21k PNG files of ≈100 KB, released 2020-05-24) has no description. Its name, file count and
   file sizes suggest the well-known 64×64 "Anime Faces" set (21,551 images) upscaled with
   **waifu2x**, a CNN super-resolution model. If so, our 128×128 "ground truth" is a downscaled
   waifu2x output and the networks partly learn to imitate waifu2x. A loss comparison stays valid
   (every condition has the same targets), but the paper must say "waifu2x-upscaled anime faces",
   not "high-resolution anime faces". Gate check R1.6 records the source sizes; look at a few
   images before writing the data section.
2. **Licence.** Kaggle lists the licence as *Unknown*. Use it for research, cite the dataset page,
   do not redistribute the images or the prepared dataset, and keep sample figures small.
