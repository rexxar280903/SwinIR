# SwinIR + Adaptive Edge-Aware Loss for Anime Face Super-Resolution

Implementasi *image super-resolution* (SR) berbasis **SwinIR** (Swin Transformer for Image Restoration) yang dimodifikasi dengan **fungsi loss berbasis edge (Sobel) adaptif** untuk meningkatkan ketajaman garis/outline pada gambar anime, dievaluasi pada dataset *Anime Faces (Waifu2x)*.

> Arsitektur backbone mengacu pada paper:
> **Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCV Workshops, 2021.**
> [arXiv:2108.10257](https://arxiv.org/abs/2108.10257) · [Official Repo](https://github.com/JingyunLiang/SwinIR)

---

## 📖 Daftar Isi

- [Ringkasan](#-ringkasan)
- [Arsitektur Model (SwinIR)](#-arsitektur-model-swinir)
- [Kontribusi / Modifikasi pada Proyek Ini](#-kontribusi--modifikasi-pada-proyek-ini)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Loss Function](#-loss-function)
- [Evaluasi & Metrik](#-evaluasi--metrik)
- [Konfigurasi Eksperimen](#-konfigurasi-eksperimen)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Menjalankan](#-cara-menjalankan)
- [Checkpoint & Metadata](#-checkpoint--metadata)
- [Batasan & Catatan Teknis](#-batasan--catatan-teknis)
- [Rencana Pengembangan](#-rencana-pengembangan)
- [Referensi](#-referensi)

---

## 🧩 Ringkasan

Proyek ini mengimplementasikan ulang arsitektur **SwinIR** secara penuh (tanpa modifikasi struktural) sebagai backbone *super-resolution* (×2), kemudian menambahkan **komponen loss tambahan berbasis gradien (edge-aware)** untuk mendorong model menghasilkan tepi/garis (outline) yang lebih tajam — karakteristik penting pada citra anime yang umumnya memiliki *line art* tegas. Selain itu, ditambahkan metrik evaluasi khusus **Edge-PSNR** untuk mengukur kualitas rekonstruksi secara spesifik pada area tepi gambar, di luar metrik standar PSNR/SSIM.

---

## 🏗️ Arsitektur Model (SwinIR)

Backbone yang digunakan adalah implementasi **SwinIR original** (RSTB + STL) tanpa perubahan struktural, terdiri dari tiga modul utama:

```
Input (LQ) ──► Shallow Feature Extraction (Conv 3×3)
                        │
                        ▼
              Deep Feature Extraction
        ┌─────────────────────────────────┐
        │  RSTB → RSTB → ... → RSTB → Conv │   (K buah Residual Swin
        └─────────────────────────────────┘    Transformer Block)
                        │   (+ long skip connection dari shallow feature)
                        ▼
              HQ Image Reconstruction
                  (Sub-pixel Conv / PixelShuffle)
                        │
                        ▼
                  Output (SR Image)
```

**Residual Swin Transformer Block (RSTB):**
- Terdiri dari beberapa **Swin Transformer Layer (STL)** + 1 convolutional layer + residual connection.
- STL menggunakan **Window-based Multi-head Self-Attention (W-MSA)** dengan *relative position bias*, serta **shifted window** secara alternatif antar-layer untuk membangun koneksi antar-window (cross-window interaction).
- Setiap STL: `LayerNorm → MSA → residual → LayerNorm → MLP (GELU) → residual`.

**Konfigurasi model yang digunakan pada proyek ini** (`model_size = 'large'`):

| Parameter | Nilai |
|---|---|
| Upscale factor | ×2 |
| Window size | 8 |
| Embedding dim | 180 |
| Depths (jumlah STL per RSTB) | [6, 6, 6, 6] |
| Jumlah attention heads | [6, 6, 6, 6] |
| MLP ratio | 2 |
| Upsampler | `pixelshuffle` |
| Residual connection di RSTB | `1conv` |
| Input size (LR) | 64×64 → Output (HR) 128×128 |

Implementasi modul (`Mlp`, `WindowAttention`, `SwinTransformerBlock`, `RSTB`, `SwinIR`, dll.) diadaptasi langsung dari kode resmi SwinIR (kredit: Ze Liu, dimodifikasi oleh Jingyun Liang), **tanpa perubahan pada arsitektur intinya**.

---

## 🚀 Kontribusi / Modifikasi pada Proyek Ini

Yang membedakan proyek ini dari SwinIR baseline (paper asli hanya menggunakan **L1 pixel loss** untuk SR klasik) adalah pada **sisi loss function dan metrik evaluasi**, bukan pada arsitektur:

### 1. Sobel-based Edge Extraction Module
Modul `SobelOperator` yang mengekstrak *gradient magnitude* dari gambar menggunakan kernel Sobel horizontal & vertikal, diterapkan secara *depthwise* (per-channel) melalui `F.conv2d` dengan `groups=C`. Modul ini menjadi dasar dari seluruh varian loss berbasis edge di bawah.

### 2. Sobel Loss
Loss tambahan yang membandingkan *edge map* hasil prediksi terhadap *edge map* ground truth, alih-alih membandingkan piksel mentah:

```
L_sobel = w · ‖Sobel(I_pred) − Sobel(I_target)‖₁
```

### 3. Combined Loss (Pixel + Sobel, weighted statis)
Kombinasi pixel loss (dapat dipilih L1 / Charbonnier / MSE) dengan Sobel loss berbobot tetap:

```
L_combined = w_pixel · L_pixel + L_sobel
```

### 4. ⭐ Adaptive Sobel Loss (loss utama yang digunakan)
**Ini adalah kontribusi utama proyek.** Berbeda dari Sobel loss biasa, bobot kontribusi tiap piksel terhadap loss edge **disesuaikan secara dinamis berdasarkan kekuatan edge ground truth itu sendiri** (min-max normalized), sehingga area dengan tepi/garis tajam diberi penalti lebih besar dibanding area datar:

```
edge_weight = (E_target − min(E_target)) / (max(E_target) − min(E_target) + ε)
L_edge      = mean( edge_weight · |E_pred − E_target| )
L_total     = w_pixel · L_pixel(L1) + w_sobel · L_edge
```

dengan `E(·) = Sobel(·)`.

**Motivasi:** pada gambar anime, garis outline (edge tegas) jauh lebih penting secara perseptual dibanding tekstur halus pada area datar (kulit, background polos). Bobot adaptif ini mendorong model untuk memprioritaskan rekonstruksi garis tanpa mengorbankan stabilitas training pada area non-edge.

### 5. Edge-PSNR — Metrik Evaluasi Tambahan
Selain PSNR dan SSIM standar, ditambahkan metrik **Edge-PSNR**: PSNR yang dihitung **hanya pada piksel yang teridentifikasi sebagai tepi** (berdasarkan *edge mask* dari gambar HR ground truth, dengan threshold magnitude Sobel tertentu). Metrik ini memberikan gambaran lebih spesifik soal seberapa baik model merekonstruksi detail garis, yang tidak ditangkap secara memadai oleh PSNR global.

```
mask        = 1[ Sobel_magnitude(I_HR) > threshold ]
Edge-MSE    = Σ( mask · (I_SR − I_HR)² ) / Σ(mask)
Edge-PSNR   = 10 · log10( 1 / Edge-MSE )
```

| Aspek | SwinIR (Paper Asli) | Proyek Ini |
|---|---|---|
| Arsitektur | RSTB + STL | **Sama (tidak diubah)** |
| Loss (SR klasik) | L1 pixel loss saja | L1 pixel loss + **Adaptive Sobel Edge Loss** |
| Metrik evaluasi | PSNR, SSIM | PSNR, SSIM, **Edge-PSNR** |
| Domain data | Natural images (DIV2K, dll.) | **Anime faces** (Waifu2x dataset) |
| Skala | ×2 / ×3 / ×4 | ×2 |

---

## 📂 Dataset & Preprocessing

- **Sumber:** [Anime Faces (Waifu2x)](https://www.kaggle.com/datasets) — dataset wajah karakter anime.
- **Preprocessing:**
  - Gambar HR diresize ke **128×128** (bicubic interpolation).
  - Gambar LR digenerate dari HR dengan downsampling ke **64×64** (bicubic), sehingga scale factor = ×2.
  - Split data: **90% train / 10% test** (`train_test_split`, `random_state=42`).
  - Disimpan dalam struktur folder `train/{HR,LR}` dan `test/{HR,LR}` sebagai pasangan gambar dengan nama file identik.
- **Dataset class:** `SRDataset` (pasangan LR–HR dari folder terpisah) dan `SingleFolderDataset` (LR digenerate on-the-fly dari folder HR tunggal) — keduanya mengembalikan tensor ternormalisasi ke rentang `[0, 1]`.

> ⚠️ Catatan: pada konfigurasi eksperimen yang disertakan, terdapat parameter `LIMIT_DATA` yang membatasi jumlah data yang diproses (berguna untuk *quick testing*/debugging pipeline). Pastikan nilai ini disesuaikan (atau di-set `None`) sebelum training skala penuh.

---

## 🎯 Loss Function

Loss dipilih melalui parameter `config.loss_type`, dengan 4 opsi: `'l1'`, `'sobel'`, `'combined'`, `'adaptive'`.

**Konfigurasi yang digunakan pada eksperimen final:**
```python
loss_type     = 'adaptive'   # AdaptiveSobelLoss
pixel_weight  = 1.0
sobel_weight  = 0.1          # direkomendasikan 0.1–0.5 untuk domain anime
```

Pixel loss dasar menggunakan **L1**, dipilih agar konsisten dengan konvensi training SR pada paper SwinIR, sebelum ditambah komponen edge adaptif.

---

## 📊 Evaluasi & Metrik

| Metrik | Deskripsi |
|---|---|
| **PSNR** | Dihitung global per-gambar menggunakan `skimage.metrics.peak_signal_noise_ratio`, dirata-rata per batch. |
| **SSIM** | Dihitung global menggunakan `skimage.metrics.structural_similarity`. |
| **Edge-PSNR** | PSNR yang dibatasi hanya pada region edge (lihat [bagian novelty](#5-edge-psnr--metrik-evaluasi-tambahan)). |

Validasi dijalankan tiap akhir epoch (`validate()`), dengan output SR di-*clamp* ke `[0, 1]` sebelum dihitung metriknya.

---

## ⚙️ Konfigurasi Eksperimen

Dikelola melalui class `AnimeConfig`:

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

Tersedia juga fungsi `main_finetuning()` untuk **melanjutkan training dari checkpoint** (resume) maupun **fine-tuning** dengan learning rate baru, lengkap dengan restore state optimizer & scheduler.

---

## 🗂️ Struktur Proyek

> *(disesuaikan dengan struktur sel pada notebook — sesuaikan kembali jika kode dipecah ke file `.py` terpisah)*

```
.
├── swinir-l1.ipynb          # Notebook utama (model, training, evaluasi)
├── dataset_split/            # Hasil preprocessing (train/test, HR/LR)
├── experiments_anime/        # Output eksperimen & log
└── model_epoch_*_*.pth       # Checkpoint hasil training
```

Modul-modul kunci dalam notebook:
- **Model:** `SwinIR`, `RSTB`, `SwinTransformerBlock`, `WindowAttention`, `Mlp`
- **Loss:** `SobelOperator`, `SobelLoss`, `CombinedLoss`, `AdaptiveSobelLoss`
- **Data:** `SRDataset`, `SingleFolderDataset`, `build_dataloaders`
- **Utilitas:** `AverageMeter`, `calculate_psnr`, `calculate_ssim`, `calculate_edge_psnr`, `get_edge_mask`
- **Training:** `build_model`, `build_loss`, `train_epoch`, `validate`, `main`, `main_finetuning`

---

## ▶️ Cara Menjalankan

```bash
pip install timm
```

1. **Preprocessing dataset** — jalankan sel *Resize & Split Dataset* untuk membentuk pasangan LR–HR dari dataset mentah.
2. **Training dari awal:**
   ```python
   main()
   ```
3. **Resume / fine-tuning dari checkpoint:**
   ```python
   main_finetuning(
       checkpoint_path="model_epoch_2_adaptive.pth",
       load_optimizer=True,
       new_lr=None,          # None = lanjutkan LR dari checkpoint
       additional_epochs=2,
       new_batch_size=8
   )
   ```
4. **Cek metadata checkpoint** tanpa load model penuh:
   ```python
   read_checkpoint_metadata("model_epoch_2_adaptive.pth")
   ```

---

## 💾 Checkpoint & Metadata

Setiap checkpoint (`.pth`) menyimpan, selain `model_state_dict`/`optimizer_state_dict`/`scheduler_state_dict`:
- Hyperparameter lengkap (scale, window size, loss type, pixel/sobel weight, dll.)
- Metrik training & validasi per epoch (`train_loss`, `pixel_loss`, `edge_loss`, `val_psnr`, `val_ssim`, `val_edge_psnr`)
- Metadata eksperimen (waktu, device, durasi epoch, institusi)

---

## ⚠️ Batasan & Catatan Teknis

- **Skala eksperimen:** konfigurasi contoh menggunakan `num_epochs = 2` dan `LIMIT_DATA` terbatas — ini cocok untuk *pipeline sanity check*, namun hasil kuantitatif penuh memerlukan training pada keseluruhan dataset dengan jumlah epoch yang representatif.
- **Kernel Sobel vertikal:** mohon diperiksa kembali baris terakhir kernel `sobel_y` pada `SobelOperator` — kernel Sobel vertikal standar adalah `[[-1,-2,-1],[0,0,0],[1,2,1]]`, namun pada implementasi saat ini baris terakhir tertulis `[1,2,3]`. Ini sebaiknya dikoreksi sebelum dilaporkan sebagai hasil final, karena akan sedikit menggeser arah gradien yang terdeteksi.
- **Arsitektur tidak dimodifikasi:** seluruh peningkatan performa (jika ada) berasal dari sisi loss/training objective, bukan dari perubahan struktural SwinIR. Hal ini perlu dinyatakan secara eksplisit di bagian metodologi agar klaim kontribusi akurat (kontribusi pada *training objective* dan *evaluation metric*, bukan pada *network architecture*).

---

## 🔭 Rencana Pengembangan

- Eksperimen ablasi sistematis antar varian loss (`l1` vs `sobel` vs `combined` vs `adaptive`) dengan jumlah epoch yang setara untuk perbandingan yang adil.
- Validasi pada scale factor lain (×3, ×4).
- Studi sensitivitas terhadap `sobel_weight` dan `edge_threshold` pada Edge-PSNR.
- Perbaikan kernel Sobel vertikal pada `SobelOperator`.

---

## 📚 Referensi

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

## 📝 Lisensi & Atribusi

Implementasi arsitektur SwinIR diadaptasi dari [repositori resmi](https://github.com/JingyunLiang/SwinIR) (Ze Liu, dimodifikasi oleh Jingyun Liang). Mohon cantumkan atribusi yang sesuai jika kode ini digunakan ulang, serta sitasi pada paper asli SwinIR di atas.
