# Panduan lengkap proyek Edge-Aware SwinIR (Bahasa Indonesia)

**Dibuat**: 2026-10-05 · **Status**: kode, protokol, dan readiness gate siap; eksperimen penuh belum dijalankan.

Dokumen ini menjelaskan semuanya dari nol: konsep, isi kode, kesalahan di notebook lama, alasan
desain eksperimen, langkah menjalankan di Kaggle besok, cara membaca hasil, dan jalur publikasi.
Dokumen teknis berbahasa Inggris untuk pembaca repo: [METHOD.md](METHOD.md) (definisi),
[EXPERIMENTS.md](EXPERIMENTS.md) (protokol), [AUDIT.md](AUDIT.md) (temuan audit).

> **Kalau waktumu cuma 10 menit:** baca §1 (apa yang berubah), lalu §7 (runbook besok).

---

## 1. Ringkasan: apa yang dikerjakan

| Sebelum | Sesudah |
|---|---|
| Satu notebook 2.200 baris, sel saling bergantung | Paket Python `edgesr/` (9 modul) + 8 skrip CLI + notebook runner Kaggle |
| Kernel Sobel salah ketik, Edge-PSNR bias −4,77 dB, baseline L1 crash di GPU | Semua diperbaiki, masing-masing dijaga oleh tes otomatis |
| Split train/test saja, 10 gambar, 2 epoch | Train/val/test 80/10/10, deduplikasi, manifest, data penuh |
| Satu run per kondisi, tanpa seed | Sweep λ (seed 0) + konfirmasi 3 seed baru, statistik berpasangan |
| Tidak ada cara tahu apakah siap | **Readiness gate**: 28 cek (R0–R7) yang harus lulus sebelum ablation |
| README mengklaim hasil yang belum ada | README jujur: "results pending", protokol pra-registrasi |

Hasil uji lokal (CPU, data sintetis, 2026-10-05): **20/20 unit test lulus**; readiness gate mode
`--quick` berstatus **READY WITH WARNINGS** (peringatannya hanya "tidak ada GPU" dan "kunci protokol
belum di-commit", dua hal yang memang baru bisa diselesaikan di Kaggle). Dry-run seluruh pipeline
(sweep → bicubic → perluasan grid → konfirmasi → analisis) berjalan sampai `RESULTS.md` terbentuk.

Yang **belum** dilakukan: menjalankan apa pun di GPU dengan data asli. Belum ada angka hasil.

---

## 2. Konsep dasar (intuisi dulu, rumus kemudian)

### 2.1 Super-resolution (SR)

SR = menebak gambar resolusi tinggi (HR) dari gambar resolusi rendah (LR). Di proyek ini:
HR 128×128 → diperkecil dengan **bicubic** menjadi LR 64×64 → jaringan harus mengembalikan
128×128. Karena kita sendiri yang membuat LR dari HR, kita punya "kunci jawaban" untuk setiap
gambar. Setting ini disebut *bicubic SR* atau *classical SR*: degradasinya diketahui dan bersih
(tanpa noise/JPEG), sehingga perbedaan hasil hanya berasal dari model/loss, bukan dari degradasi
yang tidak diketahui.

### 2.2 SwinIR dalam lima menit

SwinIR adalah transformer untuk restorasi gambar. Alurnya:

1. **Konvolusi 3×3** mengubah RGB menjadi 60 (light) atau 180 (medium) fitur per piksel.
2. **RSTB** (Residual Swin Transformer Block) ×4. Tiap RSTB berisi 6 **STL** (Swin Transformer
   Layer). Tiap STL melakukan *self-attention* hanya di dalam jendela 8×8 piksel (W-MSA), supaya
   biayanya tidak meledak. Layer berikutnya menggeser jendela 4 piksel (SW-MSA) sehingga informasi
   bisa menyeberang batas jendela.
3. **Skip connection panjang**: fitur awal ditambahkan kembali, jadi jaringan cukup mempelajari
   "detail yang hilang".
4. **Pixel shuffle** menyusun ulang kanal menjadi gambar 2× lebih besar.

Kita **tidak mengubah arsitektur sama sekali**. Kode jaringannya adalah file resmi SwinIR; cek
R3.2 mengunduh file resmi dan membandingkannya baris per baris. Semua perbedaan antar-kondisi
ada di loss. Itu penting untuk klaim: kontribusi proyek ini ada di *fungsi objektif*, bukan
arsitektur.

**Preset model:**

| Preset | Parameter | Keterangan |
|---|---|---|
| `light` | 0,91 juta | SwinIR-light dari paper. **Dipakai untuk ablation** karena 20+ run harus muat kuota GPU gratis |
| `medium` | 8,02 juta | Konfigurasi notebook lama (dulu disebut "large", padahal hanya 4 RSTB, bukan 6 seperti paper) |
| `classical` | 11,75 juta | SwinIR classical sesuai paper; terlalu berat untuk banyak run di T4 |

Paper SwinIR menulis 878K untuk SwinIR-light ×2. Selisihnya dengan 910.152 adalah 24 tabel *relative
position bias* (24 × 1.350) yang tidak dihitung paper. Jadi angkanya konsisten.

### 2.3 Operator Sobel: alat ukur "ketajaman tepi"

Sobel adalah filter 3×3 yang menghitung seberapa cepat kecerahan berubah:

```
Kx = [-1 0 1]      Ky = Kx transpos = [-1 -2 -1]
     [-2 0 2]                         [ 0  0  0]
     [-1 0 1]                         [ 1  2  1]
```

Magnitudo gradien S = √(gx² + gy²). Contoh: di area datar S = 0; di garis tepi dengan beda
kecerahan h, S = 4h. Jadi S besar = ada garis/kontur, S kecil = area datar.

Dua hal yang dulu salah dan sekarang benar:
- Baris terakhir `Ky` harus `[1 2 1]`. Notebook lama menulis `[1 2 3]`, sehingga jumlah kernel = 2 dan
  area datar berkecerahan 0,7 dianggap punya "tepi" sebesar 1,4. Filter itu jadi mengukur
  kecerahan, bukan tepi.
- Di pinggir gambar, filter perlu piksel di luar gambar. Notebook mengisi dengan **nol** (hitam),
  sehingga setiap pinggiran gambar terang tampak seperti garis tebal. Sekarang dipakai **replicate
  padding** (piksel pinggir diulang), sehingga gambar datar menghasilkan nol di mana pun.

### 2.4 Tiga loss yang dibandingkan

- **L1**: rata-rata |SR − HR| per piksel. Loss standar SwinIR. Cenderung menghasilkan gambar
  sedikit "lembut", karena kesalahan kecil di banyak piksel datar dihitung sama dengan kesalahan
  di garis.
- **Static Sobel**: L1 + λ · rata-rata |S(SR) − S(HR)|. Menghukum perbedaan *peta tepi*. Kalau SR
  kabur, garisnya lebih lebar dan lebih lemah, sehingga S(SR) ≠ S(HR) dan loss naik.
- **Adaptive Sobel** (kontribusi utama): sama dengan static, tetapi tiap piksel dikalikan bobot
  `w = (S(HR) − min) / (max − min)` yang dihitung **per gambar**. w = 1 di garis terkuat dan
  w ≈ 0 di area datar. Artinya: "kesalahan tepi hanya penting di tempat yang memang ada garisnya."

**Kenapa ini menarik untuk anime?** Gambar anime sebagian besar adalah warna datar yang dibatasi
garis tegas (line art). Static Sobel membuang sebagian besar "perhatiannya" ke area datar yang
gradiennya nol. Adaptive memusatkannya ke garis, dan blur paling terlihat di garis.

**Jebakan yang harus dipahami: kekuatan efektif.** Karena w ≤ 1 dan kebanyakan piksel datar
(w ≈ 0), suku edge adaptive jauh lebih kecil daripada static pada λ yang sama. Membandingkan
keduanya pada λ = 0,1 (rencana notebook lama) sama saja membandingkan "loss tepi kuat" dengan
"loss tepi lemah". Solusinya: λ di-sweep untuk keduanya, masing-masing dipilih λ terbaiknya
di data validasi, lalu yang dibandingkan adalah *terbaik vs terbaik*. Analisis juga melaporkan
**edge share**, yaitu berapa persen dari total loss yang benar-benar berasal dari suku tepi.

### 2.5 Metrik

Semua metrik dihitung **per gambar** lalu dirata-rata. Sebelum dihitung, SR dibulatkan ke 8-bit
(seperti disimpan ke PNG) dan 2 piksel pinggir dibuang (konvensi paper SR untuk ×2).

| Metrik | Arti sederhana | Kenapa dipakai |
|---|---|---|
| **PSNR-Y** | Seberapa kecil kesalahan piksel, di kanal kecerahan (Y). Satuan dB, makin tinggi makin baik. +0,1 dB sudah dianggap berarti di SR | Konvensi paper SR. Mata manusia lebih peka ke kecerahan daripada warna, jadi paper SR melaporkan Y |
| **SSIM-Y** | Kemiripan struktur lokal (rata-rata, kontras, korelasi), 0–1 | Standar, dengan jendela Gaussian 11×11 seperti kode SwinIR |
| **Edge-PSNR-Y** | PSNR yang hanya dihitung di piksel tepi (S(HR) > τ = 0,5) | **Endpoint utama**: langsung mengukur kualitas garis |
| **GMSD** | Seberapa tidak konsisten kemiripan gradien di seluruh gambar; makin kecil makin baik | Metrik mapan (Xue dkk., 2014) yang sensitif terhadap tepi; menangkal tuduhan "metrik buatan sendiri" |
| Edge-PSNR τ=0,25 & 1,0 | Edge-PSNR dengan ambang lain | Uji sensitivitas: apakah kesimpulan berubah kalau τ diubah |
| LPIPS (opsional) | Jarak perseptual dari fitur jaringan saraf | Untuk klaim "terlihat lebih tajam", kalau paketnya terpasang |

**Tentang τ = 0,5**: untuk loncatan kecerahan h tingkat (0–255), magnitudo Sobel = 4h/255. Jadi
τ = 0,5 ≈ loncatan 32 tingkat: garis dan kontur kuat terpilih, gradasi bayangan dan noise tidak.
Cek R1.7 melaporkan persentase piksel tepi untuk beberapa τ di data asli. **τ dikunci sebelum
melihat hasil training.** Kalau dikunci sesudah melihat hasil, orang bisa (sengaja atau tidak)
memilih τ yang paling menguntungkan.

---

## 3. Temuan audit, dengan bahasa sederhana

Detail lengkap dan bukti ada di [AUDIT.md](AUDIT.md). Angka bertanda † dari
`scripts/audit_legacy.py` pada 64 gambar sintetis (ilustrasi; jalankan ulang di data asli untuk paper).

**Kritis (hasil akan salah atau tidak ada):**

1. **Kernel Sobel salah ketik (A1).** Akibatnya bobot "adaptive" di notebook lama berkorelasi
   dengan kecerahan (r = 0,50†) hampir sama kuat dengan tepi (r = 0,58†). Piksel datar mendapat bobot
   rata-rata 0,25†, piksel tepi 0,42†: hanya 1,7× lebih besar. Setelah diperbaiki: 0,0003† vs 0,56†.
   Jadi loss di notebook lama **bukan loss tepi**. Semua kesimpulan dari loss itu tidak berlaku
   untuk metode yang ingin diteliti.
2. **Baseline L1 crash di GPU (A2).** Satu baris memakai `hr_imgs` (masih di CPU) alih-alih
   `hr_images` (di GPU). Baseline yang menjadi pembanding semua klaim tidak pernah bisa dilatih.
3. **10 gambar, 2 epoch (A3).** `LIMIT_DATA = 10`. Semua angka dari notebook adalah uji jalan
   (smoke test), bukan hasil.
4. **Tanpa split validasi (A4).** Memilih λ atau checkpoint berdasarkan data test membuat angka
   test terlalu optimistis. Reviewer yang teliti akan menolak ini.

**Mayor (angka bias atau perbandingan tercampur):**

5. **Edge-PSNR bias −4,77 dB (B1).** Error dijumlahkan di 3 kanal RGB tetapi dibagi jumlah piksel
   saja, sehingga MSE 3× terlalu besar. Biasnya selalu tepat −10·log₁₀(3) = −4,77 dB. Peringkat
   antar metode tidak berubah, tetapi angkanya tidak bisa dibandingkan dengan apa pun.
6. **Padding nol (B2).** 100%† piksel pinggir dianggap tepi; 19%† isi mask Edge-PSNR adalah pinggiran.
   Loss adaptive memberi bobot tertinggi ke pinggiran gambar.
7. **Normalisasi per batch (B3).** Bobot sebuah gambar bergantung pada gambar lain di batch yang sama.
8. **Metrik non-standar (B4).** RGB float tanpa pembulatan, tanpa crop, SSIM jendela 7×7 seragam.
   Tidak bisa dibandingkan dengan paper SR.
9. **Resume menaikkan LR lagi (B5).** Scheduler yang dipulihkan membuat LR per epoch menjadi
   2e-4 → 1e-4 → 1e-7 → **1e-4 → 2e-4**†. "Fine-tuning" ternyata mulai ulang dengan LR tinggi.
10. **Split tidak reproducible dan duplikat tidak ditangani (B6), tanpa seed (B7), static vs
    adaptive pada λ sama (B8).**

**Pertanyaan terbuka tentang data (§D):** dataset Kaggle-nya tidak punya deskripsi, lisensinya
"Unknown", dan namanya menyiratkan gambar sudah di-upscale dengan **waifu2x** (sebuah model SR!).
Kalau benar, "HR" kita sebagian berisi detail buatan waifu2x. Perbandingan antar-loss tetap sah
(target semua kondisi sama), tetapi di paper harus ditulis "waifu2x-upscaled anime faces". Cek
R1.6 mencatat ukuran sumber; **buka beberapa gambar sumber dengan mata sendiri** sebelum menulis
bagian data.

---

## 4. Peta kode

```
                   scripts/prepare_data.py
sumber Kaggle ───────────────────────────────► data/{train,val,test}/{HR,LR}/*.png
                   (edgesr/data.py)              manifest.csv, dataset_info.json
                                                        │
scripts/readiness_check.py  ◄───────────────────────────┤  (edgesr/readiness.py: R0–R7)
                                                        │
scripts/run_ablation.py ──► edgesr/ablation.py (rencana run, pilih λ dari val, perluasan grid)
        │                         │
        └──► edgesr/engine.py: train() ──► model (models/), loss (losses.py), data (data.py)
                    │                 └──► runs/<nama_run>/{config.json, train_log.csv,
                    │                                      val_log.csv, last.pt, model_final.pt}
                    └──► evaluate_run() ──► metrics.py ──► test_per_image.csv, test_summary.json
                                                        │
scripts/analyze.py ──► edgesr/stats.py ──► runs/analysis/RESULTS.md + CSV + PNG
```

| File | Isi | Kapan disentuh |
|---|---|---|
| `edgesr/models/swinir.py` | Jaringan SwinIR resmi | **Jangan diubah** (R3.2 akan gagal) |
| `edgesr/models/__init__.py` | Preset light/medium/classical, `build_model` | Kalau ingin preset baru |
| `edgesr/losses.py` | `sobel_gradients`, `sobel_magnitude`, `adaptive_edge_weight`, `EdgeAwareLoss` | Inti metode |
| `edgesr/metrics.py` | PSNR, SSIM, Y-channel, Edge-PSNR, GMSD, LPIPS opsional | Ubah = protokol berubah |
| `edgesr/data.py` | Persiapan data, dedup dHash, `SRPairDataset`, `TrainSampler` (urutan data deterministik & bisa di-resume) | Jarang |
| `edgesr/config.py` | `TrainConfig`: semua hyperparameter dalam satu tempat; `protocol_hash()` | Lewat `configs/base.json` saja |
| `edgesr/engine.py` | Loop training per iterasi, AMP, checkpoint atomik, resume, evaluasi, bicubic | Jarang |
| `edgesr/ablation.py` | Grid λ, seed, aturan seleksi, aturan perluasan grid | **Terkunci** setelah lock |
| `edgesr/stats.py` | Bootstrap dua tingkat, Wilcoxon, Holm | Jarang |
| `edgesr/readiness.py` | Seluruh cek readiness | Kalau menambah cek |
| `configs/base.json` | Nilai protokol (model, iterasi, batch, τ, dst.) | Sebelum lock saja |

**Aturan emas:** semua yang memengaruhi hasil (model, iterasi, batch, lr, τ, augmentasi) ada di
`configs/base.json` dan masuk ke `protocol_hash`. Mengubahnya setelah lock = protokol baru, dan
`analyze.py` menolak mencampur run dari protokol berbeda.

---

## 5. Desain eksperimen dan alasannya

### 5.1 Kenapa dua tahap?

- **Tahap 1, sweep (seed 0, 11 run):** L1, lalu static dan adaptive masing-masing dengan
  λ ∈ {0,05; 0,1; 0,2; 0,5; 1,0}. Tujuannya mencari λ terbaik tiap mode, **hanya dari data validasi**.
- **Tahap 1b, perluasan (0–4 run, bersyarat):** kalau λ terbaik ada di ujung grid (misalnya 1,0 =
  terbesar), mungkin optimum sebenarnya di luar grid. Maka ditambah λ ∈ {2, 5} (atau {0,02; 0,01}
  untuk ujung bawah), **satu kali saja**. Aturan ini ditulis sebelum eksperimen, jadi bukan
  "mengutak-atik sampai bagus".
- **Tahap 2, konfirmasi (seed 1, 2, 3; 9 run):** L1, static(λ*), adaptive(λ*). Seed 0 *tidak*
  dipakai lagi, karena λ yang menang di seed 0 bisa saja menang karena kebetulan seed itu
  (*winner's curse*). Seed baru memberi penilaian yang adil.

### 5.2 Aturan memilih λ (dari validasi)

1. Ambil rata-rata 3 titik validasi terakhir (lebih stabil daripada satu titik).
2. λ *eligible* jika PSNR-Y-nya paling banyak 0,10 dB di bawah L1. Kita tidak mau "tepi lebih tajam
   tapi gambar keseluruhan rusak".
3. Dari yang eligible, pilih Edge-PSNR-Y tertinggi; selisih < 0,01 dB dianggap seri, dan yang menang
   λ lebih kecil (lebih konservatif).

### 5.3 Hipotesis dan cara memutuskan

- **H1**: adaptive > L1 pada Edge-PSNR-Y.
- **H2**: adaptive > static pada Edge-PSNR-Y.
- **H3**: adaptive tidak lebih buruk dari L1 pada PSNR-Y lebih dari 0,05 dB (*non-inferiority*).

Sebuah hipotesis "didukung" hanya jika **ketiga** syarat terpenuhi: (a) interval kepercayaan 95%
di atas nol, (b) p-value Wilcoxon setelah koreksi Holm < 0,05, (c) selisihnya positif di **setiap**
seed. Syarat (c) sengaja ketat: dengan hanya 3 seed, satu seed yang ekstrem bisa menarik rata-rata.

### 5.4 Statistik, dijelaskan dengan intuisi

Ada dua sumber ketidakpastian: (1) **seed**: training ulang dengan seed lain memberi model yang
sedikit berbeda; (2) **gambar test**: set test lain akan memberi angka lain. Bootstrap dua tingkat
mensimulasikan keduanya: 10.000 kali, ambil acak 3 seed (boleh berulang) dan ambil acak gambar
test (boleh berulang), lalu hitung selisih rata-ratanya. Persentil 2,5% dan 97,5% dari 10.000
selisih itu adalah interval kepercayaan 95%.

Kenapa "berpasangan"? Run L1 seed 1 dan run adaptive seed 1 punya bobot awal, urutan data, dan
augmentasi yang **sama persis** (dijamin oleh `TrainSampler` dan cek R4.4). Jadi selisih keduanya
hanya karena loss. Membandingkan per gambar dan per seed jauh lebih sensitif daripada
membandingkan dua rata-rata begitu saja.

**Holm** mengoreksi fakta bahwa kita menguji dua hipotesis (H1, H2): makin banyak uji, makin besar
peluang "signifikan" karena kebetulan.

### 5.5 Apa itu pra-registrasi dan kenapa penting

Pra-registrasi = menulis hipotesis, metrik, aturan pemilihan, dan aturan keputusan **sebelum**
melihat hasil, lalu menguncinya (di sini: `configs/protocol_lock.json` yang di-commit ke git,
dengan hash). Manfaatnya: reviewer tahu kamu tidak memilih metrik/τ/λ yang paling menguntungkan
setelah melihat angka. Untuk aplikasi beasiswa riset, ini menunjukkan kedewasaan metodologis.
Kalau setelah lock ternyata harus mengubah sesuatu, itu boleh, asal dicatat di tabel *deviation
log* (EXPERIMENTS.md §9) beserta alasannya.

---

## 6. Readiness gate: cara membaca dan apa yang dilakukan kalau gagal

Status: `PASS` (lulus), `WARN` (bisa jalan, tapi harus diputuskan/diperiksa), `FAIL` (wajib
diperbaiki), `SKIP` (tidak berlaku di mesin ini). **Ablation tidak boleh dimulai selama ada FAIL.**

| Cek | Kalau FAIL/WARN, lakukan |
|---|---|
| R0.2 GPU | Kaggle: panel kanan → Accelerator → GPU T4 x2 |
| R0.4 Disk | Hapus folder lama di `/kaggle/working`; `last.pt` otomatis dihapus setelah run selesai |
| R1.1 Manifest / PILOT ONLY | Bangun ulang data tanpa `--limit` |
| R1.3 Degradasi | Jangan ubah file data secara manual; bangun ulang dengan `prepare_data.py` |
| R1.4 Kebocoran | Bangun ulang data (seharusnya tidak terjadi; berarti ada bug, laporkan) |
| R1.6 Sumber < 128 px | Gambar sumber terlalu kecil: HR-nya jadi hasil upscale. Pertimbangkan HR 64 / skala lain, atau buang gambar kecil |
| R1.7 Cakupan tepi | Kalau tepi < 2% atau > 40% piksel, pertimbangkan τ lain **sebelum lock**, dan catat alasannya |
| R2.x | Bug di kode loss/metrik. Jangan lanjut |
| R2.5 vs scikit-image | Implementasi PSNR/SSIM menyimpang dari referensi. Jangan lanjut |
| R3.2 Model ≠ resmi | File `swinir.py` terubah. Kembalikan dari git |
| R4.1 Overfit | Loss tidak turun di 8 gambar → ada yang rusak (lr, data, loss) |
| R4.3 Resume | Resume tidak identik → jangan pakai `--hours`/shard sampai diperbaiki |
| R4.6 AMP | Loss fp16 menyimpang/NaN → set `amp=false` di `configs/base.json` (lebih lambat ±2×) |
| R5.1 Dry run | Ada skrip yang error; baca pesan di laporan |
| R6.1 Budget | Pakai `total_iters` yang disarankan laporan, atau tambah minggu, atau pakai 2 GPU |
| R7.1 Lock | Unduh `protocol_lock.json`, salin ke `configs/`, commit & push |

---

## 7. Runbook besok di Kaggle

### 7.0 Malam ini / besok pagi: kode harus bisa diakses Kaggle

Semua perubahan ada di folder lokal `C:\Users\rexxa\Videos\LLL-Wiki\raw\experiments\SwinIR` dan
**belum di-commit atau di-push**. Pilih satu:

**A. Push ke GitHub (disarankan):**

```bash
cd /c/Users/rexxa/Videos/LLL-Wiki/raw/experiments/SwinIR
git checkout -b refactor-edgesr
git add -A
git commit -m "Refactor into edgesr package, fix notebook bugs, add readiness gate and protocol"
git push -u origin refactor-edgesr
```

Lalu merge ke `main` lewat Pull Request di GitHub (notebook runner meng-clone `main`), atau ubah
`BRANCH = "refactor-edgesr"` di sel 1 notebook.

**B. Upload sebagai dataset Kaggle:** zip folder repo (tanpa `.git`), Kaggle → Datasets → New
Dataset, lalu isi `CODE_DATASET = "/kaggle/input/<nama>"` di sel 1.

### 7.1 Siapkan notebook

1. Kaggle → Create → Notebook → menu **File → Import Notebook** → unggah
   `notebooks/kaggle_runner.ipynb`.
2. Panel kanan: **Accelerator: GPU T4 x2**, **Internet: On**, **Add Input** → cari
   `anime-faces-waifu2x` (pemilik `mcparadip`).
3. Pastikan akun Kaggle sudah verifikasi nomor HP (syarat GPU).

### 7.2 Hari pertama: data + gate (interaktif, ±20 menit)

1. Jalankan sel 1–3. Sel 2 mencetak folder sumber dan jumlah gambar (harusnya ±21 ribu). Sel 3
   membangun data dan mencetak `dataset_info.json`: **periksa** `counts`, `n_exact_duplicates_dropped`,
   `source_sizes_top10`.
2. Sel 4 (opsional): angka audit di data asli → simpan untuk paper.
3. Sel 5: **readiness gate**. Baca laporannya (tampil sebagai tabel).
4. Kalau R6.1 bilang *over budget*: ubah `"total_iters"` di `configs/base.json` ke angka yang
   disarankan, commit & push, lalu ulangi dari sel 1 (hapus `/kaggle/working/SwinIR` dulu atau
   `git pull` di dalamnya).
5. Kalau semua PASS kecuali R7.1: unduh `readiness/protocol_lock.json` dari tab Output, simpan ke
   `configs/protocol_lock.json` di laptop, commit & push. Dengan ini protokol **terkunci**. Jalankan
   gate sekali lagi: R7.1 harus PASS ("matches committed lock").

### 7.3 Menjalankan ablation (latar belakang)

1. Klik **Save Version → Save & Run All (Commit)**. Mode ini berjalan di server walau browser
   ditutup, maksimal ±12 jam per sesi. Setiap tahap berhenti sendiri sebelum batas (`--hours 11.5`).
2. Dengan T4 x2, sel 6 otomatis membagi run ke 2 GPU (`--shard 1/2` dan `2/2`), jadi waktu tunggu
   kira-kira setengahnya. Periksa di dokumentasi Kaggle bagaimana kuota dihitung untuk T4 x2.
3. Kalau sesi habis sebelum semua run selesai: buka versi notebook yang selesai → Output → jadikan
   dataset (atau tambahkan versi itu sebagai input), isi `IMPORT = ["/kaggle/input/<nama>/runs"]`
   di sel 2, lalu jalankan lagi. Run yang sudah selesai disalin dan dilewati; run yang terputus
   **dilanjutkan dari checkpoint terakhir**, hasilnya identik.

### 7.4 Setelah semua selesai

- Sel 9 membuat `runs/analysis/RESULTS.md` + gambar. Sel 10 membuat `results.zip` untuk diunduh.
- Salin `RESULTS.md` dan gambar ke repo (misalnya `results/`), tempel tabelnya di README bagian
  *Results*, commit.

### 7.5 Kuota GPU dan proyek lain

Rencana TinyLLM butuh GPU mulai **November** (pilot M5, eksperimen M7–M10, ±25 jam/minggu).
Minggu-minggu Oktober ini GPU masih bebas, jadi jalankan ablation SwinIR **sekarang** agar tidak
berebut kuota. Angka kuota mingguan Kaggle berubah-ubah; cek halaman akunmu.

### 7.5b Dua akun (dua peneliti, masing-masing GPU T4 x2)

Setiap akun milik **orang yang berbeda** (aturan Kaggle: satu orang satu akun). Dengan dua akun ada
4 GPU; setiap GPU mendapat satu *shard* (akun 1: shard 1/4 dan 2/4, akun 2: 3/4 dan 4/4). Urutan run
deterministik, jadi shard tidak pernah tumpang tindih. Perkiraan dengan 10.000 iterasi (±2,8 jam/run):

| Ronde | Akun 1 (`ACCOUNT, N_ACCOUNTS = 1, 2`) | Akun 2 (`2, 2`) | Lama |
|---|---|---|---|
| 0 | Sel 1–5, gate; R7.1 harus PASS | Sel 1–5, gate; R7.1 harus PASS | ±20 menit |
| 1 | Save & Run All: sweep 6 run + bicubic | Save & Run All: sweep 5 run | ±8,5 jam |
| Tukar | Output → New Dataset, *share* ke akun 2 | Output → New Dataset, *share* ke akun 1 | |
| 2 | Kedua dataset jadi input, `IMPORT` = kedua folder `runs`; Save & Run All | sama | ±8,5 jam |
| 3 (hanya jika grid diperluas) | Tukar lagi output ronde 2, lalu Save & Run All | sama | ±8,5 jam |
| Akhir | Impor semua output, jalankan sel 1–3, 6, 9, 10 (tanpa GPU) | | ±10 menit |

Catatan:
- Kedua notebook harus memakai **commit yang sama** (cek baris pertama output sel 1) dan lulus gate
  dengan `configs/protocol_lock.json` yang sudah di-commit. Itu menjamin data dan kode identik.
- Pemilihan λ* dihitung dari log validasi sweep yang sama, jadi kedua akun selalu memilih λ* yang
  sama (diuji dengan simulasi dua akun).
- Path dataset input di Kaggle bisa berbentuk `/kaggle/input/datasets/<pemilik>/<nama>/...`; cek
  dengan `!ls -R /kaggle/input | head` sebelum mengisi `IMPORT`.
- Di ronde 1, sel 8 hanya mencetak bahwa ia menunggu sweep akun lain. Itu normal.

### 7.5c Akun 2 menyusul setelah sesi akun 1 selesai

Kalau akun 1 sudah menjalankan ronde 1 sendirian, akun 2 **tidak** memakai `2, 2`. Jatah shard
`2, 2` hanya cocok kalau akun 1 memakai `1, 2`; kalau akun 1 memakai `1, 1` (bawaan notebook),
sebagian run akan dijalankan dua kali dan sebagian lagi tidak pernah dijalankan. Cara yang benar
untuk kedua kasus:

1. **Akun 1:** buka versi notebook yang sudah selesai → *Output* → **New Dataset**, lalu
   *Settings → Sharing* → tambahkan akun 2.
2. **Akun 2:** *File → Import Notebook* → `notebooks/kaggle_runner.ipynb` dari `main`. Panel kanan:
   GPU T4 x2, Internet On, *Add Input* → `anime-faces-waifu2x` **dan** dataset output akun 1.
3. Sel 2: biarkan `ACCOUNT, N_ACCOUNTS = 1, 1`, isi `IMPORT = ["/kaggle/input/<...>/runs"]`
   (cek path dengan `!ls -R /kaggle/input | grep -m5 "runs:"`).
4. Jalankan sel 1–6 secara interaktif. Gate (sel 5) harus lulus dengan R7.1 PASS. Sel 6 mencetak
   rencana: run dari akun 1 harus berstatus `evaluated`, sisanya `todo`.
5. **Save & Run All.** Run hasil impor dilewati; run sweep yang tersisa dibagi ke 2 GPU. Contoh:
   kalau akun 1 sudah menyelesaikan L1 + 5 static, akun 2 menjalankan 5 adaptive (3 + 2 run, ±9,5 jam).
6. Setelah sweep, sel 8 hanya memulai *extend*/*confirm* kalau satu run (`RUN_HOURS`) masih muat
   dalam sisa `SESSION_HOURS`; kalau tidak muat, ia mencetak `not started`. Itu normal. Batas ini
   mencegah sesi melewati 12 jam dan dihentikan Kaggle, yang bisa membuat output hilang.
7. **Ronde berikutnya (confirm, 9 run):** output akun 2 sudah berisi **semua** run sweep (hasil impor
   ikut tersalin). Jadikan dataset dan bagikan ke akun 1. Lalu kedua akun memasukkannya ke `IMPORT`,
   akun 1 memakai `1, 2` dan akun 2 memakai `2, 2`, dan keduanya menjalankan *Save & Run All*
   bersamaan (±9,5 jam). Kalau grid perlu diperluas, ronde ini menjalankan *extend* dulu; tukar
   output sekali lagi seperti di §7.5b.

### 7.6 Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `CUDA out of memory` | Batch 32 terlalu besar (seharusnya tidak untuk `light` di T4) | `batch_size=16` di `configs/base.json` **sebelum lock** |
| `was written with a different config` | Mengubah config lalu melanjutkan run lama | Pakai folder `--runs` baru, atau kembalikan config |
| `runs come from different protocols` | Mencampur run sebelum & sesudah ubah config | Analisis per protokol; jangan campur |
| `[confirm] waiting for N sweep runs` | Sweep belum selesai | Lanjutkan sweep dulu |
| `[confirm] run --stage extend first` | λ* di ujung grid | `run_stage("extend")` (sel 8 sudah melakukannya) |
| `no images found` | Dataset belum ditambahkan sebagai input | Add Input → dataset |
| Sesi mati di tengah run | Batas waktu/koneksi | Jalankan ulang sel yang sama; otomatis resume |

---

## 8. Membaca hasil dan menulis klaim yang jujur

`RESULTS.md` berisi: tabel utama (rata-rata ± simpangan baku 3 seed), tabel sweep (dengan edge
share), tabel hipotesis (Δ, CI 95%, Δ per seed, p Holm, keputusan), semua selisih berpasangan,
kurva validasi, dan contoh gambar (gambar test pertama menurut urutan kunci, **bukan dipilih**).

**Skenario dan cara menuliskannya** (kalimat contoh dalam bahasa Inggris untuk paper):

| Hasil | Yang boleh ditulis |
|---|---|
| H1 & H2 didukung, H3 didukung | "Adaptive edge weighting improved Edge-PSNR-Y over L1 by Δ dB (95% CI [a, b]) and over the static edge loss by Δ dB, consistently across three seeds, without a PSNR-Y loss beyond 0.05 dB." |
| H1 didukung, H2 tidak | "An edge term helps, but weighting it by edge strength gave no further gain once both were tuned on validation data." Ini tetap temuan yang berguna |
| H1 didukung, H3 tidak | "Edge gains came at a cost of Δ dB in PSNR-Y": laporkan sebagai *trade-off* |
| Tidak ada yang didukung | "Under a matched-tuning, multi-seed protocol, neither edge loss improved edge fidelity over L1 on this dataset." *Null result* yang dilaporkan jujur dengan protokol ketat tetap bisa dipublikasikan (misalnya sebagai studi replikasi/negatif) dan tetap menunjukkan kemampuan riset |
| Static menang atas adaptive | Laporkan apa adanya; ini justru menarik |

**Jangan pernah menulis:**
- "state-of-the-art" (tidak dibandingkan dengan metode SR lain pada benchmark standar);
- "novel loss function" (loss gradien sudah ada; yang baru adalah pembobotan dan protokolnya);
- angka dari satu seed sebagai hasil utama;
- "perceptually better" tanpa LPIPS atau studi pengguna;
- "high-resolution anime" kalau sumbernya ternyata hasil waifu2x;
- angka apa pun di CV/README sebelum `RESULTS.md` ada.

---

## 9. Jalur publikasi

### 9.1 Repo publik (wajib, untuk CV/beasiswa; patokan 2026-11-30)

- [ ] **LICENSE.** Kode SwinIR berlisensi Apache-2.0, jadi paling sederhana repo ini juga Apache-2.0.
      GitHub → Add file → Create new file → ketik `LICENSE` → *Choose a license template* → Apache 2.0.
      (Keputusan lisensi ada di tanganmu; ini hanya saran.)
- [ ] **CITATION.cff**: ganti `alias` dengan nama lengkapmu.
- [ ] README bagian *Results* diisi dari `RESULTS.md` (tanpa mengubah angka), plus `fig_samples.png`.
- [ ] `configs/protocol_lock.json` ter-commit **sebelum** commit hasil (urutan commit jadi bukti
      pra-registrasi).
- [ ] Jangan unggah data atau checkpoint ke git (sudah diatur `.gitignore`). Checkpoint final boleh
      dilampirkan di GitHub Release.
- [ ] Deskripsi repo + topics (super-resolution, swinir, pytorch, anime), lalu pin di profil.
- [ ] Opsional: rilis `v1.0` + DOI Zenodo (integrasi GitHub–Zenodo) agar bisa disitasi.

### 9.2 Paper

Pilihan realistis, dari yang paling cepat:

1. **Preprint arXiv** (kategori cs.CV atau eess.IV). Pengirim pertama kali mungkin perlu
   *endorsement*; cek aturan arXiv terbaru.
2. **Konferensi IEEE di Indonesia** (misalnya ICACSIS atau ICITEE) atau **jurnal nasional
   terakreditasi SINTA** di bidang informatika. Periksa CFP, ruang lingkup, biaya, dan indeksasi
   terkini sebelum memilih; hindari jurnal predator.
3. Workshop internasional bertema restorasi gambar atau *negative results*, kalau hasilnya null.

**Struktur paper (6–8 halaman) dan sumber isinya:**

| Bagian | Sumber di repo |
|---|---|
| Abstract | Tabel hipotesis + kalimat skenario §8 |
| Introduction | Masalah blur pada line art; RQ1–RQ3 (EXPERIMENTS §1) |
| Related work | SwinIR & Swin; loss gradien SR (SPSR, Gradient Variance loss); SR anime (waifu2x, Real-ESRGAN anime, AnimeSR, APISR); metrik (SSIM, GMSD, LPIPS) |
| Method | METHOD.md §1–4 |
| Experimental setup | EXPERIMENTS.md §3–5, METHOD.md §5 |
| Results | `RESULTS.md`: tabel utama, tabel sweep, tabel hipotesis, `fig_sweep.png`, `fig_curves.png`, `fig_samples.png` |
| Discussion & limitations | EXPERIMENTS.md §6, AUDIT.md §D, edge share |
| Reproducibility | Link repo, commit hash dari `protocol_lock.json`, readiness report |

Semua referensi di METHOD.md: **ambil BibTeX dari halaman resmi** (arXiv/IEEE/CVF) sebelum
mengirim. Untuk related work anime (Real-ESRGAN, AnimeSR, APISR), verifikasi tahun dan venue-nya.

### 9.3 Untuk CV dan SOP

- **Sekarang (sebelum hasil):** "Refactored a SwinIR super-resolution project into a tested
  package; audited the first version (found a Sobel-kernel error, a 4.77 dB metric bias and test-set
  model selection) and designed a pre-registered, multi-seed ablation of edge-aware losses with an
  automated readiness gate." Tanpa angka hasil.
- **Setelah hasil:** tambahkan satu kalimat dengan Δ Edge-PSNR-Y dan CI-nya, apa pun arahnya.

### 9.4 Jadwal yang disarankan

| Tanggal | Target |
|---|---|
| 6 Okt | Push kode, gate di Kaggle, lock protokol, mulai sweep |
| 7–14 Okt | Sweep (+ perluasan) dan konfirmasi |
| 15–20 Okt | Analisis, README hasil, cek gambar sumber (§3) |
| 21 Okt–15 Nov | Draf paper/preprint |
| 30 Nov | Repo rapi + CV diperbarui (patokan beasiswa) |

---

## 10. FAQ

**Kenapa tidak memakai model `medium` (8 juta parameter) seperti notebook lama?**
20+ run × medium kemungkinan besar tidak muat kuota GPU gratis. Gate mengukur keduanya di GPU
yang sama, jadi keputusan ini bisa dibuktikan dengan angka. Kalau budget cukup, kamu bisa
mengganti preset sebelum lock. Klaim di paper tetap dibatasi pada model yang dilatih.

**Apakah 3 seed cukup?**
Untuk mendeteksi efek besar, ya; untuk efek sangat kecil, tidak. Karena itu ada bootstrap dua
tingkat, syarat semua seed setuju, dan kalimat klaim yang hati-hati.

**Boleh mengubah τ atau λ grid setelah melihat hasil?**
Boleh dianalisis tambahan, tetapi hasil utama tetap memakai nilai terkunci. Perubahan dicatat di
deviation log dan kedua versi dilaporkan.

**Kenapa checkpoint akhir, bukan checkpoint terbaik?**
Memilih checkpoint terbaik berdasarkan validasi menambah satu lagi "pilihan" yang bisa bias, dan
LR yang meluruh ke ~0 membuat akhir training stabil. Semua kondisi diperlakukan sama.

**Bagaimana menambah LPIPS?**
`pip install lpips` di notebook, lalu tambahkan `--lpips` ke perintah `run_ablation.py` / `evaluate.py`.
Bobot jaringan LPIPS diunduh dari internet saat pertama dipakai. LPIPS adalah metrik sekunder.

**Bisakah menjalankan tes di laptop?**
Ya: `python tests/test_core.py` (±2 menit, CPU) dan
`python scripts/readiness_check.py --synthetic 80 --out tmp/readiness --quick` (±5 menit).

---

## 11. Glosarium

| Istilah | Arti |
|---|---|
| Ablation | Eksperimen yang mengubah satu komponen (di sini: loss) untuk melihat pengaruhnya |
| AMP / fp16 | Hitungan setengah presisi di GPU agar lebih cepat; loss dan metrik tetap presisi penuh |
| Bootstrap | Mengambil sampel ulang data (dengan pengembalian) berkali-kali untuk memperkirakan ketidakpastian |
| Checkpoint | Simpanan bobot + optimizer + langkah, untuk melanjutkan training |
| dHash | Sidik jari gambar 64-bit; gambar yang mirip punya dHash sama → dipakai untuk mencegah duplikat lintas split |
| Edge share | Porsi suku tepi dalam total loss di akhir training |
| Endpoint utama | Metrik yang menentukan keputusan hipotesis (Edge-PSNR-Y) |
| Holm | Koreksi p-value untuk beberapa uji sekaligus |
| Non-inferiority | Uji bahwa metode baru "tidak lebih buruk dari X lebih dari margin" |
| Pra-registrasi | Mengunci rencana analisis sebelum melihat hasil |
| Protocol hash | Sidik jari semua setting yang memengaruhi hasil; run dengan hash beda tidak boleh dicampur |
| Readiness gate | Kumpulan cek otomatis yang wajib lulus sebelum eksperimen mahal |
| Seed | Angka awal pembangkit acak; menentukan bobot awal, urutan data, dan augmentasi |
| Shard | Potongan daftar run, untuk dijalankan paralel di beberapa GPU/sesi |
| Wilcoxon signed-rank | Uji non-parametrik untuk selisih berpasangan |
| Y channel | Kanal kecerahan dari ruang warna YCbCr |
